"""
retriever.py
==============================================================================
Privacy-preserving dense retriever for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

Pipeline
--------
1. Encode documents and queries with ``all-MiniLM-L6-v2`` (CPU only,
   L2-normalised so dot product == cosine similarity).
2. Build a FAISS ``IndexFlatL2`` index over document embeddings.
3. For each query, retrieve a candidate pool with FAISS, then *re-rank* using

       privacy_score = cosine_similarity(q, d) - lambda * InfoNCE_risk(q, d)

   where the InfoNCE risk approximates how easily a candidate can be singled
   out from the corpus (higher risk -> higher re-identification leakage ->
   stronger penalty).

InfoNCE definition
------------------
Following the contrastive view, with temperature tau (default 0.07):

    risk_i = log( sum_j exp( sim(q, c_j) / tau ) ) - sim(q, c_i) / tau

Computed with a numerically stable log-sum-exp. The corpus acts as the
in-batch negatives approximation; this is exactly the InfoNCE denominator
minus the candidate's positive logit.

Constraints
-----------
* CPU only, < 1 GB peak RAM, deterministic seeding (random/numpy/torch == 42).
* FAISS IndexFlatL2 ONLY (no HNSW/IVF approximations).
* No network at retrieval time. The model is loaded once from local cache.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

# ----------------------------------------------------------------------------
# Reproducibility (mandated global rule)
# ----------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
try:
    import torch  # noqa: F401
    torch.manual_seed(42)
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# ----------------------------------------------------------------------------
# Required deps (assumed installed)
# ----------------------------------------------------------------------------
import faiss  # type: ignore[import-not-found]
from encoder_backends import StableTextEncoder

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("retriever")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )

# ----------------------------------------------------------------------------
# Console-safe success marker
# ----------------------------------------------------------------------------
def _ok(msg: str) -> None:
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    mark = "\u2714" if "utf" in enc else "[OK]"
    try:
        print(f"{mark} {msg}")
    except UnicodeEncodeError:  # pragma: no cover - defensive
        print(f"[OK] {msg}")


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


# ----------------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    rank: int
    doc_id: int
    text: str
    cosine: float
    infonce_risk: float
    privacy_score: float
    l2_distance: float


# ----------------------------------------------------------------------------
# PrivacyRetriever
# ----------------------------------------------------------------------------
class PrivacyRetriever:
    """FAISS IndexFlatL2 retriever with InfoNCE-based privacy re-ranking.

    Parameters
    ----------
    model_name : str
        HuggingFace SentenceTransformer model id (default ``all-MiniLM-L6-v2``).
    temperature : float
        InfoNCE temperature (default 0.07 per the spec).
    lambda_privacy : float
        Privacy-penalty coefficient lambda; final score = cos - lambda * risk.
    normalize : bool
        L2-normalise embeddings (so dot product == cosine).
    model : Optional[StableTextEncoder]
        Optional pre-loaded encoder (used for dependency injection / tests).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.07,
        lambda_privacy: float = 0.5,
        normalize: bool = True,
        model: Optional[StableTextEncoder] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if lambda_privacy < 0:
            raise ValueError("lambda_privacy must be >= 0")

        self.model_name = model_name
        self.temperature = float(temperature)
        self.lambda_privacy = float(lambda_privacy)
        self.normalize = bool(normalize)

        self._model: Optional[StableTextEncoder] = model
        self._index: Optional[faiss.Index] = None
        self._embeddings: Optional[np.ndarray] = None
        self._docs: List[str] = []
        self._dim: int = 0

    # ------------------------------------------------------------------ #
    # Encoder (lazy load)
    # ------------------------------------------------------------------ #
    @property
    def model(self) -> StableTextEncoder:
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on CPU")
            self._model = StableTextEncoder(
                self.model_name,
                device="cpu",
                local_files_only=True,
                n_features=EMBED_DIM,
            )
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of texts as float32 (N, D) embeddings."""
        if texts is None:
            raise ValueError("texts must not be None")
        if len(texts) == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        emb = self.model.encode(
            list(texts),
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        ).astype(np.float32, copy=False)
        if emb.ndim == 1:
            emb = emb[None, :]
        return np.ascontiguousarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Index
    # ------------------------------------------------------------------ #
    def build_index(self, documents: Sequence[str]) -> None:
        """Encode documents and build a FAISS IndexFlatL2 over them."""
        if not documents:
            raise ValueError("documents must be a non-empty sequence")
        docs = [str(d) for d in documents]
        emb = self.encode(docs)
        if emb.shape[0] != len(docs):
            raise RuntimeError(
                f"encoder returned {emb.shape[0]} rows for {len(docs)} docs"
            )
        self._docs = docs
        self._embeddings = emb
        self._dim = int(emb.shape[1])
        index = faiss.IndexFlatL2(self._dim)
        index.add(emb)
        self._index = index
        logger.info(f"Built FAISS IndexFlatL2 with N={index.ntotal}, D={self._dim}")

    # ------------------------------------------------------------------ #
    # InfoNCE risk
    # ------------------------------------------------------------------ #
    def infonce_score(
        self,
        query_emb: np.ndarray,
        cand_emb: np.ndarray,
        corpus_emb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Per-candidate InfoNCE re-identification risk.

        risk_i = LSE_j( sim(q, corpus_j)/tau ) - sim(q, c_i)/tau

        Where ``sim`` is dot product (== cosine when inputs are normalised).
        Implemented with numerically stable log-sum-exp.

        Returns
        -------
        np.ndarray, shape (M,), dtype float32, where M = #candidates.
        """
        q = np.atleast_2d(np.asarray(query_emb, dtype=np.float32))
        c = np.atleast_2d(np.asarray(cand_emb, dtype=np.float32))
        if q.shape[0] != 1:
            raise ValueError("infonce_score expects a single query (shape (1, D))")
        if c.shape[1] != q.shape[1]:
            raise ValueError(
                f"dim mismatch: query D={q.shape[1]} vs candidates D={c.shape[1]}"
            )

        if corpus_emb is None:
            if self._embeddings is None:
                # Fall back to candidates as their own corpus.
                K = c
            else:
                K = self._embeddings
        else:
            K = np.atleast_2d(np.asarray(corpus_emb, dtype=np.float32))
        if K.shape[1] != q.shape[1]:
            raise ValueError(
                f"dim mismatch: query D={q.shape[1]} vs corpus D={K.shape[1]}"
            )

        tau = self.temperature
        logits_qK = (q @ K.T) / tau               # (1, N)
        logits_qc = (q @ c.T) / tau               # (1, M)

        # numerically stable log-sum-exp over corpus
        m = logits_qK.max(axis=1, keepdims=True)  # (1, 1)
        lse = m + np.log(np.exp(logits_qK - m).sum(axis=1, keepdims=True))  # (1, 1)

        risk = (lse - logits_qc).squeeze(0)       # (M,)
        return risk.astype(np.float32, copy=False)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: Union[str, Sequence[str]],
        top_k: int = 5,
        candidate_pool: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k documents under the privacy-aware score.

        1. Encode the query.
        2. FAISS L2 search over a candidate pool (default 4 * top_k).
        3. Re-rank by ``cosine - lambda * infonce_risk``.
        """
        if self._index is None or self._embeddings is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if isinstance(query, str):
            queries = [query]
        else:
            queries = list(query)
            if len(queries) != 1:
                raise ValueError(
                    "PrivacyRetriever.retrieve currently supports a single query"
                )

        q_emb = self.encode(queries)              # (1, D)
        N = self._embeddings.shape[0]
        pool = min(max(candidate_pool or top_k * 4, top_k), N)

        D, I = self._index.search(q_emb, pool)    # FAISS L2 search
        cand_idx = I[0]
        cand_l2 = D[0]
        # Filter padding indices (-1) which can occur if pool > ntotal in some
        # FAISS builds; defensive only.
        valid = cand_idx >= 0
        cand_idx = cand_idx[valid]
        cand_l2 = cand_l2[valid]

        cand_emb = self._embeddings[cand_idx]
        cosine = (q_emb @ cand_emb.T).squeeze(0)  # equiv to dot for normalised
        risk = self.infonce_score(q_emb, cand_emb)
        score = cosine - self.lambda_privacy * risk

        order = np.argsort(-score)[: min(top_k, len(score))]
        results: List[RetrievalResult] = []
        for rank, j in enumerate(order, start=1):
            di = int(cand_idx[j])
            results.append(
                RetrievalResult(
                    rank=rank,
                    doc_id=di,
                    text=self._docs[di],
                    cosine=float(cosine[j]),
                    infonce_risk=float(risk[j]),
                    privacy_score=float(score[j]),
                    l2_distance=float(cand_l2[j]),
                )
            )
        return results


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * Embedding dimension and normalisation invariants.
#   * FAISS IndexFlatL2 build-and-query round trip.
#   * InfoNCE risk shape, finiteness, and non-negativity for in-corpus
#     candidates (LSE bound).
#   * retrieve() returns non-empty, score-sorted, top-k results.
#   * lambda=0 reduces to pure cosine ranking.
# ============================================================================
def _self_test() -> None:
    docs = [
        "Backpropagation computes gradients through the chain rule.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "FAISS provides efficient similarity search for dense vectors.",
        "Bloom's taxonomy categorises cognitive learning objectives.",
        "Differential privacy adds calibrated noise to protect individuals.",
        "Transformers use self-attention to process token sequences.",
        "InfoNCE is a contrastive loss that maximises mutual information.",
        "Retrieval-augmented generation grounds LLMs in external knowledge.",
    ]

    r = PrivacyRetriever(temperature=0.07, lambda_privacy=0.3)

    # 1. Embedding dimension and normalisation -------------------------------
    emb = r.encode(["hello world"])
    assert emb.shape == (1, EMBED_DIM), f"bad embedding shape: {emb.shape}"
    assert emb.dtype == np.float32, f"bad dtype: {emb.dtype}"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"embeddings not normalised: {norms}"

    # encode([]) returns a (0, D) zero array
    empty = r.encode([])
    assert empty.shape == (0, EMBED_DIM)

    # 2. Index build / FAISS query ------------------------------------------
    r.build_index(docs)
    assert r._index is not None and r._index.ntotal == len(docs)
    assert r._embeddings is not None
    assert r._embeddings.shape == (len(docs), EMBED_DIM)
    assert r._dim == EMBED_DIM

    q_emb = r.encode(["self-attention transformer architecture"])
    D_, I_ = r._index.search(q_emb, 3)
    assert D_.shape == (1, 3) and I_.shape == (1, 3)
    assert all(0 <= int(i) < len(docs) for i in I_[0])

    # 3. InfoNCE shape / numerical sanity -----------------------------------
    risk = r.infonce_score(q_emb, r._embeddings)
    assert risk.shape == (len(docs),), f"risk shape: {risk.shape}"
    assert np.all(np.isfinite(risk)), "risk contains non-finite values"
    # For in-corpus candidates: LSE >= each logit, so risk >= 0 (allow tiny eps)
    assert np.all(risk >= -1e-4), f"in-corpus risk should be >= 0, got min={risk.min()}"

    # determinism: same call twice yields the same risk vector
    risk2 = r.infonce_score(q_emb, r._embeddings)
    assert np.allclose(risk, risk2), "infonce_score is not deterministic"

    # dimension mismatch raises
    try:
        r.infonce_score(q_emb, np.zeros((2, EMBED_DIM + 1), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on dim mismatch")

    # 4. retrieve() returns ranked, non-empty results -----------------------
    out = r.retrieve("self-attention transformer architecture", top_k=3)
    assert len(out) == 3, f"expected 3 results, got {len(out)}"
    scores = [o.privacy_score for o in out]
    assert scores == sorted(scores, reverse=True), (
        f"results not sorted by privacy_score: {scores}"
    )
    # The transformer doc should appear in the top-3 for this query
    assert any(
        ("Transformer" in o.text) or ("self-attention" in o.text) for o in out
    ), f"expected transformer-related top hit, got: {[o.text for o in out]}"

    # 5. lambda=0 reduces to pure cosine -----------------------------------
    r0 = PrivacyRetriever(temperature=0.07, lambda_privacy=0.0, model=r.model)
    r0.build_index(docs)
    pure = r0.retrieve("photosynthesis chlorophyll plants", top_k=1)[0]
    assert ("Photosynthesis" in pure.text) or ("plants" in pure.text), (
        f"pure cosine top-1 wrong: {pure.text}"
    )

    # 6. retrieve before build_index -> RuntimeError ------------------------
    r_empty = PrivacyRetriever(model=r.model)
    try:
        r_empty.retrieve("anything")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError when index not built")

    # 7. invalid hyperparameters -------------------------------------------
    for kwargs in (
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"lambda_privacy": -0.1},
    ):
        try:
            PrivacyRetriever(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for kwargs={kwargs}")

    _ok("retriever.py sanity check passed")


if __name__ == "__main__":
    _self_test()
