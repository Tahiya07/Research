"""
summarizer.py
==============================================================================
Phase-4 cognitive-aware summarisation for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

End-to-end Cognitive-Aware RAG pipeline
---------------------------------------

    query
      |
      v
    retriever.PrivacyRetriever  ----->  top-k context chunks (Phase 1)
      |
      v
    classifier.BloomLDLClassifier -->  Bloom distribution + dominant level
      |                                 (Phase 3)
      v
    style adapter (this module)        choose summary style:
      |                                  remember/understand -> factual
      |                                  apply/analyze       -> reasoning
      v                                  evaluate/create     -> comparative
    models.RAGGenerator         ----->  Bloom-conditioned summary (Phase 2)

The summariser does **not** modify any earlier-phase module. It composes
them and adds:

* Style-conditioned re-formulation of the user's query.
* Optional hierarchical (per-chunk -> synthesis) summarisation, useful for
  high cognitive levels where comparison / synthesis matters.
* Structured output containing the final summary, the Bloom level used,
  the underlying distribution and confidence (when auto-inferred), the
  retrieved chunks, and the literal prompt for downstream auditing.

Constraints
-----------
* CPU only, < 1 GB RAM peak.
* Deterministic generation (greedy + seed=42, inherited from RAGGenerator).
* No external APIs.
"""

from __future__ import annotations

import logging
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

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
# Phase-1/2/3 modules (imported, not modified)
# ----------------------------------------------------------------------------
from retriever import PrivacyRetriever, RetrievalResult
from models import RAGGenerator, GenerationOutput, BLOOM_INSTRUCTIONS
from classifier import (
    BloomLDLClassifier,
    BLOOM_LEVELS,
    BLOOM_INDEX,
    ClassifierOutput,
    DEFAULT_WEIGHTS_PATH as DEFAULT_CLASSIFIER_WEIGHTS,
)

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("summarizer")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


def _ok(msg: str) -> None:
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    mark = "\u2714" if "utf" in enc else "[OK]"
    try:
        print(f"{mark} {msg}")
    except UnicodeEncodeError:  # pragma: no cover
        print(f"[OK] {msg}")


# ----------------------------------------------------------------------------
# Bloom-level -> summary style mapping
# ----------------------------------------------------------------------------
SUMMARY_STYLE: Dict[str, str] = {
    "remember":   "factual",
    "understand": "factual",
    "apply":      "reasoning",
    "analyze":    "reasoning",
    "evaluate":   "comparative",
    "create":     "comparative",
}

STYLE_INSTRUCTIONS: Dict[str, str] = {
    "factual": (
        "Produce a short, fact-based summary that lists the key facts directly "
        "supported by the context. Use plain language and avoid speculation."
    ),
    "reasoning": (
        "Produce a step-by-step reasoning summary that decomposes the topic "
        "into its main parts and explains how the context supports each step."
    ),
    "comparative": (
        "Produce a comparative / evaluative summary that contrasts perspectives, "
        "identifies trade-offs, and weighs the evidence drawn from the context."
    ),
}


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------
@dataclass
class SummaryOutput:
    summary: str
    used_bloom_level: str            # canonical lowercase Bloom key
    style: str                       # factual | reasoning | comparative
    bloom_distribution: Optional[np.ndarray] = None  # (6,) when auto-inferred
    confidence: Optional[float] = None               # 0..1 when auto-inferred
    chunks: List[RetrievalResult] = field(default_factory=list)
    prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# CognitiveSummarizer
# ----------------------------------------------------------------------------
class CognitiveSummarizer:
    """Cognitive-Aware RAG summariser composing Phases 1, 2, and 3."""

    def __init__(
        self,
        retriever: PrivacyRetriever,
        generator: RAGGenerator,
        classifier: Optional[BloomLDLClassifier] = None,
        hierarchical: bool = False,
        per_chunk_max_tokens: int = 64,
    ) -> None:
        if not isinstance(retriever, PrivacyRetriever):
            raise TypeError("retriever must be a PrivacyRetriever instance")
        if not isinstance(generator, RAGGenerator):
            raise TypeError("generator must be a RAGGenerator instance")
        if classifier is not None and not isinstance(classifier, BloomLDLClassifier):
            raise TypeError("classifier must be a BloomLDLClassifier instance")
        if per_chunk_max_tokens <= 0:
            raise ValueError("per_chunk_max_tokens must be > 0")

        self.retriever = retriever
        self.generator = generator
        self.classifier = classifier
        self.hierarchical = bool(hierarchical)
        self.per_chunk_max_tokens = int(per_chunk_max_tokens)

    # ------------------------------------------------------------------ #
    # Bloom inference
    # ------------------------------------------------------------------ #
    def _infer_bloom(
        self, query: str
    ) -> tuple[str, Optional[np.ndarray], Optional[float]]:
        """Return (bloom_level_lc, distribution, confidence)."""
        if self.classifier is None:
            return "understand", None, None
        out: ClassifierOutput = self.classifier.predict(query)
        return out.dominant_level.lower(), out.distribution, out.confidence

    @staticmethod
    def _validate_bloom(bloom: str) -> str:
        key = bloom.strip().lower()
        if key not in BLOOM_INSTRUCTIONS:
            raise ValueError(
                f"unknown bloom_level: {bloom!r}; "
                f"expected one of {list(BLOOM_INSTRUCTIONS)}"
            )
        return key

    # ------------------------------------------------------------------ #
    # Style adapters
    # ------------------------------------------------------------------ #
    @staticmethod
    def _style_for(bloom_lc: str) -> str:
        return SUMMARY_STYLE[bloom_lc]

    @staticmethod
    def _style_query(query: str, bloom_lc: str, style: str) -> str:
        """Wrap the user's query in a style-conditioned summarisation request."""
        return (
            f"Summarise the topic of the following query for a learner at the "
            f"'{bloom_lc}' Bloom level. {STYLE_INSTRUCTIONS[style]}\n\n"
            f"Query: {query.strip()}"
        )

    # ------------------------------------------------------------------ #
    # Optional: hierarchical pre-summarisation
    # ------------------------------------------------------------------ #
    def _prebuild_hierarchical_context(
        self,
        query: str,
        chunks: Sequence[RetrievalResult],
        bloom_lc: str,
    ) -> List[str]:
        """Generate a one-sentence pre-summary per chunk.

        Used only when ``hierarchical=True``. We hijack the generator to
        produce short per-chunk distillations, then the final synthesis
        step (the regular summarise call) can compare them.
        """
        condensed: List[str] = []
        prev_ctx_max = self.generator.max_tokens
        try:
            self.generator.max_tokens = self.per_chunk_max_tokens
            for c in chunks:
                # Build a tiny one-chunk prompt body.
                body = self.generator.build_prompt(
                    query=f"In one sentence, summarise this passage in the "
                          f"context of: {query}",
                    chunks=[c],
                    bloom_level=bloom_lc,
                )
                chatml = self.generator._to_chatml(body)
                reset_fn = getattr(self.generator.llm, "reset", None)
                if callable(reset_fn):
                    try:
                        reset_fn()
                    except Exception:  # pragma: no cover
                        pass
                out = self.generator.llm(
                    chatml,
                    max_tokens=self.per_chunk_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    repeat_penalty=1.1,
                    stop=["<|im_end|>", "<|im_start|>"],
                    echo=False,
                    seed=self.generator.seed,
                )
                try:
                    txt = out["choices"][0]["text"].strip()
                except (KeyError, IndexError, TypeError):
                    txt = ""
                condensed.append(txt or c.text)
        finally:
            self.generator.max_tokens = prev_ctx_max
        return condensed

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def summarize(
        self,
        query: str,
        bloom_level: Optional[str] = None,
        k: int = 3,
        max_tokens: int = 160,
    ) -> SummaryOutput:
        """Run the full Cognitive-Aware RAG pipeline.

        Parameters
        ----------
        query : str
            User query.
        bloom_level : Optional[str]
            Override the Bloom level. If None, the Phase-3 classifier infers
            it from the query (if available); else defaults to ``understand``.
        k : int
            Number of context chunks to retrieve.
        max_tokens : int
            Generation budget for the final summary.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")

        # 1. Bloom inference (or honour caller override)
        if bloom_level is None:
            bloom_lc, dist, conf = self._infer_bloom(query)
            auto = True
        else:
            bloom_lc = self._validate_bloom(bloom_level)
            dist, conf, auto = None, None, False

        style = self._style_for(bloom_lc)

        # 2. Retrieve context (Phase 1)
        chunks = self.retriever.retrieve(query, top_k=k)

        # 3. Optional hierarchical pre-summarisation for high cognitive levels
        hier_summaries: Optional[List[str]] = None
        if self.hierarchical and len(chunks) > 1:
            hier_summaries = self._prebuild_hierarchical_context(
                query, chunks, bloom_lc
            )
            # Inject the per-chunk pre-summaries by overwriting chunk text
            # only for the prompt step. We construct lightweight wrappers
            # so RAGGenerator.build_prompt can still call .text on them.
            chunks = [
                RetrievalResult(
                    rank=c.rank,
                    doc_id=c.doc_id,
                    text=hier_summaries[i],
                    cosine=c.cosine,
                    infonce_risk=c.infonce_risk,
                    privacy_score=c.privacy_score,
                    l2_distance=c.l2_distance,
                )
                for i, c in enumerate(chunks)
            ]

        # 4. Build a style-conditioned summarisation query.
        style_query = self._style_query(query, bloom_lc, style)

        # 5. Compose the final prompt manually so we can:
        #    - keep the Phase-2 prompt skeleton intact,
        #    - reuse the (possibly hierarchical) context chunks above.
        prompt = self.generator.build_prompt(
            query=style_query,
            chunks=chunks,
            bloom_level=bloom_lc,
        )
        chatml = self.generator._to_chatml(prompt)

        # 6. Run the LLM (CPU, deterministic), with a temporary max_tokens swap.
        reset_fn = getattr(self.generator.llm, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:  # pragma: no cover
                pass

        t0 = time.time()
        out = self.generator.llm(
            chatml,
            max_tokens=int(max_tokens),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
            seed=self.generator.seed,
        )
        elapsed = time.time() - t0
        try:
            summary = out["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected llm output: {out!r}") from e

        return SummaryOutput(
            summary=summary,
            used_bloom_level=bloom_lc,
            style=style,
            bloom_distribution=dist if auto else None,
            confidence=conf if auto else None,
            chunks=list(chunks),
            prompt=prompt,
            metadata={
                "auto_bloom": auto,
                "k": k,
                "max_tokens": int(max_tokens),
                "hierarchical": self.hierarchical,
                "elapsed_s": round(elapsed, 3),
                "model_path": getattr(self.generator, "model_path", "<?>"),
                "style_query": style_query,
            },
        )


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * Cognitive-Aware RAG composition with all phases live (no mocks):
#       PrivacyRetriever (FAISS+InfoNCE) + BloomLDLClassifier (LDL)
#       + RAGGenerator (Qwen-1.5B GGUF, greedy CPU).
#   * Auto-inferred Bloom level is a valid lowercase key, distribution is
#     a valid (6,) probability vector summing to 1.
#   * Style mapping yields one of {factual, reasoning, comparative}.
#   * Returned summary is non-empty and grounded in retrieved context.
#   * Determinism: re-running the same query yields the same summary.
#   * Manual override of bloom_level disables classifier outputs.
#   * Constructor input validation rejects bad components.
# ============================================================================
def _self_test() -> None:
    from pathlib import Path

    docs = [
        "Photosynthesis is the biological process by which green plants convert "
        "sunlight, water, and carbon dioxide into glucose and oxygen using "
        "chlorophyll inside chloroplasts.",
        "Backpropagation computes gradients through the chain rule for training "
        "neural networks via gradient descent.",
        "FAISS provides efficient similarity search and clustering of dense "
        "vectors at scale.",
        "Differential privacy adds calibrated noise so that individuals cannot "
        "be re-identified from aggregate statistics.",
        "Bloom's taxonomy categorises cognitive learning objectives into "
        "Remember, Understand, Apply, Analyze, Evaluate, and Create.",
    ]

    # --- Phase-1 retriever -------------------------------------------------
    retr = PrivacyRetriever(temperature=0.07, lambda_privacy=0.1)
    retr.build_index(docs)

    # --- Phase-3 classifier (load cached weights produced by classifier.py)
    weights_path = Path(DEFAULT_CLASSIFIER_WEIGHTS)
    assert weights_path.is_file(), (
        f"classifier weights not found at {weights_path}; "
        "run `python classifier.py` first."
    )
    # Reuse the retriever's encoder to avoid loading MiniLM twice.
    classifier = BloomLDLClassifier.load(weights_path, encoder=retr.model)

    # --- Phase-2 generator -------------------------------------------------
    rag = RAGGenerator(
        retriever=retr,
        n_ctx=1024,
        max_tokens=160,
        seed=42,
    )

    # ------------------------------------------------------------------ #
    # Constructor validation (no LLM calls yet)
    # ------------------------------------------------------------------ #
    summ = CognitiveSummarizer(retriever=retr, generator=rag, classifier=classifier)

    for kwargs, exc in (
        ({"retriever": "x", "generator": rag, "classifier": classifier}, TypeError),
        ({"retriever": retr, "generator": "x", "classifier": classifier}, TypeError),
        ({"retriever": retr, "generator": rag, "classifier": "x"}, TypeError),
        ({"retriever": retr, "generator": rag, "classifier": classifier,
          "per_chunk_max_tokens": 0}, ValueError),
    ):
        try:
            CognitiveSummarizer(**kwargs)  # type: ignore[arg-type]
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__} for kwargs={list(kwargs)}")

    for bad_q in ("", "   "):
        try:
            summ.summarize(bad_q)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for empty query")
    for bad_kw in ({"k": 0}, {"max_tokens": 0}):
        try:
            summ.summarize("ok?", **bad_kw)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad_kw}")
    try:
        summ.summarize("ok?", bloom_level="not-a-level")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad bloom_level")

    # ------------------------------------------------------------------ #
    # End-to-end auto-inferred run
    # ------------------------------------------------------------------ #
    res = summ.summarize("What is photosynthesis?", k=3, max_tokens=96)
    assert isinstance(res, SummaryOutput)
    assert isinstance(res.summary, str) and res.summary.strip(), (
        f"empty summary: {res.summary!r}"
    )
    assert res.used_bloom_level in BLOOM_INSTRUCTIONS, (
        f"bad bloom level: {res.used_bloom_level}"
    )
    assert res.style in ("factual", "reasoning", "comparative"), res.style
    assert SUMMARY_STYLE[res.used_bloom_level] == res.style, (
        f"style/bloom mapping inconsistent: {res.used_bloom_level} -> {res.style}"
    )
    assert res.bloom_distribution is not None
    assert res.bloom_distribution.shape == (6,)
    assert np.isclose(res.bloom_distribution.sum(), 1.0, atol=1e-5)
    assert res.confidence is not None and 0.0 <= res.confidence <= 1.0
    assert len(res.chunks) == 3, f"expected 3 chunks, got {len(res.chunks)}"
    for marker in (
        "[BOUNDED CONTEXT]",
        "[QUESTION]",
        "[COGNITIVE LEVEL]",
        "[INSTRUCTION]",
    ):
        assert marker in res.prompt, f"prompt missing marker {marker}"

    # Grounding: at least one context keyword appears in the summary.
    grounded = ("photo", "plant", "chloro", "sunlight",
                "glucose", "oxygen", "water", "carbon")
    assert any(g in res.summary.lower() for g in grounded), (
        f"summary appears ungrounded in context: {res.summary!r}"
    )

    # ------------------------------------------------------------------ #
    # Determinism (same input -> identical output)
    # ------------------------------------------------------------------ #
    res2 = summ.summarize("What is photosynthesis?", k=3, max_tokens=96)
    assert res2.summary == res.summary, (
        "Cognitive summariser is not deterministic:\n"
        f"  first : {res.summary!r}\n"
        f"  second: {res2.summary!r}"
    )

    # ------------------------------------------------------------------ #
    # Manual override path (auto_bloom=False, classifier outputs cleared)
    # ------------------------------------------------------------------ #
    res3 = summ.summarize(
        "What is photosynthesis?",
        bloom_level="evaluate",
        k=2,
        max_tokens=64,
    )
    assert res3.used_bloom_level == "evaluate"
    assert res3.style == "comparative"
    assert res3.bloom_distribution is None
    assert res3.confidence is None
    assert res3.metadata["auto_bloom"] is False
    assert res3.summary.strip(), "manual-override produced empty summary"

    logger.info(
        "summary latency: auto=%.2fs, det=%.2fs, manual=%.2fs",
        res.metadata["elapsed_s"],
        res2.metadata["elapsed_s"],
        res3.metadata["elapsed_s"],
    )
    logger.info(
        "auto-inferred bloom=%s (style=%s, conf=%.2f)",
        res.used_bloom_level, res.style, res.confidence,
    )

    _ok("CognitiveSummarizer (Phase 4) sanity check passed")


if __name__ == "__main__":
    _self_test()
