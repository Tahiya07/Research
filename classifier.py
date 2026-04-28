"""
classifier.py
==============================================================================
Phase-3 cognitive-level classifier for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

Approach: Label Distribution Learning (LDL) with ordinal Gaussian smoothing
on top of frozen ``all-MiniLM-L6-v2`` embeddings.

Bloom's revised taxonomy (ordinal, low -> high cognitive demand):

    [Remember, Understand, Apply, Analyze, Evaluate, Create]

Soft-label construction
-----------------------
For a training example annotated with level L* in {0, ..., 5}, the target
distribution is a discrete Gaussian over the level grid::

    t_i  =  exp(-(i - L*)^2 / (2 sigma^2))
    T    =  t / sum(t)

This explicitly encodes the *ordinal* prior that "Apply" is closer to
"Analyze" than to "Remember" -- exactly the behaviour the spec asks for.

Model
-----
A single linear projection ``W in R^{384 x 6}`` (plus bias) is trained to
match these target distributions in KL divergence::

    P  =  softmax(X W + b)
    L  =  KL(T || P) + (l2/2) ||W||^2

Optimised with deterministic full-batch gradient descent in pure NumPy
(no SGD ordering noise -> bit-identical reruns).

Outputs
-------
* ``distribution`` : (6,) probability vector
* ``dominant_level`` : argmax label (string)
* ``confidence`` : 1 - H(p)/log(K) in [0, 1] (entropy-based)
* ``entropy`` : natural-log predictive entropy

Constraints
-----------
* CPU only, < 1 GB RAM (encoder ~90 MB, weights ~10 KB).
* Deterministic seeding: random / numpy / torch == 42.
* Phase-1 / Phase-2 modules are NOT modified.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
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

from encoder_backends import StableTextEncoder

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("classifier")
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
# Constants
# ----------------------------------------------------------------------------
BLOOM_LEVELS: List[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]
BLOOM_INDEX: Dict[str, int] = {k.lower(): i for i, k in enumerate(BLOOM_LEVELS)}
EMBED_DIM = 384

DEFAULT_OBE_SEARCH_PATHS: List[str] = [
    "./data/obe_dataset.csv",
    "./obe_dataset.csv",
    "~/PycharmProjects/Thesis/data/obe_dataset.csv",
    "~/Documents/obe_dataset.csv",
]
DEFAULT_WEIGHTS_PATH = "./models/bloom_ldl_weights.npz"

DEFAULT_FIGSHARE_SEARCH_PATHS: List[str] = [
    "./data/figshare_combined_dataset.csv",
    "./data/exam_combined_dataset.csv",
    "./data/datasets/exam_combined_dataset.csv",
    "./exam_combined_dataset.csv",
    "~/PycharmProjects/Thesis/models/external_datasets/exam_combined_dataset.csv",
    "~/Documents/exam_combined_dataset.csv",
]

# Map common Bloom label variants (incl. original Bloom 1956) to our canonical
# revised taxonomy ordering used throughout the codebase.
_BLOOM_ALIASES: Dict[str, str] = {
    # canonical
    "remember": "Remember",
    "understand": "Understand",
    "apply": "Apply",
    "analyze": "Analyze",
    "evaluate": "Evaluate",
    "create": "Create",
    # 1956 Bloom -> revised
    "knowledge": "Remember",
    "comprehension": "Understand",
    "application": "Apply",
    "analysis": "Analyze",
    "synthesis": "Create",
    "evaluation": "Evaluate",
    # common gerunds / spelling
    "remembering": "Remember",
    "understanding": "Understand",
    "applying": "Apply",
    "analysing": "Analyze",
    "analyzing": "Analyze",
    "evaluating": "Evaluate",
    "creating": "Create",
}


def _normalise_bloom(label: object) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s:
        return None
    return _BLOOM_ALIASES.get(s)


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


# ----------------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------------
def _find_obe_dataset(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    env = os.environ.get("OBE_DATASET_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for s in DEFAULT_OBE_SEARCH_PATHS:
        p = Path(s).expanduser()
        if p.is_file():
            return p
    return None


def load_obe_dataset(
    path: Optional[Union[str, Path]] = None,
    max_per_class: int = 1000,
    text_field: str = "question",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Stratified subsample of the OBE dataset.

    Parameters
    ----------
    path : optional explicit CSV path; otherwise auto-discovered.
    max_per_class : cap per Bloom level (deterministic subsample).
    text_field : column to use as input text. ``question`` is the most
        Bloom-discriminative field by construction.
    seed : RNG seed for the stratified subsample.

    Returns
    -------
    (texts, labels) where labels are canonical-cased Bloom strings.
    """
    p = _find_obe_dataset(str(path) if path else None)
    if p is None:
        raise FileNotFoundError(
            "OBE dataset not found. Set OBE_DATASET_PATH or pass path=... "
            f"Searched: {DEFAULT_OBE_SEARCH_PATHS}"
        )
    import pandas as pd  # local import to avoid hard dep at module load

    bloom_candidates = ("bloom_level", "bloom")
    cognitive_candidates = ("cognitive_skill", "cognitive")
    usecols = [text_field, *bloom_candidates]
    try:
        head = pd.read_csv(p, nrows=1)
        cols = {str(c).strip().lower(): c for c in head.columns}
        for cand in cognitive_candidates:
            if cand in cols:
                usecols.append(cols[cand])
        for cand in bloom_candidates:
            if cand in cols:
                usecols.append(cols[cand])
    except Exception:
        pass
    df = pd.read_csv(p, usecols=lambda c: c in set(usecols), low_memory=False)
    bloom_col = "bloom_level" if "bloom_level" in df.columns else ("bloom" if "bloom" in df.columns else None)
    if bloom_col is None:
        raise RuntimeError(f"OBE dataset at {p} missing a Bloom label column (expected bloom_level or bloom)")
    cog_col = "cognitive_skill" if "cognitive_skill" in df.columns else ("cognitive" if "cognitive" in df.columns else None)
    df = df.dropna(subset=[text_field, bloom_col])
    df["bloom_level"] = (
        df[bloom_col].astype(str).str.strip().str.capitalize()
    )
    df = df[df["bloom_level"].isin(BLOOM_LEVELS)]
    df[text_field] = df[text_field].astype(str).str.strip()
    df = df[df[text_field].str.len() > 0]
    if len(df) == 0:
        raise RuntimeError(f"OBE dataset at {p} produced 0 rows after cleaning")

    rng = np.random.default_rng(seed)
    parts = []
    for lvl in BLOOM_LEVELS:
        sub = df[df["bloom_level"] == lvl]
        if len(sub) == 0:
            continue
        n = min(len(sub), int(max_per_class))
        idx = rng.choice(len(sub), size=n, replace=False)
        parts.append(sub.iloc[idx])

    out = (
        __import__("pandas").concat(parts)
        .sample(frac=1.0, random_state=int(seed))
        .reset_index(drop=True)
    )
    raw_q = out[text_field].astype(str).tolist()
    if cog_col and cog_col in out.columns:
        cog = out[cog_col].astype(str).str.strip()
        texts = [
            f"[cognitive={c}] {q}".strip() if c and c.lower() != "nan" else q
            for q, c in zip(raw_q, cog)
        ]
    else:
        texts = raw_q
    labels = out["bloom_level"].astype(str).tolist()
    logger.info(
        f"OBE dataset: {len(texts)} rows ({max_per_class}/class) loaded from {p}"
    )
    return texts, labels


def _find_figshare_exam_dataset(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the Figshare-style Bloom exam dataset on disk."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    env = os.environ.get("BLOOM_FIGSHARE_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for s in DEFAULT_FIGSHARE_SEARCH_PATHS:
        p = Path(s).expanduser()
        if p.is_file():
            return p
    return None


def load_figshare_exam_dataset(
    path: Optional[Union[str, Path]] = None,
    max_per_class: int = 2000,
    text_field: str = "QUESTION",
    label_field: str = "BT LEVEL",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Stratified subsample of a Bloom-labelled exam-question dataset.

    Expected schema (case-insensitive):
      - question column: ``QUESTION`` (or ``question``)
      - label column: ``BT LEVEL`` (or ``bloom_level`` / ``level``)
    Labels are normalised into the canonical revised Bloom levels used in
    this repository.
    """
    p = _find_figshare_exam_dataset(str(path) if path else None)
    if p is None:
        raise FileNotFoundError(
            "Figshare Bloom exam dataset not found. Set BLOOM_FIGSHARE_PATH or "
            f"pass path=... Searched: {DEFAULT_FIGSHARE_SEARCH_PATHS}"
        )
    import pandas as pd  # local import

    df = pd.read_csv(p, low_memory=False)
    cols = {str(c).strip().lower(): c for c in df.columns}
    qcol = cols.get(text_field.lower()) or cols.get("question") or cols.get("questions")
    lcol = (
        cols.get(label_field.lower())
        or cols.get("bt level")
        or cols.get("bt_level")
        or cols.get("bloom_level")
        or cols.get("level")
        or cols.get("btlevel")
    )
    if qcol is None or lcol is None:
        raise RuntimeError(
            f"Figshare Bloom CSV at {p} missing question/label columns; columns={list(df.columns)}"
        )
    df = df.dropna(subset=[qcol, lcol])
    df[qcol] = df[qcol].astype(str).str.strip()
    df[lcol] = df[lcol].astype(str).str.strip()
    df = df[df[qcol].str.len() > 0]
    if len(df) == 0:
        raise RuntimeError(f"Figshare Bloom dataset at {p} produced 0 usable rows after cleaning")

    # Normalise labels; drop unrecognised ones deterministically.
    norm = df[lcol].apply(_normalise_bloom)
    df = df.assign(_norm_label=norm)
    df = df[df["_norm_label"].notna()]
    df["_norm_label"] = df["_norm_label"].astype(str)
    df = df[df["_norm_label"].isin(BLOOM_LEVELS)]
    if len(df) == 0:
        raise RuntimeError(f"Figshare Bloom dataset at {p} produced 0 rows after label normalisation")

    rng = np.random.default_rng(seed)
    parts = []
    for lvl in BLOOM_LEVELS:
        sub = df[df["_norm_label"] == lvl]
        if len(sub) == 0:
            continue
        n = min(len(sub), int(max_per_class))
        idx = rng.choice(len(sub), size=n, replace=False)
        parts.append(sub.iloc[idx])
    out = (
        __import__("pandas").concat(parts)
        .sample(frac=1.0, random_state=int(seed))
        .reset_index(drop=True)
    )
    texts = out[qcol].astype(str).tolist()
    labels = out["_norm_label"].astype(str).tolist()
    logger.info(f"Figshare Bloom exam dataset: {len(texts)} rows ({max_per_class}/class) loaded from {p}")
    return texts, labels


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------
@dataclass
class ClassifierOutput:
    distribution: np.ndarray   # shape (6,), float32, sums to 1
    dominant_level: str        # canonical-cased BLOOM_LEVELS entry
    confidence: float          # 1 - H/log(K)
    entropy: float             # natural log entropy
    levels: List[str] = field(default_factory=lambda: list(BLOOM_LEVELS))


@dataclass
class OBEClassifierOutput:
    bloom_level: str
    cognitive_skill: str
    subject: str
    topic: str
    subtopic: str
    difficulty: str
    source_type: str
    language: str
    confidence: float
    support_count: int
    nearest_examples: List[Dict[str, str]] = field(default_factory=list)


class LocalOBEClassifier:
    """Offline kNN-style OBE metadata classifier over the local CSV."""

    def __init__(
        self,
        encoder: Optional[StableTextEncoder] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        max_rows: int = 6000,
        k: int = 5,
        model_dir: Union[str, Path] = "./models",
    ) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self.encoder = encoder
        self.model_dir = Path(model_dir)
        self.dataset_path = _find_obe_dataset(str(dataset_path) if dataset_path else None)
        if self.dataset_path is None:
            raise FileNotFoundError(
                "OBE dataset not found for LocalOBEClassifier. "
                "Set OBE_DATASET_PATH or place data/obe_dataset.csv locally."
            )
        self.max_rows = int(max_rows)
        self.k = int(k)
        self._rows: List[Dict[str, str]] = []
        self._embeddings: Optional[np.ndarray] = None
        self._pipelines: Dict[str, object] = {}

    @staticmethod
    def _clean(value: object, fallback: str = "Unknown") -> str:
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return fallback
        return s

    def _load_rows(self) -> None:
        if self._rows and self._embeddings is not None:
            return
        if self.encoder is None:
            raise RuntimeError("encoder is required for kNN fallback when trained pipelines are unavailable")
        import pandas as pd

        df = pd.read_csv(self.dataset_path, low_memory=False)
        wanted = [
            "question", "bloom_level", "cognitive_skill", "subject", "topic",
            "subtopic", "difficulty", "source_type", "language",
        ]
        cols = {str(c).strip().lower(): c for c in df.columns}
        missing = [c for c in ("question", "bloom_level") if c not in cols]
        if missing:
            raise RuntimeError(
                f"OBE dataset at {self.dataset_path} missing required columns: {missing}"
            )
        rename = {cols[c]: c for c in cols if c in wanted}
        df = df.rename(columns=rename)
        keep = [c for c in wanted if c in df.columns]
        df = df[keep].dropna(subset=["question", "bloom_level"])
        df["question"] = df["question"].astype(str).str.strip()
        df = df[df["question"].str.len() > 0]
        if len(df) == 0:
            raise RuntimeError(f"OBE dataset at {self.dataset_path} has no usable rows")
        if len(df) > self.max_rows:
            df = df.sample(n=self.max_rows, random_state=42).reset_index(drop=True)

        rows: List[Dict[str, str]] = []
        for rec in df.to_dict(orient="records"):
            bloom = _normalise_bloom(rec.get("bloom_level")) or "Understand"
            rows.append(
                {
                    "question": self._clean(rec.get("question"), fallback=""),
                    "bloom_level": bloom,
                    "cognitive_skill": self._clean(rec.get("cognitive_skill")),
                    "subject": self._clean(rec.get("subject")),
                    "topic": self._clean(rec.get("topic")),
                    "subtopic": self._clean(rec.get("subtopic")),
                    "difficulty": self._clean(rec.get("difficulty")),
                    "source_type": self._clean(rec.get("source_type")),
                    "language": self._clean(rec.get("language")),
                }
            )
        texts = [r["question"] for r in rows]
        emb = self.encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if emb.ndim == 1:
            emb = emb[None, :]
        self._rows = rows
        self._embeddings = np.ascontiguousarray(emb, dtype=np.float32)

    def _load_pipelines(self) -> None:
        if self._pipelines:
            return
        targets = ["bloom_level", "cognitive_skill", "difficulty", "source_type", "subject"]
        candidates = {
            "bloom_level": [
                self.model_dir / "figshare_bloom_tfidf.joblib",
                self.model_dir / "obe_bloom_tfidf.joblib",
                self.model_dir / "obe_bloom_level_tfidf.joblib",
            ],
            "cognitive_skill": [self.model_dir / "obe_cognitive_skill_tfidf.joblib"],
            "difficulty": [self.model_dir / "obe_difficulty_tfidf.joblib"],
            "source_type": [self.model_dir / "obe_source_type_tfidf.joblib"],
            "subject": [self.model_dir / "obe_subject_tfidf.joblib"],
        }
        loaded: Dict[str, object] = {}
        for target in targets:
            chosen = next((path for path in candidates[target] if path.is_file()), None)
            if chosen is None:
                continue
            try:
                loaded[target] = joblib.load(chosen)
            except Exception as exc:
                logger.warning(f"Skipping unreadable cached pipeline for {target}: {chosen} ({exc})")
        self._pipelines = loaded

    def _nearest_examples_lexical(self, text: str, top_n: int = 3) -> List[Dict[str, str]]:
        q_tokens = set(_clean_text(text).lower().split())
        if not q_tokens:
            return []

        import csv

        scored = []
        with self.dataset_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = {str(name).strip().lower(): str(name) for name in (reader.fieldnames or [])}
            qcol = fieldnames.get("question")
            bcol = fieldnames.get("bloom_level")
            scol = fieldnames.get("subject")
            tcol = fieldnames.get("topic")
            if qcol is None:
                return []
            for i, rec in enumerate(reader):
                if i >= 5000:
                    break
                doc_text = _clean_text(rec.get(qcol, ""))
                if not doc_text:
                    continue
                score = len(q_tokens & set(doc_text.lower().split()))
                if score > 0:
                    scored.append(
                        (
                            score,
                            {
                                "question": doc_text,
                                "bloom_level": _clean_text(rec.get(bcol, "")) if bcol else "",
                                "subject": _clean_text(rec.get(scol, "")) if scol else "",
                                "topic": _clean_text(rec.get(tcol, "")) if tcol else "",
                            },
                        )
                    )
        scored.sort(key=lambda x: (-x[0], x[1]["question"]))
        rows: List[Dict[str, str]] = []
        for _, rec in scored[:top_n]:
            rows.append(rec)
        return rows

    @staticmethod
    def _majority(rows: Sequence[Dict[str, str]], field: str) -> tuple[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            key = row.get(field, "Unknown") or "Unknown"
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return "Unknown", 0
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        return best[0], best[1]

    def predict(self, text: str) -> OBEClassifierOutput:
        if not isinstance(text, str):
            raise TypeError("predict() expects a single string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")
        self._load_pipelines()
        if "bloom_level" in self._pipelines:
            bloom_pipe = self._pipelines["bloom_level"]
            bloom = str(bloom_pipe.predict([text])[0])
            examples = self._nearest_examples_lexical(text)
            cog = "Unknown"
            difficulty = "Unknown"
            source_type = "Unknown"
            subject = examples[0]["subject"] if examples else "Unknown"
            topic = examples[0]["topic"] if examples else "Unknown"
            if "cognitive_skill" in self._pipelines:
                cog = str(self._pipelines["cognitive_skill"].predict([text])[0])
            if "difficulty" in self._pipelines:
                difficulty = str(self._pipelines["difficulty"].predict([text])[0])
            if "source_type" in self._pipelines:
                source_type = str(self._pipelines["source_type"].predict([text])[0])
            if "subject" in self._pipelines:
                subject = str(self._pipelines["subject"].predict([text])[0])
            conf = 0.0
            if hasattr(bloom_pipe, "predict_proba"):
                probs = bloom_pipe.predict_proba([text])[0]
                conf = float(np.max(probs))
            elif hasattr(bloom_pipe, "decision_function"):
                scores = bloom_pipe.decision_function([text])
                scores = np.asarray(scores, dtype=np.float64)
                if scores.ndim == 2 and scores.shape[1] > 1:
                    scores = scores[0]
                    scores = scores - np.max(scores)
                    exp_scores = np.exp(scores)
                    probs = exp_scores / np.sum(exp_scores)
                    conf = float(np.max(probs))
            return OBEClassifierOutput(
                bloom_level=bloom,
                cognitive_skill=cog,
                subject=subject,
                topic=topic,
                subtopic="Unknown",
                difficulty=difficulty,
                source_type=source_type,
                language="Unknown",
                confidence=conf,
                support_count=len(examples),
                nearest_examples=examples,
            )
        self._load_rows()
        assert self._embeddings is not None
        q = self.encoder.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        sims = (q @ self._embeddings.T).squeeze(0)
        k = min(self.k, len(self._rows))
        idx = np.argsort(-sims)[:k]
        nn = [self._rows[int(i)] for i in idx]

        bloom, bloom_votes = self._majority(nn, "bloom_level")
        cog, _ = self._majority(nn, "cognitive_skill")
        subject, _ = self._majority(nn, "subject")
        topic, _ = self._majority(nn, "topic")
        subtopic, _ = self._majority(nn, "subtopic")
        difficulty, _ = self._majority(nn, "difficulty")
        source_type, _ = self._majority(nn, "source_type")
        language, _ = self._majority(nn, "language")
        conf = float(max(0.0, min(1.0, bloom_votes / max(1, k))))
        examples = [
            {
                "question": row["question"],
                "bloom_level": row["bloom_level"],
                "subject": row["subject"],
                "topic": row["topic"],
            }
            for row in nn[:3]
        ]
        return OBEClassifierOutput(
            bloom_level=bloom,
            cognitive_skill=cog,
            subject=subject,
            topic=topic,
            subtopic=subtopic,
            difficulty=difficulty,
            source_type=source_type,
            language=language,
            confidence=conf,
            support_count=bloom_votes,
            nearest_examples=examples,
        )


# ----------------------------------------------------------------------------
# BloomLDLClassifier
# ----------------------------------------------------------------------------
class BloomLDLClassifier:
    """Linear LDL head over MiniLM embeddings with Gaussian-smoothed labels."""

    def __init__(
        self,
        encoder: Optional[StableTextEncoder] = None,
        encoder_name: str = "all-MiniLM-L6-v2",
        sigma: float = 1.0,
        # Phase-7 upgrade: hybrid ordinal targets + ordinal consistency loss.
        target_hybrid_alpha: float = 0.35,
        hard_target_beta: float = 0.35,
        ordinal_scale: float = 1.0,
        ordinal_margin: float = 0.15,
        lambda_ordinal: float = 0.05,
        lambda_entropy: float = 0.08,
        lr: float = 8.0,
        epochs: int = 1000,
        l2: float = 1e-5,
        init_std: float = 0.1,
        seed: int = 42,
    ) -> None:
        # Defaults tuned for L2-normalised MiniLM embeddings (||x||=1):
        #   * lr=8.0 with init_std=0.1 converges in <=1000 full-batch steps
        #   * smaller lr (e.g. 0.5) under-fits; lr>=16 diverges.
        # Probed on a stratified OBE subsample (80/class) to ~58% train acc.
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not (0.0 <= target_hybrid_alpha <= 1.0):
            raise ValueError("target_hybrid_alpha must be in [0, 1]")
        if not (0.0 <= hard_target_beta <= 1.0):
            raise ValueError("hard_target_beta must be in [0, 1]")
        if ordinal_scale <= 0:
            raise ValueError("ordinal_scale must be > 0")
        if ordinal_margin <= 0:
            raise ValueError("ordinal_margin must be > 0")
        if lambda_ordinal < 0:
            raise ValueError("lambda_ordinal must be >= 0")
        if lambda_entropy < 0:
            raise ValueError("lambda_entropy must be >= 0")
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        if init_std <= 0:
            raise ValueError("init_std must be > 0")

        self._enc: Optional[StableTextEncoder] = encoder
        self.encoder_name = encoder_name
        self.sigma = float(sigma)
        self.target_hybrid_alpha = float(target_hybrid_alpha)
        self.hard_target_beta = float(hard_target_beta)
        self.ordinal_scale = float(ordinal_scale)
        self.ordinal_margin = float(ordinal_margin)
        self.lambda_ordinal = float(lambda_ordinal)
        self.lambda_entropy = float(lambda_entropy)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.init_std = float(init_std)
        self.seed = int(seed)

        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self._train_loss_history: List[float] = []

    # ------------------------------------------------------------------ #
    # Encoder (lazy, shared with retriever/summarizer cache)
    # ------------------------------------------------------------------ #
    @property
    def encoder(self) -> StableTextEncoder:
        if self._enc is None:
            logger.info(f"Loading encoder '{self.encoder_name}' on CPU")
            self._enc = StableTextEncoder(
                self.encoder_name,
                device="cpu",
                local_files_only=True,
                n_features=EMBED_DIM,
            )
        return self._enc

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        emb = self.encoder.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if emb.ndim == 1:
            emb = emb[None, :]
        return np.ascontiguousarray(emb, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Soft-label construction (ordinal Gaussian)
    # ------------------------------------------------------------------ #
    def _soft_targets(self, level_indices: np.ndarray) -> np.ndarray:
        K = len(BLOOM_LEVELS)
        idx = np.arange(K, dtype=np.float32)[None, :]
        levels = level_indices.astype(np.float32)[:, None]
        diffs = (idx - levels) ** 2
        T = np.exp(-diffs / (2.0 * self.sigma ** 2))
        T = T / T.sum(axis=1, keepdims=True)
        return T.astype(np.float32)

    def _ordinal_pmf_targets(self, level_indices: np.ndarray) -> np.ndarray:
        """Ordinal-consistent target PMF derived from smoothed CDF thresholds.

        For a true class y, define a (soft) probability that the level is
        >= k using a sigmoid on (y + 0.5 - k) / s. Enforce boundary CDF:
        CDF>=0 = 1 and CDF>=K = 0, then compute the PMF by differencing.

        This yields a valid simplex distribution with an ordinal structure
        that is robust to noisy boundary labels.
        """
        K = len(BLOOM_LEVELS)
        y = level_indices.astype(np.float32)[:, None]  # (N,1)
        k = np.arange(1, K, dtype=np.float32)[None, :]  # thresholds 1..K-1
        s = float(self.ordinal_scale)
        c_inner = 1.0 / (1.0 + np.exp(-(y + 0.5 - k) / s))  # (N, K-1)
        # c[0]=1, c[K]=0
        c = np.concatenate(
            [
                np.ones((len(level_indices), 1), dtype=np.float32),
                c_inner.astype(np.float32, copy=False),
                np.zeros((len(level_indices), 1), dtype=np.float32),
            ],
            axis=1,
        )  # (N, K+1)
        pmf = (c[:, :-1] - c[:, 1:]).astype(np.float32, copy=False)  # (N,K)
        pmf = np.clip(pmf, 0.0, 1.0)
        pmf = pmf / pmf.sum(axis=1, keepdims=True)
        return pmf.astype(np.float32)

    def _hybrid_targets(self, level_indices: np.ndarray) -> np.ndarray:
        """Hybrid target distribution = Gaussian LDL + ordinal-PMF bias
        + a small hard-label component to counter label noise induced
        confidence collapse.
        """
        g = self._soft_targets(level_indices)
        o = self._ordinal_pmf_targets(level_indices)
        a = float(self.target_hybrid_alpha)
        T = (1.0 - a) * g + a * o
        # Add a small hard-label component (still a simplex PMF).
        b = float(self.hard_target_beta)
        if b > 0.0:
            K = len(BLOOM_LEVELS)
            H = np.zeros((len(level_indices), K), dtype=np.float32)
            H[np.arange(len(level_indices)), level_indices.astype(np.int64)] = 1.0
            T = (1.0 - b) * T + b * H
        T = T / T.sum(axis=1, keepdims=True)
        return T.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Numerically stable softmax
    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        m = z.max(axis=1, keepdims=True)
        e = np.exp(z - m)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # Training (deterministic full-batch GD)
    # ------------------------------------------------------------------ #
    def fit(
        self,
        texts: Sequence[str],
        levels: Sequence[str],
    ) -> "BloomLDLClassifier":
        if len(texts) != len(levels):
            raise ValueError("texts and levels must have the same length")
        if len(texts) == 0:
            raise ValueError("must provide at least one training example")

        try:
            idx = np.array(
                [BLOOM_INDEX[str(l).strip().lower()] for l in levels],
                dtype=np.int64,
            )
        except KeyError as e:
            raise ValueError(f"unknown bloom level in training labels: {e!r}") from e

        logger.info(f"Encoding {len(texts)} training texts...")
        t0 = time.time()
        X = self.encode(texts)             # (N, D), L2-normalised
        # Phase-7 supervision upgrade: hybrid ordinal targets
        T = self._hybrid_targets(idx)      # (N, K)
        N, D = X.shape
        K = T.shape[1]
        logger.info(f"Encoded in {time.time()-t0:.1f}s; X={X.shape}, T={T.shape}")

        rng = np.random.default_rng(self.seed)
        self.W = (rng.standard_normal((D, K)) * self.init_std).astype(np.float32)
        self.b = np.zeros((K,), dtype=np.float32)
        self._train_loss_history = []

        for epoch in range(self.epochs):
            logits = X @ self.W + self.b
            P = self._softmax(logits)

            kl = float(
                (T * (np.log(T + 1e-12) - np.log(P + 1e-12))).sum(axis=1).mean()
            )
            # ---- Ordinal consistency penalty (class-mean margin ordering)
            # Encourage monotonic ordering of the *expected predicted level*
            # across true Bloom classes: E[level|y=c] < E[level|y=c+1].
            levels_grid = np.arange(K, dtype=np.float32)
            mu = (P * levels_grid[None, :]).sum(axis=1)  # (N,)
            mean_mu = np.full(K, np.nan, dtype=np.float32)
            counts = np.zeros(K, dtype=np.int64)
            for c in range(K):
                m = (idx == c)
                counts[c] = int(m.sum())
                if counts[c] > 0:
                    mean_mu[c] = float(mu[m].mean())
            margin = float(self.ordinal_margin)
            ord_pen = 0.0
            # weights emphasise mid-level boundaries (Apply/Analyze/Evaluate)
            boundary_w = np.ones(K - 1, dtype=np.float32)
            boundary_w[2] = 1.6  # Apply->Analyze
            boundary_w[3] = 1.6  # Analyze->Evaluate
            # compute squared hinge penalties on adjacent gaps
            gaps = []
            for c in range(K - 1):
                if math.isnan(float(mean_mu[c])) or math.isnan(float(mean_mu[c + 1])):
                    gaps.append(0.0)
                    continue
                d_gap = float(mean_mu[c + 1] - mean_mu[c])
                v = max(0.0, margin - d_gap)
                gaps.append(d_gap)
                ord_pen += float(boundary_w[c]) * (v * v)

            # ---- Entropy regularisation (discourage collapsed uniform output)
            ent = float(-(P * np.log(P + 1e-12)).sum(axis=1).mean())
            # Total loss (for logging) matches the requested form.
            total = (
                kl
                + self.lambda_ordinal * ord_pen
                + self.lambda_entropy * ent
            )
            self._train_loss_history.append(total)

            # Base gradient from KL(T||P): dL/dlogits = (P - T) / N
            grad_logits = (P - T) / N  # (N,K)

            # Add entropy gradient: dH/dz = J @ (-(log p + 1))
            if self.lambda_entropy > 0.0:
                g = (-(np.log(P + 1e-12) + 1.0)).astype(np.float32)  # (N,K)
                pg = (P * g).sum(axis=1, keepdims=True)             # (N,1)
                grad_ent = (P * (g - pg)) / N                       # (N,K)
                grad_logits += (self.lambda_entropy * grad_ent).astype(np.float32)

            # Add ordinal margin gradient through class-mean expected level.
            if self.lambda_ordinal > 0.0:
                # d mu_i / d z_i = J_i @ levels_grid
                # with J_i @ v = p_i * (v - (p_i·v))
                p_dot_levels = (P * levels_grid[None, :]).sum(axis=1, keepdims=True)  # (N,1)
                dmu_dz = P * (levels_grid[None, :] - p_dot_levels)  # (N,K)

                # Compute d ord_pen / d mean_mu via squared hinge on gaps.
                dL_dmean = np.zeros(K, dtype=np.float32)
                for c in range(K - 1):
                    if math.isnan(float(mean_mu[c])) or math.isnan(float(mean_mu[c + 1])):
                        continue
                    d_gap = float(mean_mu[c + 1] - mean_mu[c])
                    v = margin - d_gap
                    if v <= 0:
                        continue
                    w = float(boundary_w[c])
                    # ord_pen += w * v^2 ; v = margin - (m_{c+1} - m_c)
                    # d/dm_{c+1} = -2*w*v ; d/dm_c = +2*w*v
                    d = 2.0 * w * float(v)
                    dL_dmean[c] += float(d)
                    dL_dmean[c + 1] -= float(d)

                # Scatter mean gradients back to per-sample logits via mu.
                if np.any(dL_dmean != 0.0):
                    per_sample_coeff = np.zeros(N, dtype=np.float32)
                    for c in range(K):
                        if counts[c] > 0 and dL_dmean[c] != 0.0:
                            per_sample_coeff[idx == c] = dL_dmean[c] / float(counts[c])
                    grad_ord = (per_sample_coeff[:, None] * dmu_dz) / N  # (N,K)
                    grad_logits += (self.lambda_ordinal * grad_ord).astype(np.float32)

            grad_W = (X.T @ grad_logits) + self.l2 * self.W
            grad_b = grad_logits.mean(axis=0)
            self.W -= (self.lr * grad_W).astype(np.float32)
            self.b -= (self.lr * grad_b).astype(np.float32)

            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info(
                    f"  epoch {epoch+1:>4}/{self.epochs}  "
                    f"loss={total:.4f}  KL={kl:.4f}  ord={ord_pen:.4f}  H={ent:.4f}"
                )

        logger.info(f"final training loss = {self._train_loss_history[-1]:.4f}")
        return self

    # ------------------------------------------------------------------ #
    # Ordinal evaluation metrics (publishability requirements)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rankdata(x: np.ndarray) -> np.ndarray:
        """Average-rank ties (like scipy.stats.rankdata(method='average'))."""
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
        # tie correction: assign average rank within each tie group
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            if j > i:
                avg = ranks[order[i:j + 1]].mean()
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    @staticmethod
    def spearmanr(a: Sequence[float], b: Sequence[float]) -> float:
        """Spearman rank correlation (tie-aware, deterministic)."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size != b.size or a.size < 2:
            return float("nan")
        ra = BloomLDLClassifier._rankdata(a)
        rb = BloomLDLClassifier._rankdata(b)
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
        if denom <= 0:
            return float("nan")
        return float((ra * rb).sum() / denom)

    @staticmethod
    def kendall_tau_b(a: Sequence[int], b: Sequence[int], max_n: int = 500) -> float:
        """Kendall's tau-b with tie correction (O(n^2), capped for speed)."""
        a0 = np.asarray(a, dtype=np.int64)
        b0 = np.asarray(b, dtype=np.int64)
        n = int(min(len(a0), len(b0), max_n))
        if n < 2:
            return float("nan")
        a0 = a0[:n]
        b0 = b0[:n]
        conc = disc = tie_a = tie_b = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                da = int(a0[i] - a0[j])
                db = int(b0[i] - b0[j])
                if da == 0 and db == 0:
                    tie_a += 1
                    tie_b += 1
                elif da == 0:
                    tie_a += 1
                elif db == 0:
                    tie_b += 1
                else:
                    s = da * db
                    if s > 0:
                        conc += 1
                    elif s < 0:
                        disc += 1
        num = conc - disc
        den = math.sqrt((conc + disc + tie_a) * (conc + disc + tie_b))
        return float(num / den) if den > 0 else float("nan")

    def evaluate_ordinal(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        max_kendall_n: int = 500,
    ) -> Dict[str, object]:
        """Compute publishable ordinal metrics for Bloom classification."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have same length")
        if not texts:
            return {}
        P = self.predict_distribution(texts).astype(np.float64)
        pred_idx = P.argmax(axis=1).astype(np.int64)
        true_idx = np.array([BLOOM_INDEX[str(l).lower()] for l in labels], dtype=np.int64)

        # confidence: top-1 probability and entropy-based (both reported)
        top1 = P.max(axis=1)
        ent = -(P * np.log(P + 1e-12)).sum(axis=1)
        conf_ent = 1.0 - (ent / math.log(len(BLOOM_LEVELS)))

        levels_grid = np.arange(len(BLOOM_LEVELS), dtype=np.float64)
        mu = (P * levels_grid[None, :]).sum(axis=1)  # expected level

        acc = float((pred_idx == true_idx).mean())
        ord_mae = float(np.abs(pred_idx - true_idx).mean())
        spr = float(self.spearmanr(mu, true_idx.astype(np.float64)))
        tau = float(self.kendall_tau_b(mu.round().astype(int), true_idx.tolist(), max_n=max_kendall_n))

        # Boundary confusion counts: Apply↔Analyze and Analyze↔Evaluate
        def _pair_conf(a_i: int, b_i: int) -> Dict[str, int]:
            ab = int(((true_idx == a_i) & (pred_idx == b_i)).sum())
            ba = int(((true_idx == b_i) & (pred_idx == a_i)).sum())
            aa = int(((true_idx == a_i) & (pred_idx == a_i)).sum())
            bb = int(((true_idx == b_i) & (pred_idx == b_i)).sum())
            return {"a_to_b": ab, "b_to_a": ba, "a_correct": aa, "b_correct": bb}

        boundary = {
            "Apply_vs_Analyze": _pair_conf(2, 3),
            "Analyze_vs_Evaluate": _pair_conf(3, 4),
        }

        return {
            "n": int(len(texts)),
            "accuracy": acc,
            "macro_f1": float(
                np.mean([
                    0.0 if ((pred_idx == c).sum() == 0 or (true_idx == c).sum() == 0)
                    else (2.0 * float(((pred_idx == c) & (true_idx == c)).sum())
                          / float((pred_idx == c).sum() + (true_idx == c).sum()))
                    for c in range(len(BLOOM_LEVELS))
                ])
            ),
            "ordinal_mae": ord_mae,
            "spearman": spr,
            "kendall_tau_b": tau,
            "mean_top1_confidence": float(top1.mean()),
            "mean_entropy_confidence": float(conf_ent.mean()),
            "boundary_confusions": boundary,
        }

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict_distribution(
        self, texts: Union[str, Sequence[str]]
    ) -> np.ndarray:
        """Return (N, 6) probability distributions over Bloom levels."""
        if self.W is None or self.b is None:
            raise RuntimeError("classifier not fitted; call fit() or load() first")
        if isinstance(texts, str):
            texts = [texts]
        X = self.encode(texts)
        return self._softmax(X @ self.W + self.b).astype(np.float32)

    def predict(self, text: str) -> ClassifierOutput:
        """Single-text inference returning a structured :class:`ClassifierOutput`."""
        if not isinstance(text, str):
            raise TypeError("predict() expects a single string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

        dist = self.predict_distribution([text])[0]
        dominant = BLOOM_LEVELS[int(dist.argmax())]
        ent = float(-(dist * np.log(dist + 1e-12)).sum())
        max_ent = float(np.log(len(BLOOM_LEVELS)))
        conf = float(max(0.0, min(1.0, 1.0 - ent / max_ent)))
        return ClassifierOutput(
            distribution=dist.astype(np.float32),
            dominant_level=dominant,
            confidence=conf,
            entropy=ent,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        if self.W is None or self.b is None:
            raise RuntimeError("nothing to save: classifier not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W=self.W,
            b=self.b,
            sigma=np.float32(self.sigma),
            encoder_name=np.array(self.encoder_name),
            levels=np.array(BLOOM_LEVELS),
        )
        logger.info(f"saved classifier weights to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        encoder: Optional[StableTextEncoder] = None,
    ) -> "BloomLDLClassifier":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        d = np.load(path, allow_pickle=False)
        encoder_name = str(d["encoder_name"]) if d["encoder_name"].ndim == 0 else str(d["encoder_name"].item())
        clf = cls(
            encoder=encoder,
            encoder_name=encoder_name,
            sigma=float(d["sigma"]),
        )
        clf.W = d["W"].astype(np.float32)
        clf.b = d["b"].astype(np.float32)
        if list(d["levels"].astype(str)) != BLOOM_LEVELS:
            raise ValueError("level ordering mismatch in saved weights")
        logger.info(f"loaded classifier weights from {path}")
        return clf


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * Soft-label construction is a valid distribution centred on the true level.
#   * Stratified OBE subsample loads with balanced class counts.
#   * fit() runs deterministically, KL loss decreases, accuracy > random.
#   * predict() returns a valid 6-d distribution, valid dominant level,
#     confidence in [0, 1], non-negative entropy.
#   * Determinism: predict twice -> identical distribution.
#   * save / load round trip preserves predictions exactly.
# ============================================================================
def _self_test() -> None:
    weights_path = Path(DEFAULT_WEIGHTS_PATH)

    # 1. Soft-target sanity (ordinal-Gaussian shape) ------------------------
    clf_sanity = BloomLDLClassifier(sigma=1.0)
    T = clf_sanity._soft_targets(np.array([0, 2, 5]))
    assert T.shape == (3, 6)
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-5), "soft labels not normalised"
    assert np.argmax(T, axis=1).tolist() == [0, 2, 5], "soft-label peak misplaced"
    # Neighbouring levels must have higher mass than far levels
    assert T[1, 1] > T[1, 0] > T[1, 5], "ordinal smoothing violated for Apply"

    # 2. Dataset routing (publishable supervision) --------------------------
    # Train on Figshare-style Bloom exam dataset (primary signal).
    # Use OBE only for validation/test.
    fig_path = _find_figshare_exam_dataset()
    assert fig_path is not None, (
        "Figshare Bloom exam dataset not found; set BLOOM_FIGSHARE_PATH to run the test."
    )
    train_texts, train_labels = load_figshare_exam_dataset(
        fig_path, max_per_class=120, seed=42
    )
    assert len(train_texts) > 0
    counts = {lvl: train_labels.count(lvl) for lvl in BLOOM_LEVELS}
    assert all(c > 0 for c in counts.values()), f"missing Bloom classes in Figshare: {counts}"

    obe_path = _find_obe_dataset()
    assert obe_path is not None, "OBE dataset not found; set OBE_DATASET_PATH for validation."
    val_texts, val_labels = load_obe_dataset(
        obe_path, max_per_class=60, text_field="question", seed=43
    )
    assert len(val_texts) > 0

    # 3. Train (deterministic) ---------------------------------------------
    clf = BloomLDLClassifier(sigma=1.0)  # upgraded defaults include ordinal losses
    t0 = time.time()
    clf.fit(train_texts, train_labels)
    train_dt = time.time() - t0
    assert clf.W is not None and clf.b is not None
    assert clf.W.shape == (EMBED_DIM, 6)
    # Loss should decrease by a non-trivial amount
    assert (
        clf._train_loss_history[0] - clf._train_loss_history[-1]
    ) > 1e-3, "training loss did not decrease"

    # 4. predict() structural invariants -----------------------------------
    out = clf.predict("What is the definition of photosynthesis?")
    assert isinstance(out, ClassifierOutput)
    assert out.distribution.shape == (6,)
    assert np.isclose(out.distribution.sum(), 1.0, atol=1e-5)
    assert (out.distribution >= 0).all()
    assert out.dominant_level in BLOOM_LEVELS
    assert 0.0 <= out.confidence <= 1.0
    assert out.entropy >= 0
    assert out.levels == BLOOM_LEVELS

    # 5. Determinism -------------------------------------------------------
    o1 = clf.predict("Compare and contrast supervised vs unsupervised learning.")
    o2 = clf.predict("Compare and contrast supervised vs unsupervised learning.")
    assert np.allclose(o1.distribution, o2.distribution, atol=1e-7)

    # 6. Training-set quality metrics (publishability targets) --------------
    P = clf.predict_distribution(train_texts)
    pred_idx = P.argmax(axis=1)
    true_idx = np.array([BLOOM_INDEX[l.lower()] for l in train_labels])
    acc = float((pred_idx == true_idx).mean())
    assert acc >= 0.45, f"train accuracy too low: {acc:.3f} (random=0.167)"
    logger.info(f"train accuracy={acc:.3f}, time={train_dt:.1f}s")

    # Soft-prediction quality: ordinal-distance MAE should be much smaller
    # than the random baseline of ~1.7 levels.
    ord_mae = float(np.abs(pred_idx - true_idx).mean())
    assert ord_mae <= 1.3, f"ordinal MAE too high: {ord_mae:.3f}"
    logger.info(f"ordinal MAE={ord_mae:.3f}")

    # Confidence should not collapse to ~1/6; require >0.5 mean top-1 on train.
    mean_top1 = float(P.max(axis=1).mean())
    assert mean_top1 >= 0.50, f"collapsed confidence persists: mean top-1={mean_top1:.3f}"
    logger.info(f"train mean top-1 confidence={mean_top1:.3f}")

    # Ordinal correlations (required for publishable Bloom work).
    levels_grid = np.arange(len(BLOOM_LEVELS), dtype=np.float64)
    mu = (P.astype(np.float64) * levels_grid[None, :]).sum(axis=1)
    spr = float(BloomLDLClassifier.spearmanr(mu, true_idx.astype(np.float64)))
    assert spr >= 0.60, f"Spearman too low: {spr:.3f}"
    logger.info(f"train Spearman rho={spr:.3f}")
    tau = float(BloomLDLClassifier.kendall_tau_b(mu.round().astype(int), true_idx.tolist(), max_n=400))
    logger.info(f"train Kendall tau-b={tau:.3f} (capped n=400)")

    # Validation on OBE (non-dominant signal): should run and produce
    # non-trivial correlations (not necessarily as high due to label noise).
    m_val = clf.evaluate_ordinal(val_texts[:200], val_labels[:200])
    assert m_val and "spearman" in m_val
    logger.info(f"val (OBE) metrics: {m_val}")

    # 7. Bad inputs --------------------------------------------------------
    for bad in ("", "   "):
        try:
            clf.predict(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for input {bad!r}")
    try:
        clf.predict(123)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-string input")

    # 8. Save / load round trip --------------------------------------------
    clf.save(weights_path)
    clf2 = BloomLDLClassifier.load(weights_path, encoder=clf.encoder)
    out_a = clf.predict("Design an experiment to test the effect of light on plants.")
    out_b = clf2.predict("Design an experiment to test the effect of light on plants.")
    assert np.allclose(out_a.distribution, out_b.distribution, atol=1e-7), (
        "save/load round trip changed predictions"
    )

    _ok("classifier.py sanity check passed")


if __name__ == "__main__":
    _self_test()
