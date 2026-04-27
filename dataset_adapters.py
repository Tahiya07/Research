"""
dataset_adapters.py
==============================================================================
Dataset adapter layer for the Phase-6 evaluation pipeline of
"A Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments".

This module is intentionally **decoupled** from the rest of the codebase:

* It does NOT import any Phase-1..6 module.
* It does NOT modify any earlier-phase file.
* It only exposes a unified :class:`DatasetAdapter` interface plus six
  concrete implementations:

    +------------------+----------------------+--------------------+
    | name             | task_type            | sources tried      |
    +==================+======================+====================+
    | obe              | qa                   | local CSV          |
    | bloom            | classification       | local CSV          |
    | scienceqa        | qa                   | local CSV / HF     |
    | sciqa            | qa                   | HF / synthetic     |
    | docvqa           | docvqa               | HF / synthetic     |
    | privacy          | privacy              | HF / synthetic     |
    +------------------+----------------------+--------------------+

Each adapter resolves its data file through three fallback layers:

    1. Explicit ``path=`` constructor argument (or ``DATASET_<NAME>_PATH``
       environment variable).
    2. A list of well-known local search paths
       (``./data/datasets/``, ``~/PycharmProjects/Thesis/...``, ...).
    3. The HuggingFace ``datasets`` library, *if* it is already importable
       (we never download via a fresh pip install).
    4. A small **deterministic synthetic mini-dataset** (5-18 samples)
       hard-coded in this file, so the evaluation pipeline always has
       something to validate against -- even on a clean offline machine.

The synthetic fallback is logged loudly so reviewers can see when real
data was substituted for self-test.

Unified interface
-----------------
::

    adapter = get_adapter("scienceqa", max_samples=10)
    adapter.load()
    questions = adapter.get_questions()        # List[str]
    contexts  = adapter.get_context()          # List[str]
    answers   = adapter.get_answers()          # List[str]
    labels    = adapter.get_labels()           # Optional[List[str]]
    metadata  = adapter.get_metadata()         # List[Dict[str, Any]]
    samples   = adapter.get_samples()          # List[DatasetSample]
    info      = adapter.dataset_info()         # name, task, source, n, ...

Constraints
-----------
* Pure Python + numpy, no heavy dependencies introduced here.
* Deterministic seeding: random/numpy/torch == 42.
* CPU only, no API calls beyond the local HF cache (only used if already
  installed).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("dataset_adapters")
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
# Deduplication (STRICT, deterministic, offline-only)
# ----------------------------------------------------------------------------
_CANON_RE = re.compile(r"[a-z0-9]+")


def canonical_id(question_text: str) -> str:
    """Canonicalize question text into a stable ID for deduplication."""
    toks = _CANON_RE.findall((question_text or "").lower())
    return " ".join(toks).strip()


def deduplicate_samples(
    samples: Sequence["DatasetSample"],
    *,
    seen_ids: set[str],
) -> List["DatasetSample"]:
    """Deterministically drop duplicate questions (order preserved)."""
    out: List[DatasetSample] = []
    for s in samples:
        cid = canonical_id(s.question)
        if not cid:
            continue
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        out.append(s)
    return out


# ----------------------------------------------------------------------------
# Bloom level normalisation
# ----------------------------------------------------------------------------
BLOOM_LEVELS_CANONICAL: List[str] = [
    "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create",
]
# Map original (1956) Bloom names + revised Anderson (2001) names to the
# canonical revised levels we use everywhere else in the codebase.
_BLOOM_ALIASES: Dict[str, str] = {
    # canonical
    "remember": "Remember", "understand": "Understand", "apply": "Apply",
    "analyze": "Analyze", "evaluate": "Evaluate", "create": "Create",
    # 1956 Bloom -> revised
    "knowledge": "Remember",
    "comprehension": "Understand",
    "application": "Apply",
    "analysis": "Analyze",
    "synthesis": "Create",
    "evaluation": "Evaluate",
    # common shorthand / typos
    "remembering": "Remember", "understanding": "Understand",
    "applying": "Apply", "analysing": "Analyze", "analyzing": "Analyze",
    "evaluating": "Evaluate", "creating": "Create",
}


def normalise_bloom(label: Any) -> Optional[str]:
    """Map raw label strings to canonical revised-Bloom level (or None)."""
    if label is None:
        return None
    s = str(label).strip().lower()
    if not s:
        return None
    return _BLOOM_ALIASES.get(s)


# ----------------------------------------------------------------------------
# Local-disk search paths (cheapest fallback before HF datasets)
# ----------------------------------------------------------------------------
_HOME = str(Path.home())
DEFAULT_DATA_ROOTS: List[str] = [
    "./data/datasets/",
    "./data/",
    "./",
    f"{_HOME}/PycharmProjects/Thesis/data/",
    f"{_HOME}/PycharmProjects/Thesis/models/external_datasets/",
    f"{_HOME}/Documents/",
    f"{_HOME}/.cache/research_datasets/",
]


def _find_first(filenames: Sequence[str], explicit: Optional[str] = None) -> Optional[Path]:
    """Return the first existing path among (explicit, env, search-roots × filenames)."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
    for fname in filenames:
        for root in DEFAULT_DATA_ROOTS:
            p = Path(root).expanduser() / fname
            if p.is_file():
                return p
        # also try filename relative to CWD (useful for absolute paths)
        p = Path(fname).expanduser()
        if p.is_file():
            return p
    return None


def _hf_cache_root() -> Path:
    """Return the local HuggingFace dataset cache root (no network)."""
    if os.environ.get("HF_DATASETS_CACHE"):
        return Path(os.environ["HF_DATASETS_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_dataset_cached(repo_id: str) -> bool:
    """Return True iff the dataset ``repo_id`` is already present in the
    local HuggingFace cache. We use this to gate any call to
    ``datasets.load_dataset`` so we never trigger a network download
    (the project forbids external APIs at runtime)."""
    safe = "datasets--" + repo_id.replace("/", "--")
    return (_hf_cache_root() / safe).is_dir()


def _try_import_hf_datasets():  # noqa: ANN201 - dynamic import
    """Return the ``datasets`` module if importable AND the local cache
    contains at least one dataset, else ``None``.

    Loading is always done with ``HF_DATASETS_OFFLINE=1`` to forbid
    network access even if the user has internet available.
    """
    try:
        # Force offline mode for any subsequent HF dataset call.
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import datasets as _hf  # type: ignore[import-not-found]
        return _hf
    except Exception:  # pragma: no cover
        return None


# ============================================================================
# Sample container + base adapter
# ============================================================================
@dataclass
class DatasetSample:
    """Unified record produced by every adapter.

    ``question`` and ``answer`` are mandatory (use empty strings only when
    the task literally has no answer, e.g., privacy ASR queries). Other
    fields are best-effort.
    """
    question: str
    answer: str = ""
    context: str = ""
    label: Optional[str] = None
    subject: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Abstract dataset adapter exposing a uniform ``get_*()`` API."""

    name: str = "base"
    task_type: str = "qa"  # qa | classification | privacy | docvqa

    def __init__(
        self,
        max_samples: Optional[int] = None,
        seed: int = 42,
        path: Optional[str] = None,
        prefer_synthetic: bool = False,
    ) -> None:
        self.max_samples = max_samples
        self.seed = int(seed)
        self.path = path
        self.prefer_synthetic = bool(prefer_synthetic)
        self._samples: List[DatasetSample] = []
        self._source: str = "<unloaded>"  # "local:<path>" / "hf:<id>" / "synthetic"

    # --- subclass hook ------------------------------------------------------
    @abstractmethod
    def _load_raw(self) -> List[DatasetSample]:
        """Load (potentially many) raw samples; ``load()`` handles capping."""
        ...

    # --- core lifecycle -----------------------------------------------------
    def load(self) -> List[DatasetSample]:
        """Load + (optionally) deterministically subsample to ``max_samples``."""
        if self._samples:
            return self._samples
        raw = self._load_raw()
        if not raw:
            raise RuntimeError(
                f"adapter '{self.name}' produced 0 samples (source={self._source})"
            )
        # STRICT per-run deduplication (order preserved; no global state).
        seen_ids: set[str] = set()
        raw = deduplicate_samples(raw, seen_ids=seen_ids)
        if self.max_samples and len(raw) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(raw), size=self.max_samples, replace=False)
            raw = [raw[int(i)] for i in sorted(idx)]
        self._samples = raw
        logger.info(
            "[%s] loaded %d samples (source=%s)", self.name, len(raw), self._source
        )
        return self._samples

    # --- unified getters ----------------------------------------------------
    def get_questions(self) -> List[str]:
        return [s.question for s in self._ensure()]

    def get_context(self) -> List[str]:
        return [s.context for s in self._ensure()]

    def get_answers(self) -> List[str]:
        return [s.answer for s in self._ensure()]

    def get_labels(self) -> Optional[List[str]]:
        labs = [s.label for s in self._ensure()]
        if all(l is None for l in labs):
            return None
        # fill missing with empty string (preserves length)
        return [str(l) if l is not None else "" for l in labs]

    def get_metadata(self) -> List[Dict[str, Any]]:
        out = []
        for s in self._ensure():
            d = dict(s.metadata)
            d.setdefault("subject", s.subject)
            out.append(d)
        return out

    def get_samples(self) -> List[DatasetSample]:
        return list(self._ensure())

    def dataset_info(self) -> Dict[str, Any]:
        s = self._ensure()
        return {
            "name": self.name,
            "task_type": self.task_type,
            "n_samples": len(s),
            "source": self._source,
            "max_samples": self.max_samples,
            "seed": self.seed,
        }

    # --- internals ----------------------------------------------------------
    def _ensure(self) -> List[DatasetSample]:
        if not self._samples:
            self.load()
        return self._samples

    def __len__(self) -> int:
        return len(self._ensure())


# ============================================================================
# 1. OBE adapter -- thin wrapper around the existing OBE CSV
# ============================================================================
class OBEAdapter(DatasetAdapter):
    """OBE academic dataset (subject / question / answer / summary / source_text /
    bloom_level). This is the original Phase-6 baseline dataset."""

    name = "obe"
    task_type = "qa"

    def _load_raw(self) -> List[DatasetSample]:
        path = _find_first(
            ["obe_dataset.csv", "Obe Dataset.csv", "OBE.csv"],
            explicit=self.path or os.environ.get("OBE_DATASET_PATH"),
        )
        if path is None:
            self._source = "synthetic"
            return _SYNTHETIC_OBE
        self._source = f"local:{path}"
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas required for OBE CSV") from e

        df = pd.read_csv(
            path,
            usecols=lambda c: c in {
                "subject", "topic", "bloom_level", "language", "source_text",
                "summary", "question", "answer",
            },
            low_memory=False,
        )
        if "language" in df.columns:
            df = df[df["language"].astype(str).str.strip().str.lower() == "en"]
        df = df.dropna(
            subset=[c for c in ("question", "answer", "source_text") if c in df.columns]
        )
        out: List[DatasetSample] = []
        for _, row in df.iterrows():
            bloom_raw = row.get("bloom_level", None)
            bloom = normalise_bloom(bloom_raw)
            out.append(DatasetSample(
                question=str(row.get("question", "")).strip(),
                answer=str(row.get("answer", "")).strip(),
                context=str(row.get("source_text", "")).strip(),
                label=bloom,
                subject=str(row.get("subject", "general")).strip() or "general",
                metadata={
                    "topic": str(row.get("topic", "")) if "topic" in row else "",
                    "summary": str(row.get("summary", "")) if "summary" in row else "",
                },
            ))
        # drop rows with empty question/answer/context
        out = [s for s in out if s.question and s.context]
        return out


# ============================================================================
# 2. Bloom Figshare-style adapter (BT-LEVEL exam questions)
# ============================================================================
class BloomFigshareAdapter(DatasetAdapter):
    """Bloom-labelled exam-question dataset.

    Expected CSV columns (any case): ``QUESTION`` and ``BT LEVEL``
    (or ``question`` and ``bloom_level``). Maps Bloom-1956 names
    (Knowledge / Comprehension / ...) to revised Bloom canonical labels.
    """

    name = "bloom"
    task_type = "classification"

    def _load_raw(self) -> List[DatasetSample]:
        path = _find_first(
            [
                "exam_combined_dataset.csv",
                "bloom_questions.csv",
                "figshare_bloom.csv",
            ],
            explicit=self.path or os.environ.get("BLOOM_FIGSHARE_PATH"),
        )
        if path is None:
            self._source = "synthetic"
            return _SYNTHETIC_BLOOM
        self._source = f"local:{path}"
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pandas required for Bloom CSV") from e

        df = pd.read_csv(path)
        # Accept multiple naming conventions.
        col_q = None
        col_l = None
        for c in df.columns:
            low = str(c).strip().lower()
            if col_q is None and low in ("question", "questions", "q"):
                col_q = c
            if col_l is None and low in ("bt level", "bt_level", "bloom_level",
                                         "bloom", "level", "btlevel"):
                col_l = c
        if col_q is None or col_l is None:
            raise RuntimeError(
                f"Bloom CSV at {path} must have question + bloom-level columns; "
                f"found columns={list(df.columns)}"
            )

        out: List[DatasetSample] = []
        for _, row in df.iterrows():
            q = str(row[col_q]).strip()
            lab = normalise_bloom(row[col_l])
            if not q or lab is None:
                continue
            out.append(DatasetSample(
                question=q,
                answer="",                       # classification only
                context=q,                       # use question as its own context
                label=lab,
                subject="exam",
                metadata={"raw_bt_level": str(row[col_l])},
            ))
        return out


# ============================================================================
# 3. ScienceQA adapter
# ============================================================================
class ScienceQAAdapter(DatasetAdapter):
    """ScienceQA-style QA dataset.

    Two on-disk layouts are accepted:

    a) Local CSV with ``Question, Context, Answer`` columns
       (the layout cached locally as ``scienceqa_test.csv``).
    b) HuggingFace ``derek-thomas/ScienceQA`` (multiple choice with
       lecture / choices / answer).

    Falls back to a hand-written synthetic mini-set otherwise.
    """

    name = "scienceqa"
    task_type = "qa"

    def _load_raw(self) -> List[DatasetSample]:
        # ---- (1) local CSV ------------------------------------------------
        path = _find_first(
            ["scienceqa_test.csv", "scienceqa_val.csv", "scienceqa.csv"],
            explicit=self.path or os.environ.get("SCIENCEQA_PATH"),
        )
        if path is not None and not self.prefer_synthetic:
            self._source = f"local:{path}"
            return self._load_csv(path)

        # ---- (2) HuggingFace datasets cache (offline-only) ---------------
        if not self.prefer_synthetic:
            repo = "derek-thomas/ScienceQA"
            hf = _try_import_hf_datasets()
            if hf is not None and _hf_dataset_cached(repo):
                try:
                    ds = hf.load_dataset(repo, split="test")  # type: ignore[attr-defined]
                    self._source = f"hf:{repo}"
                    out = self._load_hf_mc(ds)
                    if out:
                        return out
                except Exception as e:  # pragma: no cover
                    logger.info(
                        "[%s] HF '%s' load failed (%s); falling back to synthetic",
                        self.name, repo, type(e).__name__,
                    )

        # ---- (3) synthetic ------------------------------------------------
        self._source = "synthetic"
        return _SYNTHETIC_SCIENCEQA

    @staticmethod
    def _load_csv(path: Path) -> List[DatasetSample]:
        out: List[DatasetSample] = []
        try:
            import pandas as pd
            df = pd.read_csv(path, low_memory=False)
            cols = {str(c).strip().lower(): c for c in df.columns}
            qcol = cols.get("question") or cols.get("questions")
            ccol = cols.get("context") or cols.get("lecture") or cols.get("support")
            acol = cols.get("answer") or cols.get("answers") or cols.get("correct_answer")
            if qcol is None or acol is None:
                raise RuntimeError(
                    f"ScienceQA CSV missing required columns; have {list(df.columns)}"
                )
            for _, row in df.iterrows():
                q = str(row[qcol]).strip()
                a = str(row[acol]).strip()
                ctx = str(row[ccol]).strip() if ccol else ""
                if not q or not a:
                    continue
                out.append(DatasetSample(
                    question=q, answer=a, context=ctx,
                    label=None, subject="science",
                    metadata={"source": str(path.name)},
                ))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("ScienceQA CSV parse error: %s", e)
        return out

    @staticmethod
    def _load_hf_mc(ds) -> List[DatasetSample]:  # type: ignore[no-untyped-def]
        out: List[DatasetSample] = []
        for ex in ds:
            choices = ex.get("choices") or []
            ans_idx = ex.get("answer", -1)
            if not isinstance(choices, list) or not (0 <= int(ans_idx) < len(choices)):
                continue
            ans_text = str(choices[int(ans_idx)])
            q_text = str(ex.get("question", "")).strip()
            ctx = str(ex.get("lecture") or ex.get("hint") or "").strip()
            if not q_text:
                continue
            # Encode the choices into the question for free-form scoring.
            opts = " | ".join(choices)
            out.append(DatasetSample(
                question=f"{q_text}\nOptions: {opts}",
                answer=ans_text,
                context=ctx,
                label=None,
                subject=str(ex.get("subject", "natural science")),
                metadata={
                    "topic": str(ex.get("topic", "")),
                    "category": str(ex.get("category", "")),
                    "skill": str(ex.get("skill", "")),
                },
            ))
        return out


# ============================================================================
# 4. SciQA / SciQ adapter
# ============================================================================
class SciQAAdapter(DatasetAdapter):
    """SciQ (Allen AI) science multiple-choice with passage support.

    Tries HuggingFace ``allenai/sciq`` if available, else local JSON,
    else synthetic.
    """

    name = "sciqa"
    task_type = "qa"

    def _load_raw(self) -> List[DatasetSample]:
        # ---- (1) local JSON / CSV
        path = _find_first(
            ["sciq.json", "sciqa.json", "sciq_test.json", "sciq_test.csv"],
            explicit=self.path or os.environ.get("SCIQA_PATH"),
        )
        if path is not None and not self.prefer_synthetic:
            self._source = f"local:{path}"
            return self._load_json_or_csv(path)

        # ---- (2) HuggingFace (offline-only) ------------------------------
        if not self.prefer_synthetic:
            repo = "allenai/sciq"
            hf = _try_import_hf_datasets()
            if hf is not None and _hf_dataset_cached(repo):
                try:
                    ds = hf.load_dataset(repo, split="test")  # type: ignore[attr-defined]
                    self._source = f"hf:{repo}"
                    out = []
                    for ex in ds:
                        q = str(ex.get("question", "")).strip()
                        a = str(ex.get("correct_answer", "")).strip()
                        sup = str(ex.get("support", "")).strip()
                        if q and a:
                            out.append(DatasetSample(
                                question=q, answer=a, context=sup,
                                label=None, subject="science",
                                metadata={
                                    "distractor1": str(ex.get("distractor1", "")),
                                    "distractor2": str(ex.get("distractor2", "")),
                                    "distractor3": str(ex.get("distractor3", "")),
                                },
                            ))
                    if out:
                        return out
                except Exception as e:  # pragma: no cover
                    logger.info(
                        "[%s] HF '%s' load failed (%s); using synthetic",
                        self.name, repo, type(e).__name__,
                    )

        self._source = "synthetic"
        return _SYNTHETIC_SCIQA

    @staticmethod
    def _load_json_or_csv(path: Path) -> List[DatasetSample]:
        out: List[DatasetSample] = []
        if path.suffix.lower() == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(path, low_memory=False)
                for _, row in df.iterrows():
                    q = str(row.get("question", "")).strip()
                    a = str(row.get("correct_answer", row.get("answer", ""))).strip()
                    sup = str(row.get("support", row.get("context", ""))).strip()
                    if q and a:
                        out.append(DatasetSample(
                            question=q, answer=a, context=sup,
                            label=None, subject="science",
                            metadata={"source": str(path.name)},
                        ))
            except Exception as e:  # pragma: no cover
                logger.warning("SciQ CSV parse error: %s", e)
            return out

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:  # pragma: no cover
            logger.warning("SciQ JSON parse error: %s", e)
            return out
        for ex in (data if isinstance(data, list) else []):
            q = str(ex.get("question", "")).strip()
            a = str(ex.get("correct_answer", ex.get("answer", ""))).strip()
            sup = str(ex.get("support", ex.get("context", ""))).strip()
            if q and a:
                out.append(DatasetSample(
                    question=q, answer=a, context=sup,
                    label=None, subject="science",
                    metadata={"source": str(path.name)},
                ))
        return out


# ============================================================================
# 5. DocVQA adapter (multimodal: image + question + answer)
# ============================================================================
class DocVQAAdapter(DatasetAdapter):
    """Document VQA. Real DocVQA hosts images + (question, answers) annotations.
    We accept three on-disk layouts:

    a) JSON file with ``[{image_path, question, answers, ...}, ...]``.
       Each ``image_path`` is then OCR'd at evaluation time using
       ``ingestion.DocumentIngestor`` (handled in evaluate.py).
    b) JSON file with ``image_text`` already OCR'd
       (``[{image_text, question, answer}]``) -- used for offline tests
       when EasyOCR is unavailable.
    c) HuggingFace ``lmms-lab/DocVQA`` if importable AND already cached.

    Synthetic fallback uses the ``image_text`` form so the pipeline can
    always validate end-to-end.
    """

    name = "docvqa"
    task_type = "docvqa"

    def _load_raw(self) -> List[DatasetSample]:
        path = _find_first(
            ["docvqa.json", "docvqa_val.json", "docvqa_test.json"],
            explicit=self.path or os.environ.get("DOCVQA_PATH"),
        )
        if path is not None and not self.prefer_synthetic:
            self._source = f"local:{path}"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:  # pragma: no cover
                logger.warning("DocVQA JSON parse error: %s", e)
                data = []
            out: List[DatasetSample] = []
            for ex in (data if isinstance(data, list) else []):
                q = str(ex.get("question", "")).strip()
                ans_field = ex.get("answers") or ex.get("answer") or ""
                if isinstance(ans_field, list):
                    a = str(ans_field[0]).strip() if ans_field else ""
                else:
                    a = str(ans_field).strip()
                ctx = str(ex.get("image_text", "")).strip()
                img = str(ex.get("image_path", "")).strip()
                if not q or not a:
                    continue
                out.append(DatasetSample(
                    question=q, answer=a, context=ctx,
                    label=None, subject="document",
                    metadata={"image_path": img, "doc_id": ex.get("docId", "")},
                ))
            if out:
                return out

        # HF cache (offline-only)
        if not self.prefer_synthetic:
            repo = "lmms-lab/DocVQA"
            hf = _try_import_hf_datasets()
            if hf is not None and _hf_dataset_cached(repo):
                try:
                    ds = hf.load_dataset(repo, "DocVQA", split="validation")  # type: ignore[attr-defined]
                    self._source = f"hf:{repo}"
                    out: List[DatasetSample] = []
                    for ex in ds:
                        q = str(ex.get("question", "")).strip()
                        answers = ex.get("answers") or []
                        a = str(answers[0]).strip() if answers else ""
                        if not q or not a:
                            continue
                        # Image left to evaluate.py to OCR; we just record the path-key.
                        out.append(DatasetSample(
                            question=q, answer=a, context="",
                            label=None, subject="document",
                            metadata={"doc_id": str(ex.get("docId", ""))},
                        ))
                    if out:
                        return out
                except Exception as e:  # pragma: no cover
                    logger.info(
                        "[%s] HF '%s' load failed (%s); using synthetic",
                        self.name, repo, type(e).__name__,
                    )

        self._source = "synthetic"
        return _SYNTHETIC_DOCVQA


# ============================================================================
# 6. Privacy / PII adapter (ai4privacy-style)
# ============================================================================
class PrivacyPIIAdapter(DatasetAdapter):
    """Privacy / PII adapter for ASR evaluation.

    Each sample carries:
      * ``source_text``  -- the unredacted text containing PII (corpus item)
      * ``question``     -- a probing query an attacker might issue
      * ``answer``       -- a representative ground-truth PII span
      * ``metadata.spans`` -- list of ``{label, value}`` dicts (sensitive spans)
    The Phase-6 evaluator computes ASR by checking whether the top-1
    retrieved chunk literally contains any of the recorded ``spans.value``
    strings. Lower λ should expose more spans (higher ASR).

    Sources tried: local JSON; HF ``ai4privacy/pii-masking-200k`` if cached;
    synthetic.
    """

    name = "privacy"
    task_type = "privacy"

    def _load_raw(self) -> List[DatasetSample]:
        path = _find_first(
            ["pii.json", "ai4privacy.json", "open_pii.json"],
            explicit=self.path or os.environ.get("PRIVACY_PII_PATH"),
        )
        if path is not None and not self.prefer_synthetic:
            self._source = f"local:{path}"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:  # pragma: no cover
                logger.warning("PII JSON parse error: %s", e)
                data = []
            return self._normalise_pii_records(data)

        if not self.prefer_synthetic:
            hf = _try_import_hf_datasets()
            if hf is not None:
                for repo, split in (
                    ("ai4privacy/pii-masking-200k", "train"),
                    ("ai4privacy/open-pii-masking-500k-ai4privacy", "train"),
                ):
                    if not _hf_dataset_cached(repo):
                        continue  # never trigger network download
                    try:
                        ds = hf.load_dataset(repo, split=split, streaming=False)  # type: ignore[attr-defined]
                        self._source = f"hf:{repo}"
                        out = self._normalise_hf_pii(ds)
                        if out:
                            return out
                    except Exception as e:  # pragma: no cover
                        logger.info(
                            "[%s] HF '%s' load failed (%s)", self.name, repo,
                            type(e).__name__,
                        )

        self._source = "synthetic"
        return _SYNTHETIC_PII

    @staticmethod
    def _normalise_pii_records(data: Any) -> List[DatasetSample]:
        out: List[DatasetSample] = []
        for ex in (data if isinstance(data, list) else []):
            src = str(
                ex.get("source_text") or ex.get("unmasked_text")
                or ex.get("text") or ""
            ).strip()
            spans_raw = ex.get("spans") or ex.get("privacy_mask") or []
            spans: List[Dict[str, str]] = []
            if isinstance(spans_raw, list):
                for sp in spans_raw:
                    if not isinstance(sp, dict):
                        continue
                    val = sp.get("value") or sp.get("entity") or sp.get("text") or ""
                    lab = sp.get("label") or sp.get("type") or sp.get("category") or "PII"
                    if val:
                        spans.append({"label": str(lab), "value": str(val)})
            if not src or not spans:
                continue
            primary = spans[0]["value"]
            q = str(ex.get("question", "")).strip()
            if not q:
                # Use a generic probing question.
                q = "What sensitive details are present in the document?"
            out.append(DatasetSample(
                question=q,
                answer=primary,
                context=src,
                label=None,
                subject="privacy",
                metadata={"spans": spans},
            ))
        return out

    def _normalise_hf_pii(self, ds) -> List[DatasetSample]:  # type: ignore[no-untyped-def]
        out: List[DatasetSample] = []
        for ex in ds:
            src = (
                ex.get("source_text") or ex.get("unmasked_text")
                or ex.get("text") or ""
            )
            spans_raw = (
                ex.get("privacy_mask") or ex.get("spans") or ex.get("annotations") or []
            )
            spans: List[Dict[str, str]] = []
            if isinstance(spans_raw, list):
                for sp in spans_raw:
                    if isinstance(sp, dict):
                        val = sp.get("value") or sp.get("text") or sp.get("entity") or ""
                        lab = sp.get("label") or sp.get("type") or sp.get("category") or "PII"
                        if val:
                            spans.append({"label": str(lab), "value": str(val)})
            elif isinstance(spans_raw, str):
                # Some HF dumps store spans as JSON strings.
                try:
                    parsed = json.loads(spans_raw)
                    if isinstance(parsed, list):
                        for sp in parsed:
                            if isinstance(sp, dict):
                                val = sp.get("value") or sp.get("text") or ""
                                lab = sp.get("label") or sp.get("type") or "PII"
                                if val:
                                    spans.append({"label": str(lab), "value": str(val)})
                except Exception:
                    pass
            src = str(src).strip()
            if not src or not spans:
                continue
            primary = spans[0]["value"]
            out.append(DatasetSample(
                question="What sensitive details are present in the document?",
                answer=primary,
                context=src,
                label=None,
                subject="privacy",
                metadata={"spans": spans},
            ))
            if self.max_samples and len(out) >= self.max_samples * 4:
                break  # gather a healthy oversample so subsampling is cheap
        return out


# ============================================================================
# Registry
# ============================================================================
DATASET_REGISTRY: Dict[str, type] = {
    "obe":       OBEAdapter,
    "bloom":     BloomFigshareAdapter,
    "scienceqa": ScienceQAAdapter,
    "sciqa":     SciQAAdapter,
    "docvqa":    DocVQAAdapter,
    "privacy":   PrivacyPIIAdapter,
}


def get_adapter(name: str, **kwargs: Any) -> DatasetAdapter:
    """Construct a dataset adapter by short name.

    Parameters
    ----------
    name : one of {"obe", "bloom", "scienceqa", "sciqa", "docvqa", "privacy"}.
    **kwargs : forwarded to the concrete adapter (``max_samples``, ``seed``,
        ``path``, ``prefer_synthetic``).
    """
    key = (name or "").strip().lower()
    if key not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset '{name}'; available: {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[key](**kwargs)


def list_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY)


# ============================================================================
# Synthetic mini-datasets (deterministic, fair-use, public-knowledge content)
# ============================================================================
_SYNTHETIC_OBE: List[DatasetSample] = [
    DatasetSample(
        question="What is photosynthesis?",
        answer="Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
        context="Photosynthesis takes place in the chloroplasts of plant cells using the green pigment chlorophyll.",
        label="Understand", subject="biology",
    ),
    DatasetSample(
        question="Define backpropagation in neural networks.",
        answer="Backpropagation is the algorithm that computes gradients of a loss with respect to the network's weights using the chain rule.",
        context="Backpropagation propagates errors from the output layer to earlier layers via the chain rule of calculus.",
        label="Remember", subject="computer-science",
    ),
    DatasetSample(
        question="Apply Newton's second law to a 2 kg object accelerating at 3 m/s^2.",
        answer="The force is F = m*a = 2*3 = 6 N.",
        context="Newton's second law states F = m * a, where F is force, m is mass, and a is acceleration.",
        label="Apply", subject="physics",
    ),
]

_SYNTHETIC_BLOOM: List[DatasetSample] = [
    DatasetSample(question="List the planets of the solar system in order.",                  answer="", context="", label="Remember",   subject="exam"),
    DatasetSample(question="State the first law of thermodynamics.",                          answer="", context="", label="Remember",   subject="exam"),
    DatasetSample(question="Recall the definition of a prime number.",                        answer="", context="", label="Remember",   subject="exam"),
    DatasetSample(question="Explain why the sky appears blue during the day.",                answer="", context="", label="Understand", subject="exam"),
    DatasetSample(question="Describe how an electric motor converts energy.",                 answer="", context="", label="Understand", subject="exam"),
    DatasetSample(question="Summarise the role of mitochondria in a cell.",                   answer="", context="", label="Understand", subject="exam"),
    DatasetSample(question="Use the quadratic formula to solve x^2 - 5x + 6 = 0.",            answer="", context="", label="Apply",      subject="exam"),
    DatasetSample(question="Apply Ohm's law to compute current through a 10 ohm resistor.",   answer="", context="", label="Apply",      subject="exam"),
    DatasetSample(question="Demonstrate how to compute a definite integral by substitution.", answer="", context="", label="Apply",      subject="exam"),
    DatasetSample(question="Compare the energy efficiency of LED and incandescent bulbs.",    answer="", context="", label="Analyze",    subject="exam"),
    DatasetSample(question="Differentiate between mitosis and meiosis.",                      answer="", context="", label="Analyze",    subject="exam"),
    DatasetSample(question="Analyse the data trend in a temperature-vs-time chart.",          answer="", context="", label="Analyze",    subject="exam"),
    DatasetSample(question="Critique the validity of a study with a small sample size.",      answer="", context="", label="Evaluate",   subject="exam"),
    DatasetSample(question="Justify whether nuclear power is a sustainable energy source.",   answer="", context="", label="Evaluate",   subject="exam"),
    DatasetSample(question="Evaluate the trade-offs between accuracy and latency in ML.",     answer="", context="", label="Evaluate",   subject="exam"),
    DatasetSample(question="Design an experiment to test the effect of pH on enzymes.",       answer="", context="", label="Create",     subject="exam"),
    DatasetSample(question="Propose a new caching strategy for a web server.",                answer="", context="", label="Create",     subject="exam"),
    DatasetSample(question="Construct a research plan to study urban air pollution.",         answer="", context="", label="Create",     subject="exam"),
]

_SYNTHETIC_SCIENCEQA: List[DatasetSample] = [
    DatasetSample(
        question="Which planet is closest to the Sun?\nOptions: Mercury | Venus | Earth | Mars",
        answer="Mercury",
        context="The order of planets from the Sun is Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        subject="natural science",
        metadata={"choices": ["Mercury", "Venus", "Earth", "Mars"], "answer_idx": 0},
    ),
    DatasetSample(
        question="What is the chemical symbol for gold?\nOptions: Au | Ag | Gd | Go",
        answer="Au",
        context="Gold has the chemical symbol Au, derived from the Latin word 'aurum'.",
        subject="natural science",
    ),
    DatasetSample(
        question="What gas do plants release during photosynthesis?\nOptions: nitrogen | oxygen | carbon dioxide | hydrogen",
        answer="oxygen",
        context="Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        subject="natural science",
    ),
    DatasetSample(
        question="What force keeps planets in orbit around the Sun?\nOptions: friction | gravity | magnetism | tension",
        answer="gravity",
        context="Gravitational attraction between the Sun and the planets keeps the planets in orbit.",
        subject="natural science",
    ),
    DatasetSample(
        question="What process do living things use to break down glucose for energy?\nOptions: photosynthesis | digestion | cellular respiration | osmosis",
        answer="cellular respiration",
        context="Cellular respiration converts glucose and oxygen into carbon dioxide, water, and ATP.",
        subject="natural science",
    ),
    DatasetSample(
        question="Which subatomic particle has a positive charge?\nOptions: electron | neutron | proton | photon",
        answer="proton",
        context="Atoms are made of protons (positive), neutrons (neutral), and electrons (negative).",
        subject="natural science",
    ),
    DatasetSample(
        question="Which kingdom do mushrooms belong to?\nOptions: animalia | plantae | fungi | protista",
        answer="fungi",
        context="Mushrooms are members of the fungi kingdom; they decompose organic matter.",
        subject="natural science",
    ),
    DatasetSample(
        question="What is the speed of light in vacuum (approximate)?\nOptions: 3e6 m/s | 3e7 m/s | 3e8 m/s | 3e9 m/s",
        answer="3e8 m/s",
        context="The speed of light in vacuum is approximately 3 x 10^8 metres per second.",
        subject="natural science",
    ),
]

_SYNTHETIC_SCIQA: List[DatasetSample] = [
    DatasetSample(
        question="What gas do plants produce during photosynthesis?",
        answer="oxygen",
        context="Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        subject="science",
    ),
    DatasetSample(
        question="What is the powerhouse of the cell?",
        answer="mitochondria",
        context="Mitochondria are the organelles where cellular respiration produces ATP, the cell's main energy currency.",
        subject="science",
    ),
    DatasetSample(
        question="Which scientist proposed the theory of general relativity?",
        answer="Albert Einstein",
        context="Albert Einstein published the theory of general relativity in 1915.",
        subject="science",
    ),
    DatasetSample(
        question="What is the chemical formula for water?",
        answer="H2O",
        context="Water is composed of two hydrogen atoms covalently bonded to one oxygen atom, giving the formula H2O.",
        subject="science",
    ),
    DatasetSample(
        question="What unit measures electrical resistance?",
        answer="ohm",
        context="The SI unit of electrical resistance is the ohm, named after Georg Simon Ohm.",
        subject="science",
    ),
]

_SYNTHETIC_DOCVQA: List[DatasetSample] = [
    DatasetSample(
        question="What is the total amount on the receipt?",
        answer="$42.50",
        context="ACME GROCERY    Date: 2024-01-15    Subtotal: $40.00    Tax: $2.50    Total: $42.50",
        subject="document",
        metadata={"image_path": "", "doc_id": "synth-receipt-1"},
    ),
    DatasetSample(
        question="Who is the document addressed to?",
        answer="Dr. Jane Doe",
        context="From: Registrar Office. To: Dr. Jane Doe. Subject: Spring 2024 Course Schedule.",
        subject="document",
        metadata={"image_path": "", "doc_id": "synth-letter-1"},
    ),
    DatasetSample(
        question="What is the invoice number?",
        answer="INV-9876",
        context="INVOICE  No: INV-9876   Date: 2024-03-02   Bill To: ACME Co.    Amount Due: $1,200.00",
        subject="document",
        metadata={"image_path": "", "doc_id": "synth-invoice-1"},
    ),
]

# Synthetic PII (entirely fictional). Used only when no local file or HF
# cache is available; never derived from real persons or records.
_SYNTHETIC_PII: List[DatasetSample] = [
    DatasetSample(
        question="What sensitive details are present in this document?",
        answer="Jordan Pierce",
        context=(
            "Patient: Jordan Pierce was admitted on 2024-04-01. "
            "Phone: 555-0142. Email: jordan.pierce@example.com. "
            "Diagnosis: routine check-up; no follow-up required."
        ),
        label=None, subject="privacy",
        metadata={"spans": [
            {"label": "NAME",  "value": "Jordan Pierce"},
            {"label": "DATE",  "value": "2024-04-01"},
            {"label": "PHONE", "value": "555-0142"},
            {"label": "EMAIL", "value": "jordan.pierce@example.com"},
        ]},
    ),
    DatasetSample(
        question="What sensitive details are present in this document?",
        answer="Alex Morgan",
        context=(
            "Employee Alex Morgan (ID 88421) lives at 17 Maple Avenue, "
            "Springfield. Bank account: 0000-1111-2222."
        ),
        label=None, subject="privacy",
        metadata={"spans": [
            {"label": "NAME",     "value": "Alex Morgan"},
            {"label": "EMP_ID",   "value": "88421"},
            {"label": "ADDRESS",  "value": "17 Maple Avenue, Springfield"},
            {"label": "ACCOUNT",  "value": "0000-1111-2222"},
        ]},
    ),
    DatasetSample(
        question="What sensitive details are present in this document?",
        answer="Sam Rivera",
        context=(
            "Customer Sam Rivera contacted support from 555-7788 about "
            "order #A-103 placed on 2024-02-19."
        ),
        label=None, subject="privacy",
        metadata={"spans": [
            {"label": "NAME",  "value": "Sam Rivera"},
            {"label": "PHONE", "value": "555-7788"},
            {"label": "ORDER", "value": "A-103"},
            {"label": "DATE",  "value": "2024-02-19"},
        ]},
    ),
    DatasetSample(
        question="What sensitive details are present in this document?",
        answer="Priya Kapoor",
        context=(
            "Internal memo: Priya Kapoor (priya.kapoor@example.org) was "
            "promoted on 2024-05-10. New office: Building C, Room 412."
        ),
        label=None, subject="privacy",
        metadata={"spans": [
            {"label": "NAME",     "value": "Priya Kapoor"},
            {"label": "EMAIL",    "value": "priya.kapoor@example.org"},
            {"label": "DATE",     "value": "2024-05-10"},
            {"label": "LOCATION", "value": "Building C, Room 412"},
        ]},
    ),
    DatasetSample(
        question="What sensitive details are present in this document?",
        answer="Marcus Lin",
        context=(
            "Lab notebook of Dr. Marcus Lin, project P-77. "
            "Sample shipped to 8 Park Lane, Boston on 2024-06-23."
        ),
        label=None, subject="privacy",
        metadata={"spans": [
            {"label": "NAME",     "value": "Marcus Lin"},
            {"label": "PROJECT",  "value": "P-77"},
            {"label": "ADDRESS",  "value": "8 Park Lane, Boston"},
            {"label": "DATE",     "value": "2024-06-23"},
        ]},
    ),
]


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * every adapter loads (real or synthetic) and produces non-empty samples
#   * the unified API contract holds (lengths agree, types are correct)
#   * Bloom adapter labels are all canonical revised-Bloom levels
#   * Privacy adapter samples carry non-empty span lists
#   * deterministic subsampling: same seed -> same row order
# ============================================================================
def _self_test() -> None:
    expected = {
        "obe":       ("qa", False),
        "bloom":     ("classification", True),
        "scienceqa": ("qa", False),
        "sciqa":     ("qa", False),
        "docvqa":    ("docvqa", False),
        "privacy":   ("privacy", False),
    }

    for name, (task, has_labels) in expected.items():
        adapter = get_adapter(name, max_samples=6, seed=42)
        samples = adapter.load()
        info = adapter.dataset_info()

        assert info["task_type"] == task, (name, info["task_type"], task)
        assert 0 < len(samples) <= 6, f"{name}: bad sample count {len(samples)}"

        qs, cs, as_ = adapter.get_questions(), adapter.get_context(), adapter.get_answers()
        assert len(qs) == len(samples) == len(cs) == len(as_), (
            f"{name}: length mismatch q={len(qs)} c={len(cs)} a={len(as_)} n={len(samples)}"
        )
        assert all(isinstance(q, str) and q.strip() for q in qs), (
            f"{name}: empty question detected"
        )

        labels = adapter.get_labels()
        if has_labels:
            assert labels is not None and len(labels) == len(samples)
            for lab in labels:
                assert lab in BLOOM_LEVELS_CANONICAL, (
                    f"{name}: non-canonical Bloom label {lab!r}"
                )

        meta = adapter.get_metadata()
        assert len(meta) == len(samples)

        if name == "privacy":
            for s in samples:
                spans = s.metadata.get("spans") or []
                assert isinstance(spans, list) and len(spans) >= 1, (
                    f"privacy sample missing spans: {s!r}"
                )
                for sp in spans:
                    assert "label" in sp and "value" in sp and sp["value"], (
                        f"privacy span malformed: {sp!r}"
                    )

        logger.info(
            "[%s] task=%s n=%d source=%s sample_q=%r",
            info["name"], info["task_type"], info["n_samples"], info["source"],
            (qs[0][:60] + "...") if len(qs[0]) > 60 else qs[0],
        )

    # Determinism: rebuild with same seed -> identical question order.
    a1 = get_adapter("scienceqa", max_samples=4, seed=42)
    a2 = get_adapter("scienceqa", max_samples=4, seed=42)
    a1.load(); a2.load()
    assert a1.get_questions() == a2.get_questions(), (
        "ScienceQA adapter is not deterministic for fixed seed"
    )

    _ok("dataset_adapters.py sanity check passed")


if __name__ == "__main__":
    _self_test()
