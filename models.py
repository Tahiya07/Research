"""
models.py
==============================================================================
Phase-2 RAG generation system for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

Wraps a Qwen-1.5B GGUF (4-bit) model loaded via ``llama-cpp-python`` and
combines it with the privacy-aware retriever from Phase 1.

Pipeline
--------
1. Retrieve top-k chunks from :class:`retriever.PrivacyRetriever`.
2. Construct a structured prompt with explicit Bloom-level conditioning::

       [BOUNDED CONTEXT]
       {retrieved_chunks}

       [QUESTION]
       {query}

       [COGNITIVE LEVEL]
       {bloom_level_instruction}

       [INSTRUCTION]
       Answer strictly using provided context. Do not hallucinate.

3. Wrap that body in Qwen's ChatML template (system + user + assistant) and
   run greedy CPU inference (temperature=0.0, top_k=1, seed=42) for
   deterministic outputs.
4. Return the answer, the retrieved chunks, and the literal prompt body so
   downstream evaluation modules can audit the generation.

Constraints
-----------
* CPU only, < 1 GB peak RAM (mmap-backed weights, n_ctx=1024).
* No external APIs, no fine-tuning, no training in this phase.
* Phase-1 modules (``ingestion.py``, ``retriever.py``) are NOT modified.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
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
# Required deps (assumed installed)
# ----------------------------------------------------------------------------
from llama_cpp import Llama  # type: ignore[import-not-found]

# Phase-1 module (unchanged; imported as-is)
from retriever import PrivacyRetriever, RetrievalResult

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("models")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


# ----------------------------------------------------------------------------
# Console-safe success print
# ----------------------------------------------------------------------------
def _ok(msg: str) -> None:
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    mark = "\u2714" if "utf" in enc else "[OK]"
    try:
        print(f"{mark} {msg}")
    except UnicodeEncodeError:  # pragma: no cover
        print(f"[OK] {msg}")


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_N_CTX = 1024              # small context window -> tight RAM budget
DEFAULT_MAX_TOKENS = 256
DEFAULT_N_THREADS = max(1, (os.cpu_count() or 4) // 2)
DEFAULT_SEED = 42

SYSTEM_PROMPT = (
    "You are a careful academic assistant. "
    "Answer using ONLY the provided context. "
    "If the context is insufficient, reply exactly: "
    "\"I don't know based on the provided context.\" "
    "Be concise, factual, and never invent details."
)

# Bloom's revised taxonomy -> generation instruction
BLOOM_INSTRUCTIONS: Dict[str, str] = {
    "remember":   "Provide accurate factual recall directly from the context.",
    "understand": "Explain the concept clearly in your own words using the context.",
    "apply":      "Provide step-by-step reasoning that applies the context to the question.",
    "analyze":    "Decompose the question into parts and analyse each using the context.",
    "evaluate":   "Compare and contrast the relevant ideas in the context and judge them.",
    "create":     "Synthesise a new, coherent answer that combines ideas from the context.",
}

# Where to look for a Qwen GGUF if no explicit path is given.
CANDIDATE_GGUF_GLOBS: List[str] = [
    "./models/*.gguf",
    "./*.gguf",
    "~/models/*.gguf",
    "~/PycharmProjects/Thesis/models/*.gguf",
    "~/Downloads/*.gguf",
    "~/.cache/llama.cpp/*.gguf",
    "~/.cache/llama-cpp-python/*.gguf",
]


def _find_qwen_gguf(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate a Qwen GGUF file via explicit arg, env var, or known caches."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None

    env = os.environ.get("QWEN_GGUF_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p

    candidates: List[Path] = []
    for pattern in CANDIDATE_GGUF_GLOBS:
        for found in glob(str(Path(pattern).expanduser())):
            p = Path(found)
            if p.is_file():
                candidates.append(p)

    if not candidates:
        return None

    # Rank: prefer "qwen" in filename, then files closest to ~1 GB.
    def _rank(p: Path):
        name = p.name.lower()
        size = p.stat().st_size
        return (
            0 if "qwen" in name else 1,
            abs(size - 1_000_000_000),
        )

    candidates.sort(key=_rank)
    return candidates[0]


# ----------------------------------------------------------------------------
# Output container
# ----------------------------------------------------------------------------
@dataclass
class GenerationOutput:
    answer: str
    chunks: List[RetrievalResult]
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# RAGGenerator
# ----------------------------------------------------------------------------
class RAGGenerator:
    """Retrieval-Augmented Generation over Qwen-1.5B GGUF (CPU, 4-bit)."""

    def __init__(
        self,
        retriever: PrivacyRetriever,
        model_path: Optional[str] = None,
        n_ctx: int = DEFAULT_N_CTX,
        n_threads: int = DEFAULT_N_THREADS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: int = DEFAULT_SEED,
        llm: Optional[Llama] = None,
    ) -> None:
        if not isinstance(retriever, PrivacyRetriever):
            raise TypeError("retriever must be a PrivacyRetriever instance")
        if n_ctx < 256:
            raise ValueError("n_ctx must be >= 256")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if n_threads <= 0:
            raise ValueError("n_threads must be > 0")

        self.retriever = retriever
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)

        if llm is not None:
            self.llm = llm
            self.model_path = getattr(llm, "model_path", "<injected>")
            logger.info("RAGGenerator using injected Llama instance")
        else:
            mp = _find_qwen_gguf(model_path)
            if mp is None:
                raise FileNotFoundError(
                    "Could not locate a Qwen GGUF model. "
                    "Set QWEN_GGUF_PATH or pass model_path=... "
                    f"Searched: {CANDIDATE_GGUF_GLOBS}"
                )
            self.model_path = str(mp)
            logger.info(
                f"Loading Qwen GGUF from {mp} "
                f"(n_ctx={self.n_ctx}, threads={self.n_threads})"
            )
            self.llm = Llama(
                model_path=str(mp),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                seed=self.seed,
                use_mmap=True,
                use_mlock=False,
                verbose=False,
                logits_all=False,
            )

    # ------------------------------------------------------------------ #
    # Bloom level handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm_bloom(level: str) -> str:
        if not isinstance(level, str):
            raise TypeError("bloom_level must be a string")
        key = level.strip().lower()
        if key not in BLOOM_INSTRUCTIONS:
            raise ValueError(
                f"unknown bloom level: {level!r}; "
                f"expected one of {list(BLOOM_INSTRUCTIONS)}"
            )
        return key

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def build_prompt(
        self,
        query: str,
        chunks: Sequence[Union[RetrievalResult, str]],
        bloom_level: str,
        safety_instruction: Optional[str] = None,
    ) -> str:
        """Build the structured RAG prompt body (without ChatML wrapping)."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        bl = self._norm_bloom(bloom_level)

        ctx_blocks: List[str] = []
        for i, c in enumerate(chunks, start=1):
            text = getattr(c, "text", None)
            if text is None:
                text = str(c)
            text = text.strip()
            if text:
                ctx_blocks.append(f"[{i}] {text}")
        ctx_str = "\n\n".join(ctx_blocks) if ctx_blocks else "(no context retrieved)"

        return (
            "[BOUNDED CONTEXT]\n"
            f"{ctx_str}\n\n"
            "[QUESTION]\n"
            f"{query.strip()}\n\n"
            "[COGNITIVE LEVEL]\n"
            f"Bloom level = {bl}. {BLOOM_INSTRUCTIONS[bl]}\n\n"
            "[INSTRUCTION]\n"
            "Answer strictly using provided context. Do not hallucinate."
            + (f"\n{str(safety_instruction).strip()}" if safety_instruction and str(safety_instruction).strip() else "")
        )

    def _to_chatml(self, body: str) -> str:
        """Wrap the prompt body in Qwen-Instruct ChatML."""
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{body}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _run_chatml(
        self,
        chatml: str,
        *,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, float]:
        """Run deterministic CPU inference on an already-built ChatML prompt."""
        reset_fn = getattr(self.llm, "reset", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"llm.reset() failed: {e}")

        t0 = time.time()
        out = self.llm(
            chatml,
            max_tokens=int(max_tokens or self.max_tokens),
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False,
            seed=self.seed,
        )
        elapsed = time.time() - t0

        try:
            text = out["choices"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected llama-cpp output: {out!r}") from e
        return text, elapsed

    def generate_from_chunks(
        self,
        query: str,
        chunks: Sequence[Union[RetrievalResult, str]],
        *,
        bloom_level: str = "understand",
        max_tokens: Optional[int] = None,
        safety_instruction: Optional[str] = None,
    ) -> GenerationOutput:
        """Run generation from caller-supplied context chunks."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        bl = self._norm_bloom(bloom_level)
        chunk_list = list(chunks)
        prompt = self.build_prompt(query, chunk_list, bl, safety_instruction=safety_instruction)
        text, elapsed = self._run_chatml(
            self._to_chatml(prompt),
            max_tokens=max_tokens,
        )
        return GenerationOutput(
            answer=text,
            chunks=chunk_list,
            prompt=prompt,
            metadata={
                "model_path": self.model_path,
                "bloom_level": bl,
                "k": len(chunk_list),
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "max_tokens": int(max_tokens or self.max_tokens),
                "elapsed_s": round(elapsed, 3),
            },
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate_answer(
        self,
        query: str,
        bloom_level: str = "understand",
        k: int = 5,
    ) -> GenerationOutput:
        """Retrieve, build prompt, run Qwen, return :class:`GenerationOutput`."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        bl = self._norm_bloom(bloom_level)

        chunks = self.retriever.retrieve(query, top_k=k)
        return self.generate_from_chunks(
            query,
            chunks,
            bloom_level=bl,
            max_tokens=self.max_tokens,
        )


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Validates:
#   * Prompt construction: required markers, query echo, Bloom conditioning.
#   * Bloom level validation (bad strings raise).
#   * Constructor input validation.
#   * Llama load from local Qwen GGUF (auto-discovered).
#   * generate_answer end-to-end on "What is photosynthesis?":
#       - returns GenerationOutput with k retrieved chunks
#       - answer is non-empty
#       - answer contains at least one keyword grounded in the context
#   * Determinism: same query twice -> identical answer string.
# ============================================================================
def _self_test() -> None:
    docs = [
        "Photosynthesis is the biological process by which green plants convert "
        "sunlight, water, and carbon dioxide into glucose and oxygen using the "
        "pigment chlorophyll inside chloroplasts.",
        "Backpropagation computes gradients through the chain rule for "
        "training neural networks via gradient descent.",
        "FAISS is a library that performs efficient similarity search and "
        "clustering of dense vectors at scale.",
        "Differential privacy provides statistical guarantees by adding "
        "calibrated noise so individuals cannot be re-identified.",
        "Bloom's taxonomy categorises cognitive learning objectives into "
        "Remember, Understand, Apply, Analyze, Evaluate, and Create.",
    ]

    # --- Phase-1 retriever (small mock corpus) ------------------------------
    retr = PrivacyRetriever(temperature=0.07, lambda_privacy=0.1)
    retr.build_index(docs)

    # --- Constructor input validation (no model load needed) ----------------
    class _DummyLlama:
        model_path = "<dummy>"

        def __call__(self, *args, **kwargs):
            return {"choices": [{"text": "Photosynthesis converts sunlight, "
                                         "water and carbon dioxide into "
                                         "glucose and oxygen."}]}

    dummy = RAGGenerator(retriever=retr, llm=_DummyLlama())  # type: ignore[arg-type]

    # 1. build_prompt structure ---------------------------------------------
    chunks = retr.retrieve("What is photosynthesis?", top_k=2)
    prompt = dummy.build_prompt("What is photosynthesis?", chunks, "understand")
    for marker in (
        "[BOUNDED CONTEXT]",
        "[QUESTION]",
        "[COGNITIVE LEVEL]",
        "[INSTRUCTION]",
    ):
        assert marker in prompt, f"prompt missing required marker: {marker}"
    assert "What is photosynthesis?" in prompt
    assert "Bloom level = understand" in prompt
    assert "Do not hallucinate" in prompt

    # 2. invalid bloom level -------------------------------------------------
    for bad in ("not-a-level", "", "EVALUATEX"):
        try:
            dummy.build_prompt("q?", chunks, bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for bloom={bad!r}")

    # 3. invalid query / k ---------------------------------------------------
    for bad_q in ("", "   "):
        try:
            dummy.generate_answer(bad_q)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for empty query")
    try:
        dummy.generate_answer("ok?", k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k=0")

    # 4. constructor parameter validation -----------------------------------
    for kwargs in (
        {"n_ctx": 64},
        {"max_tokens": 0},
        {"n_threads": 0},
    ):
        try:
            RAGGenerator(retriever=retr, llm=_DummyLlama(), **kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for kwargs={kwargs}")

    # bad retriever type
    try:
        RAGGenerator(retriever="not-a-retriever", llm=_DummyLlama())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for non-retriever")

    # ------------------------------------------------------------------ #
    # End-to-end with the REAL Qwen GGUF
    # ------------------------------------------------------------------ #
    rag = RAGGenerator(
        retriever=retr,
        n_ctx=1024,
        n_threads=DEFAULT_N_THREADS,
        max_tokens=80,        # keep latency reasonable in CI
        seed=DEFAULT_SEED,
    )

    res = rag.generate_answer(
        "What is photosynthesis?",
        bloom_level="understand",
        k=3,
    )

    assert isinstance(res, GenerationOutput)
    assert isinstance(res.answer, str) and len(res.answer.strip()) > 0, (
        f"empty answer: {res.answer!r}"
    )
    assert len(res.chunks) == 3, f"expected 3 chunks, got {len(res.chunks)}"
    assert "[BOUNDED CONTEXT]" in res.prompt
    assert "What is photosynthesis?" in res.prompt
    assert "model_path" in res.metadata

    # Grounding sanity: at least one context keyword appears in the answer.
    grounded_keywords = (
        "photo", "plant", "chloro", "sunlight",
        "glucose", "oxygen", "water", "carbon",
    )
    assert any(k in res.answer.lower() for k in grounded_keywords), (
        f"answer appears ungrounded in context: {res.answer!r}"
    )

    # Determinism: re-run the same query and expect the same string.
    res2 = rag.generate_answer(
        "What is photosynthesis?",
        bloom_level="understand",
        k=3,
    )
    assert res2.answer == res.answer, (
        "Generation is not deterministic:\n"
        f"  first : {res.answer!r}\n"
        f"  second: {res2.answer!r}"
    )

    logger.info(
        "generation latency: %.2fs (first), %.2fs (second), answer_len=%d",
        res.metadata["elapsed_s"],
        res2.metadata["elapsed_s"],
        len(res.answer),
    )

    _ok("RAGGenerator sanity check passed")


if __name__ == "__main__":
    _self_test()
