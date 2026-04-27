"""
ingestion.py
==============================================================================
Multi-modal document ingestion pipeline for the
"Lightweight Multi-Modal Tiny LLM Framework for Privacy-Preserving Academic
Assistance in University Environments" research codebase.

Responsibilities
----------------
1. Parse PDF documents with PyMuPDF (text-first, fully offline).
2. Extract text from images with EasyOCR (lazy-loaded, CPU only).
3. Chunk arbitrary text with a token window + overlap policy.
4. Provide a single :class:`DocumentIngestor` entry point that dispatches
   on file extension and returns a list of :class:`DocumentChunk` records.

Constraints
-----------
* CPU only, < 1 GB peak RAM, deterministic, no external API calls.
* No installation logic in this file (assumed dependencies).
* Reproducible: random / numpy / torch seeded to 42.
"""

from __future__ import annotations

import logging
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np


def _ok(msg: str) -> None:
    """Console-safe success print (falls back to ASCII on cp1252 stdouts)."""
    enc = (getattr(sys.stdout, "encoding", None) or "utf-8").lower()
    mark = "\u2714" if "utf" in enc else "[OK]"
    try:
        print(f"{mark} {msg}")
    except UnicodeEncodeError:  # pragma: no cover - defensive
        print(f"[OK] {msg}")

# ----------------------------------------------------------------------------
# Reproducibility (mandated global rule)
# ----------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
try:
    import torch  # noqa: F401  (only needed for seeding here)
    torch.manual_seed(42)
except Exception:  # pragma: no cover - torch is an assumed dep, but be safe
    torch = None  # type: ignore[assignment]


# ----------------------------------------------------------------------------
# Optional heavy dependencies (lazy / guarded)
# ----------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]
    _HAS_PYMUPDF = False

# EasyOCR is large; we only construct its Reader on first image call.
_EASYOCR_READER = None  # type: ignore[var-annotated]


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger("ingestion")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s: %(message)s",
    )


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class DocumentChunk:
    """A single chunk of text produced by the ingestion pipeline."""

    chunk_id: int
    source: str
    modality: str  # one of {"pdf", "image", "text"}
    page: Optional[int]
    text: str

    def __len__(self) -> int:  # convenience: len(chunk) == #chars
        return len(self.text)


# ----------------------------------------------------------------------------
# Main ingestor
# ----------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_TEXT_EXTS = {".txt", ".md"}


class DocumentIngestor:
    """End-to-end multi-modal document ingestion (PDF + OCR + chunking)."""

    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        languages: Optional[Sequence[str]] = None,
        normalize_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.languages = list(languages) if languages else ["en"]
        self.normalize_whitespace = bool(normalize_whitespace)

    # ------------------------------------------------------------------ #
    # Text utilities
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_iter(self, text: str) -> Iterable[str]:
        """Token-window chunker with overlap.

        We use whitespace tokens (a deterministic, model-agnostic proxy for
        sub-word tokens). Each yielded chunk has at most ``chunk_size`` tokens
        and consecutive chunks share ``chunk_overlap`` tokens.
        """
        if not text:
            return
        tokens = text.split()
        if not tokens:
            return
        size = self.chunk_size
        step = max(1, size - self.chunk_overlap)
        n = len(tokens)
        start = 0
        while start < n:
            window = tokens[start : start + size]
            if not window:
                break
            yield " ".join(window)
            if start + size >= n:
                break
            start += step

    # ------------------------------------------------------------------ #
    # PDF
    # ------------------------------------------------------------------ #
    def load_pdf(self, path: Union[str, Path]) -> List[DocumentChunk]:
        """Parse a PDF and return chunked text per page."""
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing.")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(str(path))
        pages: List[tuple[int, str]] = []
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text") or ""
                if self.normalize_whitespace:
                    text = self._normalize(text)
                if text:
                    pages.append((page_idx + 1, text))
        finally:
            doc.close()

        chunks: List[DocumentChunk] = []
        cid = 0
        for page_num, text in pages:
            for piece in self._chunk_iter(text):
                chunks.append(
                    DocumentChunk(
                        chunk_id=cid,
                        source=str(path),
                        modality="pdf",
                        page=page_num,
                        text=piece,
                    )
                )
                cid += 1
        return chunks

    # ------------------------------------------------------------------ #
    # Image / OCR
    # ------------------------------------------------------------------ #
    def _get_easyocr_reader(self):
        """Lazily build the EasyOCR reader (CPU only)."""
        global _EASYOCR_READER
        if _EASYOCR_READER is None:
            try:
                import easyocr  # local, lazy import
            except ImportError as e:  # pragma: no cover - assumed dep
                raise ImportError(
                    "easyocr is required for image ingestion. "
                    "Install easyocr to enable OCR."
                ) from e
            _EASYOCR_READER = easyocr.Reader(
                self.languages, gpu=False, verbose=False
            )
        return _EASYOCR_READER

    def load_image(self, path: Union[str, Path]) -> List[DocumentChunk]:
        """OCR an image and return chunked text."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        reader = self._get_easyocr_reader()
        result = reader.readtext(str(path), detail=0, paragraph=True)
        text = "\n".join(result).strip()
        if self.normalize_whitespace:
            text = self._normalize(text)

        chunks: List[DocumentChunk] = []
        cid = 0
        for piece in self._chunk_iter(text):
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    source=str(path),
                    modality="image",
                    page=None,
                    text=piece,
                )
            )
            cid += 1
        return chunks

    # ------------------------------------------------------------------ #
    # Plain text
    # ------------------------------------------------------------------ #
    def chunk_text(
        self,
        text: str,
        source: str = "<inline>",
        modality: str = "text",
    ) -> List[DocumentChunk]:
        """Chunk an arbitrary string and return :class:`DocumentChunk` records."""
        if self.normalize_whitespace:
            text = self._normalize(text)
        chunks: List[DocumentChunk] = []
        cid = 0
        for piece in self._chunk_iter(text):
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    source=source,
                    modality=modality,
                    page=None,
                    text=piece,
                )
            )
            cid += 1
        return chunks

    # ------------------------------------------------------------------ #
    # Top-level dispatch
    # ------------------------------------------------------------------ #
    def process(self, path: Union[str, Path]) -> List[DocumentChunk]:
        """Dispatch on file extension and run the appropriate parser."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(path)
        if ext in SUPPORTED_IMAGE_EXTS:
            return self.load_image(path)
        if ext in SUPPORTED_TEXT_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return self.chunk_text(f.read(), source=str(path), modality="text")
        raise ValueError(f"Unsupported file type: {ext}")


# ============================================================================
# SELF-TEST
# ----------------------------------------------------------------------------
# Runs when this module is executed directly. Validates:
#   * chunk_text behaviour: count, max length, overlap consistency.
#   * PDF parsing on a synthetic in-memory PDF (PyMuPDF round trip).
#   * dispatcher behaviour for .txt files.
#   * input-validation errors are raised as expected.
# OCR is skipped if easyocr is not present (graceful degradation).
# ============================================================================
def _build_dummy_pdf(path: Path) -> bool:
    if not _HAS_PYMUPDF:
        return False
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Lightweight Tiny LLM Framework. " * 20)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Privacy preserving retrieval module test page.")
    doc.save(str(path))
    doc.close()
    return True


def _self_test() -> None:
    ing = DocumentIngestor(chunk_size=20, chunk_overlap=5)

    # 1. chunk_text on a deterministic synthetic string ----------------------
    txt = " ".join(f"tok{i}" for i in range(95))
    cks = ing.chunk_text(txt, source="dummy", modality="text")
    assert len(cks) >= 1, "chunk_text returned no chunks"
    for c in cks:
        n_tokens = len(c.text.split())
        assert 1 <= n_tokens <= ing.chunk_size, (
            f"chunk size violation: got {n_tokens}, max {ing.chunk_size}"
        )
        assert c.modality == "text"
        assert c.source == "dummy"

    # consecutive chunks must share at least `overlap` tokens (set-intersection)
    if len(cks) >= 2:
        a = cks[0].text.split()
        b = cks[1].text.split()
        shared = len(set(a[-ing.chunk_overlap :]) & set(b[: ing.chunk_overlap]))
        assert shared >= 1, "overlap policy broken: expected shared tokens"

    # IDs are contiguous and start at 0
    assert [c.chunk_id for c in cks] == list(range(len(cks))), "chunk ids broken"

    # 2. PDF round trip ------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "sample.pdf"
        built = _build_dummy_pdf(pdf_path)
        if built:
            chunks = ing.load_pdf(pdf_path)
            assert len(chunks) >= 1, "PDF chunks empty"
            assert any("Tiny" in c.text for c in chunks), "PDF text missing"
            assert all(c.modality == "pdf" for c in chunks)
            assert all(c.page is not None for c in chunks)
        else:
            logger.warning("PyMuPDF unavailable; skipped PDF self-test.")

    # 3. process() dispatch on .txt -----------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "doc.txt"
        p.write_text("hello world " * 50, encoding="utf-8")
        chunks = ing.process(p)
        assert len(chunks) >= 1
        assert chunks[0].modality == "text"
        assert "hello" in chunks[0].text

    # 4. unsupported extension raises ---------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "x.unknown"
        bad.write_text("nope", encoding="utf-8")
        try:
            ing.process(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unsupported ext")

    # 5. invalid constructor parameters -------------------------------------
    for kwargs in ({"chunk_size": 0}, {"chunk_size": 10, "chunk_overlap": 10}):
        try:
            DocumentIngestor(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for kwargs={kwargs}")

    _ok("ingestion.py sanity check passed")


if __name__ == "__main__":
    _self_test()
