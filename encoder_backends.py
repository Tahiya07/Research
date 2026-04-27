from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger("encoder_backends")


def resolve_local_transformers_path(model_name: str) -> Optional[str]:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)

    repo_candidates = [model_name]
    if "/" not in model_name:
        repo_candidates.append(f"sentence-transformers/{model_name}")

    cache_root = (
        Path(os.environ.get("HF_HUB_CACHE", ""))
        if os.environ.get("HF_HUB_CACHE")
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    for repo_id in repo_candidates:
        model_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
        refs_main = model_cache / "refs" / "main"
        snapshots = model_cache / "snapshots"
        if refs_main.is_file():
            snapshot = snapshots / refs_main.read_text(encoding="utf-8").strip()
            if snapshot.is_dir():
                return str(snapshot)
        if snapshots.is_dir():
            for snapshot in sorted(snapshots.iterdir(), reverse=True):
                if snapshot.is_dir() and (snapshot / "config.json").is_file():
                    return str(snapshot)
    return None


class StableTextEncoder:
    """Offline-first encoder with a transformers backend and hashing fallback."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        local_files_only: bool = True,
        n_features: int = 384,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.local_files_only = bool(local_files_only)
        self.n_features = int(n_features)
        self.backend = "hashing"
        self._tokenizer = None
        self._model = None
        self._hashing = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )
        self._init_backend()

    def _init_backend(self) -> None:
        model_ref = resolve_local_transformers_path(self.model_name) or self.model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_ref,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModel.from_pretrained(
                model_ref,
                local_files_only=self.local_files_only,
            )
            if torch is not None:
                self._model.to(self.device)
                self._model.eval()
            self.backend = "transformers"
            logger.info("StableTextEncoder backend=transformers model=%s", model_ref)
        except Exception as exc:
            self._tokenizer = None
            self._model = None
            self.backend = "hashing"
            logger.warning(
                "StableTextEncoder falling back to hashing backend for '%s': %s",
                self.model_name,
                exc,
            )

    def _encode_transformers(
        self,
        texts: Sequence[str],
        batch_size: int,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        assert self._tokenizer is not None and self._model is not None and torch is not None
        rows = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            toks = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                out = self._model(**toks)
                last_hidden = out.last_hidden_state
                mask = toks["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                arr = pooled.cpu().numpy().astype(np.float32, copy=False)
                rows.append(arr)
        emb = np.vstack(rows) if rows else np.zeros((0, self.n_features), dtype=np.float32)
        if normalize_embeddings and emb.size:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-12, None)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def _encode_hashing(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool,
    ) -> np.ndarray:
        mat = self._hashing.transform(list(texts))
        emb = mat.toarray().astype(np.float32, copy=False)
        if normalize_embeddings and emb.size:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-12, None)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        if len(texts) == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)
        if self.backend == "transformers" and torch is not None:
            return self._encode_transformers(texts, batch_size, normalize_embeddings)
        return self._encode_hashing(texts, normalize_embeddings)
