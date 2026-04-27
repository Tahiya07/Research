from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st

from classifier import BLOOM_LEVELS, DEFAULT_WEIGHTS_PATH, BloomLDLClassifier
from evaluate import (
    DEFAULT_N_CTX,
    DEFAULT_N_THREADS,
    exact_match,
    measure_model_file_mb,
    measure_rss_mb,
    measure_uss_mb,
    meteor_lite,
    rouge_l,
    token_f1,
)
from ingestion import DocumentChunk, DocumentIngestor
from models import RAGGenerator
from retriever import PrivacyRetriever, RetrievalResult
from summarizer import CognitiveSummarizer
from uncertainty import UncertaintyEngine


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

APP_TITLE = "Lightweight Multi-Modal Tiny LLM Demo"
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "md"]


def _init_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.caption(
        "Upload PDF/image/text sources, build a local retrieval corpus, and test QA or "
        "summarization with Bloom-level uncertainty, retrieval traces, and runtime metrics."
    )


def _runtime() -> Dict[str, Any]:
    if "demo_runtime" not in st.session_state:
        ingestor = DocumentIngestor(chunk_size=220, chunk_overlap=32)
        retriever = PrivacyRetriever(lambda_privacy=0.5)
        classifier = BloomLDLClassifier.load(DEFAULT_WEIGHTS_PATH, encoder=retriever.model)
        generator = RAGGenerator(
            retriever=retriever,
            n_ctx=DEFAULT_N_CTX,
            n_threads=DEFAULT_N_THREADS,
            max_tokens=160,
        )
        summarizer = CognitiveSummarizer(
            retriever=retriever,
            generator=generator,
            classifier=classifier,
            hierarchical=True,
            per_chunk_max_tokens=64,
        )
        uncertainty = UncertaintyEngine(K=len(BLOOM_LEVELS), n_bins=10)
        st.session_state.demo_runtime = {
            "ingestor": ingestor,
            "retriever": retriever,
            "classifier": classifier,
            "generator": generator,
            "summarizer": summarizer,
            "uncertainty": uncertainty,
        }
    return st.session_state.demo_runtime


def _ingest_uploaded_files(
    ingestor: DocumentIngestor,
    uploads: Sequence[Any],
    pasted_text: str,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    temp_paths: List[Path] = []
    try:
        for upload in uploads:
            suffix = Path(upload.name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getbuffer())
                tmp_path = Path(tmp.name)
            temp_paths.append(tmp_path)
            chunks.extend(ingestor.process(tmp_path))
            for c in chunks[-max(1, len(chunks)):]:
                if c.source == str(tmp_path):
                    c.source = upload.name
        if pasted_text.strip():
            chunks.extend(
                ingestor.chunk_text(
                    pasted_text,
                    source="<pasted_text>",
                    modality="text",
                )
            )
        return chunks
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass


def _set_active_corpus(
    runtime: Dict[str, Any],
    chunks: Sequence[DocumentChunk],
    lambda_privacy: float,
) -> None:
    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        raise ValueError("No usable text chunks were extracted from the uploaded inputs.")
    retriever: PrivacyRetriever = runtime["retriever"]
    retriever.lambda_privacy = float(lambda_privacy)
    retriever.build_index(texts)
    st.session_state.active_chunks = list(chunks)
    st.session_state.active_chunk_texts = texts
    st.session_state.active_sources = sorted({c.source for c in chunks})
    st.session_state.corpus_ready = True


def _preview_chunk_table(chunks: Sequence[DocumentChunk]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in chunks[:18]:
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "modality": c.modality,
                "page": c.page,
                "chars": len(c.text),
                "preview": c.text[:180] + ("..." if len(c.text) > 180 else ""),
            }
        )
    return rows


def _run_qa(runtime: Dict[str, Any], query: str, bloom_level: str, top_k: int) -> Dict[str, Any]:
    generator: RAGGenerator = runtime["generator"]
    t0 = time.perf_counter()
    output = generator.generate_answer(query=query, bloom_level=bloom_level, k=top_k)
    elapsed = time.perf_counter() - t0
    return {
        "text": output.answer,
        "chunks": output.chunks,
        "prompt": output.prompt,
        "latency_s": elapsed,
        "metadata": output.metadata,
    }


def _run_summary(
    runtime: Dict[str, Any],
    query: str,
    top_k: int,
    max_tokens: int,
) -> Dict[str, Any]:
    summarizer: CognitiveSummarizer = runtime["summarizer"]
    t0 = time.perf_counter()
    output = summarizer.summarize(query=query, k=top_k, max_tokens=max_tokens)
    elapsed = time.perf_counter() - t0
    return {
        "text": output.summary,
        "chunks": output.chunks,
        "prompt": output.prompt,
        "latency_s": elapsed,
        "metadata": output.metadata,
    }


def _reference_metrics(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "exact_match": float(exact_match(prediction, reference)),
        "token_f1": float(token_f1(prediction, reference)),
        "rouge_l": float(rouge_l(prediction, reference)),
        "meteor_lite": float(meteor_lite(prediction, reference)),
    }


def _show_retrieval_trace(chunks: Sequence[RetrievalResult]) -> None:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "privacy_score": round(float(chunk.privacy_score), 4),
                "cosine": round(float(chunk.cosine), 4),
                "infonce_risk": round(float(chunk.infonce_risk), 4),
                "preview": chunk.text[:220] + ("..." if len(chunk.text) > 220 else ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def main() -> None:
    _init_page()

    try:
        runtime = _runtime()
    except Exception as exc:
        st.error(f"Failed to initialize the local demo stack: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Demo Controls")
        lambda_privacy = st.slider("Privacy lambda", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=8, value=4, step=1)
        max_tokens = st.slider("Max generation tokens", min_value=48, max_value=256, value=160, step=16)
        mode = st.radio("Task", options=["Question Answering", "Summarization"], index=0)
        st.caption(
            "Images rely on local OCR support. If OCR dependencies are unavailable, use PDF or pasted text."
        )

    uploads = st.file_uploader(
        "Upload academic sources",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        help="Supported: PDF, image, TXT, and Markdown files.",
    )
    pasted_text = st.text_area(
        "Or paste text directly",
        height=180,
        placeholder="Paste notes, lecture text, textbook excerpts, or any academic content here.",
    )

    col_build, col_status = st.columns([1, 2])
    with col_build:
        build_clicked = st.button("Build / Refresh Corpus", use_container_width=True)
    with col_status:
        if st.session_state.get("corpus_ready"):
            st.success("Corpus ready for retrieval and generation.")
        else:
            st.info("Build the corpus after uploading or pasting source material.")

    if build_clicked:
        try:
            chunks = _ingest_uploaded_files(runtime["ingestor"], uploads or [], pasted_text)
            _set_active_corpus(runtime, chunks, lambda_privacy=lambda_privacy)
            st.success(f"Indexed {len(chunks)} chunks from {len(st.session_state.active_sources)} source(s).")
        except Exception as exc:
            st.error(f"Corpus build failed: {exc}")

    if st.session_state.get("corpus_ready"):
        chunks: List[DocumentChunk] = st.session_state.active_chunks
        modalities = Counter(c.modality for c in chunks)
        stat_cols = st.columns(4)
        stat_cols[0].metric("Sources", len(st.session_state.active_sources))
        stat_cols[1].metric("Chunks", len(chunks))
        stat_cols[2].metric("Modalities", ", ".join(f"{k}:{v}" for k, v in sorted(modalities.items())))
        stat_cols[3].metric("Private RAM (USS)", f"{measure_uss_mb():.1f} MB")

        with st.expander("Corpus Preview", expanded=False):
            st.dataframe(_preview_chunk_table(chunks), use_container_width=True, hide_index=True)

    query_label = "Ask a question" if mode == "Question Answering" else "Request a summary"
    query_placeholder = (
        "Example: Explain the main idea and give one application."
        if mode == "Question Answering"
        else "Example: Summarize this topic for an undergraduate learner."
    )
    query = st.text_area(query_label, height=120, placeholder=query_placeholder)
    reference = st.text_area(
        "Optional reference answer / summary for scoring",
        height=120,
        placeholder="Paste a gold answer or summary to compute EM/F1/ROUGE-L/METEOR-lite.",
    )

    if st.button("Run Inference", type="primary", use_container_width=True):
        if not st.session_state.get("corpus_ready"):
            st.error("Build the corpus first.")
            st.stop()
        if not query.strip():
            st.error("Enter a question or summary request first.")
            st.stop()

        runtime["retriever"].lambda_privacy = float(lambda_privacy)
        classifier: BloomLDLClassifier = runtime["classifier"]
        uncertainty: UncertaintyEngine = runtime["uncertainty"]

        try:
            class_out = classifier.predict(query)
            bloom_level = class_out.dominant_level.lower()
            bloom_uncertainty = uncertainty.compute_bloom_uncertainty(class_out.distribution)

            if mode == "Question Answering":
                result = _run_qa(runtime, query=query, bloom_level=bloom_level, top_k=top_k)
            else:
                result = _run_summary(runtime, query=query, top_k=top_k, max_tokens=max_tokens)

            rss_mb = measure_rss_mb()
            uss_mb = measure_uss_mb()
            model_mb = measure_model_file_mb(runtime["generator"])
            metrics = _reference_metrics(result["text"], reference) if reference.strip() else None

            st.subheader("Model Output")
            st.write(result["text"])

            metric_cols = st.columns(5)
            metric_cols[0].metric("Bloom Level", class_out.dominant_level)
            metric_cols[1].metric("Classifier Confidence", f"{class_out.confidence:.3f}")
            metric_cols[2].metric("Bloom Uncertainty", f"{bloom_uncertainty:.3f}")
            metric_cols[3].metric("Latency", f"{result['latency_s']:.2f} s")
            metric_cols[4].metric("Retrieved Chunks", len(result["chunks"]))

            infra_cols = st.columns(4)
            infra_cols[0].metric("RSS", f"{rss_mb:.1f} MB")
            infra_cols[1].metric("USS", f"{uss_mb:.1f} MB")
            infra_cols[2].metric("Model mmap", f"{model_mb:.1f} MB")
            infra_cols[3].metric("Privacy λ", f"{lambda_privacy:.2f}")

            if metrics:
                st.subheader("Reference-Based Quality Metrics")
                ref_cols = st.columns(4)
                ref_cols[0].metric("EM", f"{metrics['exact_match']:.3f}")
                ref_cols[1].metric("Token F1", f"{metrics['token_f1']:.3f}")
                ref_cols[2].metric("ROUGE-L", f"{metrics['rouge_l']:.3f}")
                ref_cols[3].metric("METEOR-lite", f"{metrics['meteor_lite']:.3f}")

            with st.expander("Bloom Distribution", expanded=False):
                rows = [
                    {"level": level, "probability": round(float(prob), 4)}
                    for level, prob in zip(BLOOM_LEVELS, class_out.distribution.tolist())
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)

            with st.expander("Retrieved Contexts", expanded=True):
                _show_retrieval_trace(result["chunks"])

            with st.expander("Prompt / Audit Trace", expanded=False):
                st.code(result["prompt"], language="text")

            with st.expander("Run Metadata", expanded=False):
                st.json(result["metadata"])

        except Exception as exc:
            st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
