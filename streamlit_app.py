from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import streamlit as st

from classifier import (
    BLOOM_LEVELS,
    DEFAULT_WEIGHTS_PATH,
    BloomLDLClassifier,
    LocalOBEClassifier,
    OBEClassifierOutput,
)
from evaluate import (
    DEFAULT_N_CTX,
    DEFAULT_N_THREADS,
    _apply_retrieval_governor,
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
from privacy_guard import (
    STUDENT_REFUSAL,
    allowed_chunks_for_role,
    assess_student_query_against_protected_corpus,
    partition_chunks,
    policy_instruction,
    screen_generation_output,
)
from retriever import PrivacyRetriever, RetrievalResult
from summarizer import CognitiveSummarizer
from uncertainty import UncertaintyEngine


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

APP_TITLE = "Lightweight Multi-Modal Tiny LLM Demo"
UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp", "txt", "md"]
DEFAULT_FAISS_POOL = 20


def _init_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="A",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title(APP_TITLE)
    st.caption(
        "Upload PDF/image/text sources, build a local retrieval corpus, classify exam "
        "questions into Bloom and OBE-aligned labels, and test bounded local QA or summarization."
    )


def _runtime() -> Dict[str, Any]:
    if "demo_runtime" not in st.session_state:
        ingestor = DocumentIngestor(chunk_size=220, chunk_overlap=32)
        retriever = PrivacyRetriever(lambda_privacy=0.5)
        protected_retriever = PrivacyRetriever(lambda_privacy=0.5, model=retriever.model)
        classifier = BloomLDLClassifier.load(DEFAULT_WEIGHTS_PATH, encoder=retriever.model)
        obe_classifier = LocalOBEClassifier(encoder=retriever.model)
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
            "protected_retriever": protected_retriever,
            "classifier": classifier,
            "obe_classifier": obe_classifier,
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
            prev_len = len(chunks)
            chunks.extend(ingestor.process(tmp_path))
            for c in chunks[prev_len:]:
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
    upload_scope: str,
    content_type: str,
) -> None:
    texts = [c.text for c in chunks if c.text.strip()]
    if not texts:
        raise ValueError("No usable text chunks were extracted from the uploaded inputs.")
    for chunk in chunks:
        chunk.access_level = "protected" if upload_scope == "protected" else "public"
        chunk.content_type = str(content_type)

    state_key = "protected_chunks" if upload_scope == "protected" else "public_chunks"
    retr_key = "protected_retriever" if upload_scope == "protected" else "retriever"
    existing: List[DocumentChunk] = list(st.session_state.get(state_key, []))
    existing.extend(chunks)
    texts = [c.text for c in existing if c.text.strip()]

    retriever: PrivacyRetriever = runtime[retr_key]
    retriever.lambda_privacy = float(lambda_privacy)
    retriever.build_index(texts)

    st.session_state[state_key] = existing
    st.session_state["public_corpus_ready"] = bool(st.session_state.get("public_chunks"))
    st.session_state["protected_corpus_ready"] = bool(st.session_state.get("protected_chunks"))
    st.session_state["corpus_ready"] = bool(
        st.session_state.get("public_corpus_ready") or st.session_state.get("protected_corpus_ready")
    )
    all_chunks = list(st.session_state.get("public_chunks", [])) + list(st.session_state.get("protected_chunks", []))
    st.session_state.active_chunks = all_chunks
    st.session_state.active_sources = sorted({c.source for c in all_chunks})


def _preview_chunk_table(chunks: Sequence[DocumentChunk]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for c in chunks[:18]:
        rows.append(
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "modality": c.modality,
                "page": c.page,
                "access": c.access_level,
                "content_type": c.content_type,
                "chars": len(c.text),
                "preview": c.text[:180] + ("..." if len(c.text) > 180 else ""),
            }
        )
    return rows


def _governed_chunks(
    retriever: PrivacyRetriever,
    query: str,
    top_k: int,
    governor_preset: str,
) -> List[RetrievalResult]:
    pool_n = max(DEFAULT_FAISS_POOL, int(top_k))
    pool = retriever.retrieve(query, top_k=pool_n, candidate_pool=pool_n)
    chunks, _ = _apply_retrieval_governor(
        pool,
        governor_preset,
        query,
        final_k=top_k,
        retr=retriever,
    )
    return chunks


def _run_qa(
    runtime: Dict[str, Any],
    query: str,
    bloom_level: str,
    top_k: int,
    governor_preset: str,
    retriever: PrivacyRetriever,
    safety_instruction: str,
) -> Dict[str, Any]:
    generator: RAGGenerator = runtime["generator"]
    t0 = time.perf_counter()
    chunks = _governed_chunks(retriever, query, top_k=top_k, governor_preset=governor_preset)
    output = generator.generate_from_chunks(query, chunks, bloom_level=bloom_level, safety_instruction=safety_instruction)
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
    governor_preset: str,
    retriever: PrivacyRetriever,
    safety_instruction: str,
) -> Dict[str, Any]:
    summarizer: CognitiveSummarizer = runtime["summarizer"]
    t0 = time.perf_counter()
    chunks = _governed_chunks(retriever, query, top_k=top_k, governor_preset=governor_preset)
    output = summarizer.summarize(
        query=query,
        k=top_k,
        max_tokens=max_tokens,
        retrieved_chunks=chunks,
        safety_instruction=safety_instruction,
    )
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


def _show_retrieval_trace(chunks: Sequence[RetrievalResult], protected_mode: bool) -> None:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "rank": chunk.rank,
                "doc_id": chunk.doc_id,
                "privacy_score": round(float(chunk.privacy_score), 4),
                "cosine": round(float(chunk.cosine), 4),
                "infonce_risk": round(float(chunk.infonce_risk), 4),
                "preview": (
                    "[protected snippet hidden]"
                    if protected_mode
                    else chunk.text[:220] + ("..." if len(chunk.text) > 220 else "")
                ),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _show_obe_result(out: OBEClassifierOutput) -> None:
    cols = st.columns(5)
    cols[0].metric("Bloom Level", out.bloom_level)
    cols[1].metric("Cognitive Skill", out.cognitive_skill)
    cols[2].metric("Subject", out.subject)
    cols[3].metric("Topic", out.topic)
    cols[4].metric("Confidence", f"{out.confidence:.2f}")

    cols2 = st.columns(4)
    cols2[0].metric("Subtopic", out.subtopic)
    cols2[1].metric("Difficulty", out.difficulty)
    cols2[2].metric("Source Type", out.source_type)
    cols2[3].metric("Language", out.language)

    with st.expander("Nearest Local OBE Examples", expanded=False):
        st.dataframe(out.nearest_examples, use_container_width=True, hide_index=True)


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
        requester_role = st.radio("Requester role", options=["Student", "Teacher / Moderator"], index=0)
        upload_scope = st.radio("Upload target", options=["Public Learning Corpus", "Protected Exam Corpus"], index=0)
        content_type = st.selectbox(
            "Uploaded content type",
            options=["study_material", "lecture_notes", "exam_paper", "moderation_material"],
            index=0,
        )
        mode = st.radio(
            "Task",
            options=["Question Answering", "Summarization", "Exam Question Classification"],
            index=0,
        )
        protected_mode = st.toggle("Protected exam mode", value=True)
        governor_preset = "strong" if protected_mode else "mild"
        search_scope = "protected" if requester_role == "Teacher / Moderator" and upload_scope == "Protected Exam Corpus" else "public"
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
            public_n = len(st.session_state.get("public_chunks", []))
            protected_n = len(st.session_state.get("protected_chunks", []))
            st.success(f"Corpora ready. Public chunks: {public_n}, Protected chunks: {protected_n}.")
        else:
            st.info("Build the corpus after uploading or pasting source material.")

    if build_clicked:
        try:
            chunks = _ingest_uploaded_files(runtime["ingestor"], uploads or [], pasted_text)
            _set_active_corpus(
                runtime,
                chunks,
                lambda_privacy=lambda_privacy,
                upload_scope="protected" if upload_scope == "Protected Exam Corpus" else "public",
                content_type=content_type,
            )
            st.success(f"Indexed {len(chunks)} chunks into the {upload_scope.lower()}.")
        except Exception as exc:
            st.error(f"Corpus build failed: {exc}")

    if st.session_state.get("corpus_ready") and mode != "Exam Question Classification":
        visible_chunks = allowed_chunks_for_role(
            "teacher" if requester_role == "Teacher / Moderator" else "student",
            st.session_state.get("public_chunks", []),
            st.session_state.get("protected_chunks", []),
            search_scope,
        )
        chunks: List[DocumentChunk] = visible_chunks
        modalities = Counter(c.modality for c in chunks)
        stat_cols = st.columns(4)
        stat_cols[0].metric("Visible Sources", len({c.source for c in chunks}))
        stat_cols[1].metric("Chunks", len(chunks))
        stat_cols[2].metric("Modalities", ", ".join(f"{k}:{v}" for k, v in sorted(modalities.items())))
        stat_cols[3].metric("Private RAM (USS)", f"{measure_uss_mb():.1f} MB")

        with st.expander("Corpus Preview", expanded=False):
            if protected_mode:
                st.info("Protected exam mode hides raw chunk previews.")
            else:
                st.dataframe(_preview_chunk_table(chunks), use_container_width=True, hide_index=True)

    if mode == "Question Answering":
        query_label = "Ask a question"
        query_placeholder = "Example: Explain the main idea and give one application."
    elif mode == "Summarization":
        query_label = "Request a summary"
        query_placeholder = "Example: Summarize this topic for an undergraduate learner."
    else:
        query_label = "Enter an exam question"
        query_placeholder = "Example: Compare TCP and UDP for reliability and latency trade-offs."

    query = st.text_area(query_label, height=120, placeholder=query_placeholder)
    reference = st.text_area(
        "Optional reference answer / summary for scoring",
        height=120,
        placeholder="Paste a gold answer or summary to compute EM/F1/ROUGE-L/METEOR-lite.",
    )

    if st.button("Run Inference", type="primary", use_container_width=True):
        public_chunks = list(st.session_state.get("public_chunks", []))
        protected_chunks = list(st.session_state.get("protected_chunks", []))
        role_key = "teacher" if requester_role == "Teacher / Moderator" else "student"
        visible_chunks = allowed_chunks_for_role(role_key, public_chunks, protected_chunks, search_scope)
        if mode != "Exam Question Classification" and not visible_chunks:
            st.error("Build the appropriate corpus first.")
            st.stop()
        if not query.strip():
            st.error("Enter a question or summary request first.")
            st.stop()

        runtime["retriever"].lambda_privacy = float(lambda_privacy)
        runtime["protected_retriever"].lambda_privacy = float(lambda_privacy)
        classifier: BloomLDLClassifier = runtime["classifier"]
        obe_classifier: LocalOBEClassifier = runtime["obe_classifier"]
        uncertainty: UncertaintyEngine = runtime["uncertainty"]

        try:
            query_policy = assess_student_query_against_protected_corpus(query, protected_chunks)
            if role_key == "student" and protected_chunks and not query_policy.allowed:
                st.error(STUDENT_REFUSAL)
                st.stop()

            class_out = classifier.predict(query)
            bloom_level = class_out.dominant_level.lower()
            bloom_uncertainty = uncertainty.compute_bloom_uncertainty(class_out.distribution)

            if mode == "Exam Question Classification":
                st.subheader("Exam Classification")
                _show_obe_result(obe_classifier.predict(query))
                with st.expander("Bloom Distribution", expanded=False):
                    rows = [
                        {"level": level, "probability": round(float(prob), 4)}
                        for level, prob in zip(BLOOM_LEVELS, class_out.distribution.tolist())
                    ]
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                st.stop()

            if mode == "Question Answering":
                result = _run_qa(
                    runtime,
                    query=query,
                    bloom_level=bloom_level,
                    top_k=top_k,
                    governor_preset=governor_preset,
                    retriever=runtime["protected_retriever"] if role_key == "teacher" and search_scope == "protected" else runtime["retriever"],
                    safety_instruction=policy_instruction(role_key, search_scope),
                )
            else:
                result = _run_summary(
                    runtime,
                    query=query,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    governor_preset=governor_preset,
                    retriever=runtime["protected_retriever"] if role_key == "teacher" and search_scope == "protected" else runtime["retriever"],
                    safety_instruction=policy_instruction(role_key, search_scope),
                )

            output_policy = screen_generation_output(role_key, query, result["text"], protected_chunks)
            if not output_policy.allowed:
                result["text"] = STUDENT_REFUSAL
                result["metadata"]["privacy_block_reason"] = output_policy.reason
                result["metadata"]["privacy_risk_score"] = round(float(output_policy.risk_score), 4)
                result["chunks"] = []
                result["prompt"] = "[protected prompt hidden after policy block]"

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
            infra_cols[3].metric("Privacy lambda", f"{lambda_privacy:.2f}")

            if role_key == "student" and protected_chunks:
                st.caption("Protected-corpus policy is active for student-facing requests.")

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
                _show_retrieval_trace(result["chunks"], protected_mode=protected_mode)

            with st.expander("Prompt / Audit Trace", expanded=False):
                if protected_mode:
                    st.info("Protected exam mode hides raw prompt bodies to reduce reconstruction risk.")
                else:
                    st.code(result["prompt"], language="text")

            with st.expander("Run Metadata", expanded=False):
                st.json(result["metadata"])

        except Exception as exc:
            st.error(f"Inference failed: {exc}")


if __name__ == "__main__":
    main()
