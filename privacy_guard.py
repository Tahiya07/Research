from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ingestion import DocumentChunk
from retriever import RetrievalResult


STUDENT_REFUSAL = (
    "I can't reveal or reconstruct protected exam content. "
    "I can help with Bloom classification guidance, study concepts, or similarly scoped practice questions instead."
)

QUERY_RISK_PATTERNS = [
    r"\bverbatim\b",
    r"\bexact (text|question|paper)\b",
    r"\breconstruct\b",
    r"\brecover\b",
    r"\breveal\b",
    r"\bshow (me )?(the )?(question paper|exam paper|uploaded exam)\b",
    r"\bwhat (are|were) the questions\b",
    r"\bfull (question paper|exam paper|paper)\b",
    r"\blist (all|the) questions\b",
    r"\bcopy\b",
    r"\bquote\b",
    r"\bmoderation paper\b",
]

PROTECTED_ARTIFACT_PATTERNS = [
    r"\buploaded\b",
    r"\bprotected\b",
    r"\bexam\b",
    r"\bquestion paper\b",
    r"\bpaper\b",
    r"\bmoderation\b",
]


@dataclass
class PrivacyDecision:
    allowed: bool
    reason: str
    risk_score: float


def _norm_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _overlap_ratio(a: str, b: str) -> float:
    aa = Counter(_norm_tokens(a))
    bb = Counter(_norm_tokens(b))
    n = max(1, sum(aa.values()))
    return float(sum((aa & bb).values()) / n)


def assess_query_privacy_risk(query: str) -> PrivacyDecision:
    q = (query or "").strip()
    if not q:
        return PrivacyDecision(False, "empty_query", 1.0)
    hits = 0
    for pat in QUERY_RISK_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            hits += 1
    risk = min(1.0, 0.2 * hits)
    if hits > 0:
        return PrivacyDecision(False, "reconstruction_intent_detected", risk)
    return PrivacyDecision(True, "ok", risk)


def assess_student_query_against_protected_corpus(
    query: str,
    protected_chunks: Sequence[DocumentChunk],
) -> PrivacyDecision:
    base = assess_query_privacy_risk(query)
    if not base.allowed:
        return base
    union = protected_text_union(protected_chunks)
    if not union:
        return PrivacyDecision(True, "no_protected_corpus", 0.0)
    overlap = _overlap_ratio(query, union)
    mentions_artifact = any(
        re.search(pat, query or "", flags=re.IGNORECASE) for pat in PROTECTED_ARTIFACT_PATTERNS
    )
    if mentions_artifact and overlap >= 0.12:
        return PrivacyDecision(False, "artifact_reference_with_overlap", overlap)
    if overlap >= 0.85 and len(_norm_tokens(query)) >= 8:
        return PrivacyDecision(False, "query_overlap_high", overlap)
    return PrivacyDecision(True, "ok", overlap)


def partition_chunks(chunks: Sequence[DocumentChunk]) -> Dict[str, List[DocumentChunk]]:
    out = {"public": [], "protected": []}
    for chunk in chunks:
        level = "protected" if getattr(chunk, "access_level", "public") == "protected" else "public"
        out[level].append(chunk)
    return out


def allowed_chunks_for_role(
    requester_role: str,
    public_chunks: Sequence[DocumentChunk],
    protected_chunks: Sequence[DocumentChunk],
    access_scope: str,
) -> List[DocumentChunk]:
    role = requester_role.lower().strip()
    scope = access_scope.lower().strip()
    if role in {"teacher", "moderator", "admin"} and scope == "protected":
        return list(protected_chunks)
    return list(public_chunks)


def protected_text_union(chunks: Sequence[DocumentChunk], max_chars: int = 40_000) -> str:
    buf: List[str] = []
    total = 0
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        piece = text[:remaining]
        buf.append(piece)
        total += len(piece)
    return "\n".join(buf)


def screen_generation_output(
    requester_role: str,
    query: str,
    answer: str,
    protected_chunks: Sequence[DocumentChunk],
) -> PrivacyDecision:
    role = requester_role.lower().strip()
    if role in {"teacher", "moderator", "admin"}:
        return PrivacyDecision(True, "teacher_access", 0.0)
    risk = assess_student_query_against_protected_corpus(query, protected_chunks)
    if not risk.allowed:
        return risk
    union = protected_text_union(protected_chunks)
    if not union:
        return PrivacyDecision(True, "no_protected_corpus", 0.0)
    overlap = _overlap_ratio(answer, union)
    if overlap >= 0.70 and len(_norm_tokens(answer)) >= 8:
        return PrivacyDecision(False, "protected_overlap_high", overlap)
    return PrivacyDecision(True, "ok", overlap)


def policy_instruction(requester_role: str, access_scope: str) -> str:
    role = requester_role.lower().strip()
    scope = access_scope.lower().strip()
    if role in {"teacher", "moderator", "admin"} and scope == "protected":
        return (
            "This is a protected moderation workflow. Use the context only for classification, moderation support, "
            "or high-level analysis. Do not quote long spans verbatim."
        )
    return (
        "Never reveal, reconstruct, or quote protected exam content. If the request seeks exact exam wording, "
        "full question lists, or reconstruction, refuse and offer high-level study help instead."
    )
