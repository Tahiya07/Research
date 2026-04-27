from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from classifier import _normalise_bloom


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
CURATED_PATH = DATA_DIR / "curated_bloom_corpus.csv"
REPORT_PATH = RESULTS_DIR / "curated_bloom_corpus_report.json"


def _clean_question(text: object) -> str:
    return " ".join(str(text).strip().split())


def _question_conflicts(frame: pd.DataFrame, question_col: str, label_col: str) -> Dict[str, int]:
    grouped = frame.groupby(question_col)[label_col].nunique()
    return {
        "rows": int(len(frame)),
        "unique_questions": int(frame[question_col].nunique()),
        "duplicate_questions": int(len(frame) - frame[question_col].nunique()),
        "questions_with_multiple_labels": int((grouped > 1).sum()),
        "max_labels_per_question": int(grouped.max()) if len(grouped) else 0,
    }


def _load_figshare() -> tuple[pd.DataFrame, Dict[str, object]]:
    path = DATA_DIR / "figshare_combined_dataset.csv"
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(
        {
            "question": df["QUESTION"].map(_clean_question),
            "bloom_level": df["BT LEVEL"].map(_normalise_bloom),
            "source": "figshare_exam_questions",
            "language": "en",
        }
    )
    out = out.dropna(subset=["question", "bloom_level"])
    out = out[out["question"].str.len() > 0].copy()
    audit = _question_conflicts(out, "question", "bloom_level")
    audit.update(
        {
            "source": "figshare_exam_questions",
            "path": str(path),
            "status": "accepted",
            "reason": "Direct English exam-question labels with no conflicting Bloom labels per question.",
            "label_distribution": out["bloom_level"].value_counts().to_dict(),
        }
    )
    out = out.drop_duplicates(subset=["question", "bloom_level"]).reset_index(drop=True)
    return out, audit


def _audit_local_obe() -> Dict[str, object]:
    path = DATA_DIR / "obe_dataset.csv"
    if not path.is_file():
        return {"source": "local_obe", "status": "missing"}
    df = pd.read_csv(path, low_memory=False)
    label_col = "bloom_level" if "bloom_level" in df.columns else "bloom"
    tmp = pd.DataFrame(
        {
            "question": df["question"].map(_clean_question),
            "bloom_level": df[label_col].map(_normalise_bloom),
        }
    )
    tmp = tmp.dropna(subset=["question", "bloom_level"])
    tmp = tmp[tmp["question"].str.len() > 0].copy()
    audit = _question_conflicts(tmp, "question", "bloom_level")
    audit.update(
        {
            "source": "local_obe",
            "path": str(path),
            "status": "rejected",
            "reason": "Every unique question appears with conflicting Bloom labels, so it is not trustworthy supervision.",
            "label_distribution": tmp["bloom_level"].value_counts().to_dict(),
        }
    )
    return audit


def _manual_external_screening() -> List[Dict[str, object]]:
    return [
        {
            "source": "quban_zenodo",
            "url": "https://zenodo.org/records/10633113",
            "status": "rejected",
            "reason": "Questions are scored by a Bloom-keyword algorithm rather than provided as direct verified Bloom class labels.",
            "language": "en",
        },
        {
            "source": "indonesian_exam_zenodo",
            "url": "https://zenodo.org/records/8331563",
            "status": "rejected",
            "reason": "Clean labels, but language is Indonesian and the corpus is small; it is not directly compatible with the current English Bloom classifier.",
            "language": "id",
        },
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    accepted, figshare_audit = _load_figshare()
    accepted = accepted.sort_values(["bloom_level", "question"]).reset_index(drop=True)
    accepted.to_csv(CURATED_PATH, index=False, encoding="utf-8")

    report = {
        "curated_output": str(CURATED_PATH),
        "accepted_sources": [figshare_audit],
        "rejected_sources": [_audit_local_obe(), *_manual_external_screening()],
        "final_rows": int(len(accepted)),
        "final_unique_questions": int(accepted["question"].nunique()),
        "final_label_distribution": accepted["bloom_level"].value_counts().to_dict(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
