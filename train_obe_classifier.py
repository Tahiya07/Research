from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from classifier import (
    BLOOM_LEVELS,
    _find_figshare_exam_dataset,
    _find_obe_dataset,
    load_curated_bloom_dataset,
    load_obe_dataset,
)


SEED = 42
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1500,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def _split(
    texts: Sequence[str],
    labels: Sequence[str],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    x_train, x_val, y_train, y_val = train_test_split(
        list(texts),
        list(labels),
        test_size=0.2,
        random_state=SEED,
        stratify=list(labels),
    )
    return x_train, x_val, y_train, y_val


def _metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "report": classification_report(
            y_true,
            y_pred,
            labels=BLOOM_LEVELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def _question_conflict_stats(frame: pd.DataFrame, question_col: str, label_col: str) -> Dict[str, int]:
    clean = frame[[question_col, label_col]].dropna().copy()
    clean[question_col] = clean[question_col].astype(str).str.strip()
    clean[label_col] = clean[label_col].astype(str).str.strip()
    clean = clean[clean[question_col].str.len() > 0]
    grouped = clean.groupby(question_col)[label_col].nunique()
    return {
        "rows": int(len(clean)),
        "unique_questions": int(clean[question_col].nunique()),
        "duplicate_questions": int(len(clean) - clean[question_col].nunique()),
        "questions_with_multiple_labels": int((grouped > 1).sum()),
        "max_labels_per_question": int(grouped.max()) if len(grouped) else 0,
    }


def _audit_obe() -> Dict[str, object]:
    path = _find_obe_dataset()
    if path is None:
        return {"dataset": "obe", "available": False}
    df = pd.read_csv(path, low_memory=False)
    question_col = "question"
    label_col = "bloom_level" if "bloom_level" in df.columns else "bloom"
    stats = _question_conflict_stats(df, question_col, label_col)
    stats.update(
        {
            "dataset": "obe",
            "available": True,
            "path": str(path),
            "label_distribution": df[label_col].astype(str).value_counts().to_dict(),
        }
    )
    return stats


def _audit_figshare() -> Dict[str, object]:
    path = _find_figshare_exam_dataset()
    if path is None:
        return {"dataset": "figshare", "available": False}
    df = pd.read_csv(path, low_memory=False)
    question_col = "QUESTION"
    label_col = "BT LEVEL"
    stats = _question_conflict_stats(df, question_col, label_col)
    stats.update(
        {
            "dataset": "figshare",
            "available": True,
            "path": str(path),
            "label_distribution": df[label_col].astype(str).value_counts().to_dict(),
        }
    )
    return stats


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    audit = {
        "obe": _audit_obe(),
        "figshare": _audit_figshare(),
    }

    # Bloom training should use the curated clean corpus if present.
    # Today that corpus is seeded by the accepted Figshare exam dataset,
    # while noisier candidates are screened out in the report.
    texts, labels = load_curated_bloom_dataset(max_per_class=2000, seed=SEED)
    x_train, x_val, y_train, y_val = _split(texts, labels)

    pipe = _make_pipeline()
    t0 = time.time()
    pipe.fit(x_train, y_train)
    fit_s = time.time() - t0

    pred_train = pipe.predict(x_train)
    pred_val = pipe.predict(x_val)

    results = {
        "dataset": "curated_bloom_corpus",
        "seed": SEED,
        "method": {
            "input_columns": ["question"],
            "target_column": "bloom_level",
            "model": "tfidf + logistic_regression",
        },
        "dataset_audit": audit,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "fit_seconds": round(fit_s, 3),
        "train_metrics": _metrics(y_train, pred_train),
        "val_metrics": _metrics(y_val, pred_val),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = RESULTS_DIR / "bloom_dataset_audit.json"
    out_path = RESULTS_DIR / "curated_bloom_train_results.json"
    model_path = MODEL_DIR / "curated_bloom_tfidf.joblib"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(pipe, model_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
