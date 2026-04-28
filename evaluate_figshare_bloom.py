from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from classifier import BLOOM_LEVELS, _find_figshare_exam_dataset, _normalise_bloom
from bloom_models import (
    HierarchicalBloomClassifier,
    OrdinalThresholdClassifier,
    make_linear_svm_pipeline,
    make_logreg_pipeline,
)


SEED = 42
VERSION = "figshare_bloom_v1"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

LABEL_MAPPING = {
    "Knowledge": "Remember",
    "Comprehension": "Understand",
    "Application": "Apply",
    "Analysis": "Analyze",
    "Evaluation": "Evaluate",
    "Synthesis": "Create",
}


def _make_models() -> Dict[str, object]:
    return {
        "logreg_balanced": make_logreg_pipeline(class_weight="balanced"),
        "linear_svm_balanced": make_linear_svm_pipeline(class_weight="balanced"),
        "hierarchical_svm": HierarchicalBloomClassifier(class_weight="balanced"),
        "ordinal_threshold": OrdinalThresholdClassifier(class_weight="balanced"),
    }


def _clean_question(value: object) -> str:
    return " ".join(str(value).strip().split())


def _read_and_prepare() -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = _find_figshare_exam_dataset()
    if path is None:
        raise FileNotFoundError("Figshare dataset not found on disk.")

    raw = pd.read_csv(path, low_memory=False)
    qcol = "QUESTION"
    lcol = "BT LEVEL"
    if qcol not in raw.columns or lcol not in raw.columns:
        raise RuntimeError(f"Expected QUESTION and BT LEVEL columns in {path}")

    frame = pd.DataFrame(
        {
            "question": raw[qcol].map(_clean_question),
            "original_label": raw[lcol].astype(str).str.strip(),
        }
    )
    audit: Dict[str, object] = {
        "dataset_name": "figshare",
        "dataset_path": str(path),
        "dataset_version": VERSION,
        "label_mapping": LABEL_MAPPING,
        "raw_rows": int(len(frame)),
    }

    frame = frame.dropna(subset=["question", "original_label"]).copy()
    frame = frame[frame["question"].str.len() > 0].copy()
    audit["non_empty_rows"] = int(len(frame))
    audit["original_label_distribution"] = frame["original_label"].value_counts().to_dict()

    frame["bloom_level"] = frame["original_label"].map(_normalise_bloom)
    frame = frame[frame["bloom_level"].notna()].copy()
    frame["bloom_level"] = frame["bloom_level"].astype(str)
    audit["normalized_rows"] = int(len(frame))
    audit["normalized_label_distribution"] = frame["bloom_level"].value_counts().to_dict()

    exact_dedup = frame.drop_duplicates(subset=["question", "bloom_level"]).copy()
    audit["rows_after_exact_dedup"] = int(len(exact_dedup))

    conflicts = exact_dedup.groupby("question")["bloom_level"].nunique()
    conflicting_questions = set(conflicts[conflicts > 1].index.tolist())
    audit["conflicting_questions_removed"] = int(len(conflicting_questions))
    if conflicting_questions:
        exact_dedup = exact_dedup[~exact_dedup["question"].isin(conflicting_questions)].copy()

    final_df = exact_dedup.drop_duplicates(subset=["question"]).reset_index(drop=True)
    audit["final_rows"] = int(len(final_df))
    audit["final_unique_questions"] = int(final_df["question"].nunique())
    audit["final_label_distribution"] = final_df["bloom_level"].value_counts().to_dict()
    audit["conflict_resolution_strategy"] = (
        "Remove exact duplicates first, then remove any question text that maps to multiple Bloom labels, "
        "then keep one row per unique question."
    )
    return final_df, audit


def _split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["bloom_level"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["bloom_level"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _metric_bundle(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=BLOOM_LEVELS)
    y_true_idx = np.array([BLOOM_LEVELS.index(y) for y in y_true], dtype=np.int32)
    y_pred_idx = np.array([BLOOM_LEVELS.index(y) for y in y_pred], dtype=np.int32)
    ordinal_distance = np.abs(y_true_idx - y_pred_idx)
    mis = []
    for i, actual in enumerate(BLOOM_LEVELS):
        for j, predicted in enumerate(BLOOM_LEVELS):
            if i == j or cm[i, j] == 0:
                continue
            mis.append({"actual": actual, "predicted": predicted, "count": int(cm[i, j])})
    mis.sort(key=lambda item: -item["count"])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mean_ordinal_error": float(np.mean(ordinal_distance)),
        "within_one_level_accuracy": float(np.mean(ordinal_distance <= 1)),
        "severe_error_rate": float(np.mean(ordinal_distance >= 2)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=BLOOM_LEVELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "top_misclassifications": mis[:10],
    }


def _majority_baseline(y_train: List[str], y_eval: List[str]) -> Dict[str, object]:
    majority = max(set(y_train), key=y_train.count)
    pred = [majority] * len(y_eval)
    out = _metric_bundle(y_eval, pred)
    out["majority_class"] = majority
    return out


def _random_baseline(y_train: List[str], y_eval: List[str]) -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(np.array(y_train), return_counts=True)
    probs = counts / counts.sum()
    pred = rng.choice(classes, size=len(y_eval), p=probs).tolist()
    out = _metric_bundle(y_eval, pred)
    out["sampling_distribution"] = {str(k): float(v) for k, v in zip(classes, probs)}
    return out


def _cross_validate(models: Dict[str, object], x_dev: List[str], y_dev: List[str]) -> Dict[str, object]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    report: Dict[str, object] = {}
    for name, model in models.items():
        fold_scores: List[float] = []
        fold_distance: List[float] = []
        for train_idx, val_idx in skf.split(x_dev, y_dev):
            x_train = [x_dev[i] for i in train_idx]
            y_train = [y_dev[i] for i in train_idx]
            x_val = [x_dev[i] for i in val_idx]
            y_val = [y_dev[i] for i in val_idx]
            fitted = clone(model)
            fitted.fit(x_train, y_train)
            pred = fitted.predict(x_val)
            fold_scores.append(float(f1_score(y_val, pred, average="macro")))
            true_idx = np.array([BLOOM_LEVELS.index(y) for y in y_val], dtype=np.int32)
            pred_idx = np.array([BLOOM_LEVELS.index(y) for y in pred], dtype=np.int32)
            fold_distance.append(float(np.mean(np.abs(true_idx - pred_idx))))
        report[name] = {
            "macro_f1_mean": float(np.mean(fold_scores)),
            "macro_f1_std": float(np.std(fold_scores)),
            "mean_ordinal_error": float(np.mean(fold_distance)),
            "fold_macro_f1": fold_scores,
        }
    return report


def _rank_key(metrics: Dict[str, object]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["within_one_level_accuracy"]),
        -float(metrics["mean_ordinal_error"]),
    )


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / f"{VERSION}_train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_DIR / f"{VERSION}_val.csv", index=False, encoding="utf-8")
    test_df.to_csv(DATA_DIR / f"{VERSION}_test.csv", index=False, encoding="utf-8")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(DATA_DIR / f"{VERSION}.csv", index=False, encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    df, audit = _read_and_prepare()
    train_df, val_df, test_df = _split_dataset(df)
    _save_splits(train_df, val_df, test_df)

    split_report = {
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "train_distribution": train_df["bloom_level"].value_counts().to_dict(),
        "val_distribution": val_df["bloom_level"].value_counts().to_dict(),
        "test_distribution": test_df["bloom_level"].value_counts().to_dict(),
    }

    x_train = train_df["question"].tolist()
    y_train = train_df["bloom_level"].tolist()
    x_val = val_df["question"].tolist()
    y_val = val_df["bloom_level"].tolist()
    x_test = test_df["question"].tolist()
    y_test = test_df["bloom_level"].tolist()

    models = _make_models()
    cv_report = _cross_validate(models, x_train, y_train)
    validation_candidates: Dict[str, object] = {}
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_val)
        validation_candidates[name] = _metric_bundle(y_val, pred)
    best_name = max(validation_candidates.items(), key=lambda item: _rank_key(item[1]))[0]
    val_pred = clone(models[best_name]).fit(x_train, y_train).predict(x_val)

    x_dev = x_train + x_val
    y_dev = y_train + y_val
    best_model = clone(models[best_name])
    t0 = time.time()
    best_model.fit(x_dev, y_dev)
    fit_seconds = time.time() - t0

    test_pred = best_model.predict(x_test)

    baselines = {
        "majority_val": _majority_baseline(y_train, y_val),
        "majority_test": _majority_baseline(y_dev, y_test),
        "random_val": _random_baseline(y_train, y_val),
        "random_test": _random_baseline(y_dev, y_test),
    }

    results = {
        "dataset": "figshare",
        "dataset_version": VERSION,
        "seed": SEED,
        "preprocessing_audit": audit,
        "split_report": split_report,
        "evaluation_protocol": {
            "train_val_test": [70, 15, 15],
            "class_balancing": "model-level class_weight='balanced' for learned classifiers",
            "model_selection": "5-fold stratified cross-validation on the train split",
            "validation_usage": "single held-out validation split for final model selection before test",
            "final_metric_split": "held-out test set",
            "features": "TF-IDF word 1-2 grams + char 3-5 grams",
            "hierarchical_modeling": "candidate hierarchical lower-order vs higher-order pipeline included",
            "ordinal_evaluation": "mean ordinal error, within-one-level accuracy, severe error rate",
        },
        "baselines": baselines,
        "candidate_models": cv_report,
        "candidate_validation_metrics": validation_candidates,
        "selected_model": best_name,
        "fit_seconds_on_dev": round(fit_seconds, 3),
        "validation_metrics": _metric_bundle(y_val, val_pred),
        "test_metrics": _metric_bundle(y_test, test_pred),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{VERSION}_evaluation.json"
    model_path = MODELS_DIR / "figshare_bloom_tfidf.joblib"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(best_model, model_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
