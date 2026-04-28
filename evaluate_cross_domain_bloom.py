from __future__ import annotations

import json
import os
import random
import re
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from bloom_models import make_linear_svm_pipeline, make_logreg_pipeline
from classifier import _normalise_bloom


SEED = 42
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

FIGSHARE_PATH = DATA_DIR / "figshare_bloom_v1.csv"
DEFAULT_MOOCRADAR_PATHS = [
    "./data/problem.json",
    "./data/moocradar_problem.json",
    "./data/moocradar/problem.json",
]

LOWER = {"Remember", "Understand", "Apply"}
HIGHER = {"Analyze", "Evaluate", "Create"}
MID3 = {"Apply", "Analyze"}
HIGH3 = {"Evaluate", "Create"}


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _find_moocradar_problem_file() -> Optional[Path]:
    env = os.environ.get("MOOCRADAR_PROBLEM_PATH")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for raw in DEFAULT_MOOCRADAR_PATHS:
        p = Path(raw).expanduser()
        if p.is_file():
            return p
    return None


def _read_json_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    rows: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _lookup_case_insensitive(record: dict, candidates: Sequence[str]) -> Optional[object]:
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _flatten_iterable_text(value: object) -> str:
    if isinstance(value, str):
        return _clean_text(value)
    if isinstance(value, list):
        parts = [_flatten_iterable_text(x) for x in value]
        return _clean_text(" ".join(p for p in parts if p))
    if isinstance(value, dict):
        parts = [_flatten_iterable_text(v) for v in value.values()]
        return _clean_text(" ".join(p for p in parts if p))
    return _clean_text(value)


def _extract_question_text(record: dict) -> str:
    direct = _lookup_case_insensitive(
        record,
        [
            "question",
            "question_text",
            "problem_text",
            "content",
            "detail",
            "body",
            "stem",
            "title",
            "text",
            "exercise",
            "exercise_text",
            "desc",
            "description",
        ],
    )
    text = ""
    if isinstance(direct, str) and _looks_like_serialized_detail(direct):
        text = _extract_from_serialized_detail(direct)
    elif direct is not None:
        text = _flatten_iterable_text(direct)
    if len(text) >= 10:
        return text

    # Fall back to a best-effort scan over string-bearing fields.
    best = ""
    for key, value in record.items():
        key_l = str(key).strip().lower()
        if key_l in {"problem_id", "exercise_id", "course_id", "knowledge_type"}:
            continue
        if isinstance(value, str) and _looks_like_serialized_detail(value):
            candidate = _extract_from_serialized_detail(value)
        else:
            candidate = _flatten_iterable_text(value)
        if any(tok in key_l for tok in ["question", "problem", "content", "text", "body", "stem", "desc", "detail"]):
            if len(candidate) > len(best):
                best = candidate
    return best


def _looks_like_serialized_detail(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") and "problem_id" in t and "exercise_id" in t


def _extract_from_serialized_detail(text: str) -> str:
    try:
        obj = ast.literal_eval(text)
    except Exception:
        return text
    if not isinstance(obj, dict):
        return text
    parts: List[str] = []
    for key in ("title", "content", "typetext"):
        value = obj.get(key)
        if value:
            parts.append(_clean_text(value))
    option = obj.get("option")
    if isinstance(option, dict):
        opt_text = " ".join(_clean_text(v) for v in option.values() if _clean_text(v))
        if opt_text:
            parts.append(opt_text)
    if not parts:
        return ""
    return _clean_text(" ".join(parts))


def _normalise_mooc_label(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        idx = int(value)
        if 1 <= idx <= 6:
            return ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][idx - 1]
    raw = _clean_text(value)
    if not raw:
        return None

    mapped = _normalise_bloom(raw)
    if mapped:
        return mapped

    lower = raw.lower()
    m = re.search(r"\bc\s*([1-6])\b", lower)
    if m:
        idx = int(m.group(1))
        return ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][idx - 1]

    digits = re.search(r"\b([1-6])\b", lower)
    if digits and any(tok in lower for tok in ["bloom", "cognitive", "level", "taxonomy"]):
        idx = int(digits.group(1))
        return ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"][idx - 1]

    return None


def _collapse_label(label: str, scheme: str) -> str:
    if scheme == "binary":
        return "Lower" if label in LOWER else "Higher"
    if scheme == "ternary":
        if label in {"Remember", "Understand"}:
            return "Low"
        if label in MID3:
            return "Mid"
        if label in HIGH3:
            return "High"
    raise ValueError(f"unknown scheme: {scheme}")


def _metric_bundle(y_true: List[str], y_pred: List[str], labels: Sequence[str]) -> Dict[str, object]:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    order = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = np.array([order[y] for y in y_true], dtype=np.int32)
    y_pred_idx = np.array([order[y] for y in y_pred], dtype=np.int32)
    distances = np.abs(y_true_idx - y_pred_idx)
    mis = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            mis.append({"actual": actual, "predicted": predicted, "count": int(cm[i, j])})
    mis.sort(key=lambda item: -item["count"])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mean_ordinal_error": float(np.mean(distances)),
        "within_one_level_accuracy": float(np.mean(distances <= 1)),
        "severe_error_rate": float(np.mean(distances >= 2)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(labels),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "top_misclassifications": mis[:10],
    }


def _read_figshare() -> pd.DataFrame:
    if not FIGSHARE_PATH.is_file():
        raise FileNotFoundError(f"Missing Figshare versioned dataset at {FIGSHARE_PATH}")
    df = pd.read_csv(FIGSHARE_PATH, low_memory=False)
    expected = {"question", "bloom_level"}
    if not expected.issubset(set(df.columns)):
        raise RuntimeError(f"Expected columns {expected} in {FIGSHARE_PATH}, got {list(df.columns)}")
    out = df[["question", "bloom_level"]].copy()
    out["question"] = out["question"].map(_clean_text)
    out["bloom_level"] = out["bloom_level"].map(_normalise_bloom)
    out = out.dropna(subset=["question", "bloom_level"])
    out = out[out["question"].str.len() > 0]
    out = out.drop_duplicates(subset=["question", "bloom_level"]).drop_duplicates(subset=["question"]).reset_index(drop=True)
    out["source"] = "figshare"
    return out


def _read_moocradar() -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = _find_moocradar_problem_file()
    if path is None:
        raise FileNotFoundError(
            "MoocRadar problem metadata not found. Place a compatible file at one of: "
            + ", ".join(DEFAULT_MOOCRADAR_PATHS)
            + " or set MOOCRADAR_PROBLEM_PATH."
        )
    records = _read_json_records(path)
    rows = []
    for rec in records:
        question = _extract_question_text(rec)
        label_raw = _lookup_case_insensitive(
            rec,
            [
                "cognitive_dimension",
                "cognitive_level",
                "cognitive",
                "bloom_level",
                "bloom",
                "taxonomy",
                "level",
            ],
        )
        label = _normalise_mooc_label(label_raw)
        if not question or not label:
            continue
        rows.append(
            {
                "problem_id": _lookup_case_insensitive(rec, ["problem_id", "id"]),
                "question": question,
                "bloom_level": label,
            }
        )
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError(f"Parsed 0 usable rows from MoocRadar problem file {path}")

    before = len(df)
    df = df.drop_duplicates(subset=["question", "bloom_level"]).copy()
    conflicts = df.groupby("question")["bloom_level"].nunique()
    conflicting_questions = set(conflicts[conflicts > 1].index.tolist())
    if conflicting_questions:
        df = df[~df["question"].isin(conflicting_questions)].copy()
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    df["source"] = "moocradar"
    audit = {
        "dataset_name": "moocradar",
        "dataset_path": str(path),
        "raw_records": int(len(records)),
        "usable_rows_before_dedup": int(before),
        "questions_removed_for_conflicts": int(len(conflicting_questions)),
        "final_rows": int(len(df)),
        "final_label_distribution": df["bloom_level"].value_counts().to_dict(),
    }
    return df, audit


def _make_reduced(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["bloom_level"].map(lambda x: _collapse_label(str(x), scheme))
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out


def _fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: Sequence[str]) -> Dict[str, object]:
    x_train = train_df["question"].tolist()
    y_train = train_df["label"].tolist()
    x_test = test_df["question"].tolist()
    y_test = test_df["label"].tolist()

    models = {
        "logreg_balanced": make_logreg_pipeline(class_weight="balanced"),
        "linear_svm_balanced": make_linear_svm_pipeline(class_weight="balanced"),
    }
    scored = {}
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_test)
        scored[name] = _metric_bundle(y_test, pred, labels)
    best_name = max(scored.items(), key=lambda kv: (kv[1]["macro_f1"], kv[1]["within_one_level_accuracy"], -kv[1]["mean_ordinal_error"]))[0]
    best_model = clone(models[best_name]).fit(x_train, y_train)
    best_pred = best_model.predict(x_test)
    return {
        "candidate_metrics": scored,
        "selected_model": best_name,
        "selected_metrics": _metric_bundle(y_test, best_pred, labels),
    }


def _within_dataset_eval(df: pd.DataFrame, labels: Sequence[str]) -> Dict[str, object]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )
    out = _fit_predict(train_df.reset_index(drop=True), test_df.reset_index(drop=True), labels)
    out["split"] = {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
    }
    return out


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    fig = _read_figshare()
    mooc, mooc_audit = _read_moocradar()

    report = {
        "seed": SEED,
        "primary_dataset": {
            "name": "figshare",
            "path": str(FIGSHARE_PATH),
            "rows": int(len(fig)),
            "label_distribution": fig["bloom_level"].value_counts().to_dict(),
        },
        "external_dataset": mooc_audit,
        "schemes": {},
    }

    for scheme, labels in [("binary", ["Lower", "Higher"]), ("ternary", ["Low", "Mid", "High"])]:
        fig_r = _make_reduced(fig, scheme)
        mooc_r = _make_reduced(mooc, scheme)
        report["schemes"][scheme] = {
            "labels": list(labels),
            "figshare_distribution": fig_r["label"].value_counts().to_dict(),
            "moocradar_distribution": mooc_r["label"].value_counts().to_dict(),
            "figshare_to_moocradar": _fit_predict(fig_r, mooc_r, labels),
            "moocradar_to_figshare": _fit_predict(mooc_r, fig_r, labels),
            "within_dataset_figshare": _within_dataset_eval(fig_r, labels),
            "within_dataset_moocradar": _within_dataset_eval(mooc_r, labels),
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cross_dataset_bloom_transfer.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
