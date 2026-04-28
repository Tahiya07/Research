from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


BLOOM_LEVELS: List[str] = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

LOWER_ORDER = ["Remember", "Understand", "Apply"]
HIGHER_ORDER = ["Analyze", "Evaluate", "Create"]


def make_feature_union() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
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
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_linear_svm_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_union()),
            ("clf", LinearSVC(class_weight=class_weight)),
        ]
    )


def make_logreg_pipeline(class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        [
            ("features", make_feature_union()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )


class HierarchicalBloomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weight: str | None = "balanced") -> None:
        self.class_weight = class_weight
        self.root_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.lower_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.higher_ = make_linear_svm_pipeline(class_weight=class_weight)
        self.classes_ = np.array(BLOOM_LEVELS)

    def fit(self, X: List[str], y: List[str]):
        X_list = list(X)
        y_list = list(y)
        root_y = ["lower" if label in LOWER_ORDER else "higher" for label in y_list]
        self.root_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.root_.fit(X_list, root_y)

        lower_X = [x for x, label in zip(X_list, y_list) if label in LOWER_ORDER]
        lower_y = [label for label in y_list if label in LOWER_ORDER]
        higher_X = [x for x, label in zip(X_list, y_list) if label in HIGHER_ORDER]
        higher_y = [label for label in y_list if label in HIGHER_ORDER]

        self.lower_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.higher_ = make_linear_svm_pipeline(class_weight=self.class_weight)
        self.lower_.fit(lower_X, lower_y)
        self.higher_.fit(higher_X, higher_y)
        return self

    def predict(self, X: List[str]) -> np.ndarray:
        out = []
        root_pred = self.root_.predict(X)
        for text, branch in zip(X, root_pred):
            if branch == "lower":
                out.append(str(self.lower_.predict([text])[0]))
            else:
                out.append(str(self.higher_.predict([text])[0]))
        return np.array(out)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        root_scores = self.root_.decision_function(X)
        root_scores = np.asarray(root_scores, dtype=np.float64).reshape(-1)
        p_higher = 1.0 / (1.0 + np.exp(-root_scores))
        p_lower = 1.0 - p_higher

        lower_scores = np.asarray(self.lower_.decision_function(X), dtype=np.float64)
        higher_scores = np.asarray(self.higher_.decision_function(X), dtype=np.float64)
        if lower_scores.ndim == 1:
            lower_scores = lower_scores[:, None]
        if higher_scores.ndim == 1:
            higher_scores = higher_scores[:, None]

        lower_probs = _softmax_rows(lower_scores)
        higher_probs = _softmax_rows(higher_scores)

        out = np.zeros((len(X), len(BLOOM_LEVELS)), dtype=np.float64)
        for i in range(len(X)):
            out[i, 0:3] = p_lower[i] * lower_probs[i]
            out[i, 3:6] = p_higher[i] * higher_probs[i]
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
        return out


class OrdinalThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, class_weight: str | None = "balanced") -> None:
        self.class_weight = class_weight
        self.vectorizer_ = make_feature_union()
        self.threshold_models_: List[LogisticRegression] = []
        self.classes_ = np.array(BLOOM_LEVELS)

    def fit(self, X: List[str], y: List[str]):
        X_list = list(X)
        y_idx = np.array([BLOOM_LEVELS.index(label) for label in y], dtype=np.int32)
        self.vectorizer_ = make_feature_union()
        X_vec = self.vectorizer_.fit_transform(X_list)
        self.threshold_models_ = []
        for threshold in range(len(BLOOM_LEVELS) - 1):
            binary = (y_idx > threshold).astype(np.int32)
            clf = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight=self.class_weight,
            )
            clf.fit(X_vec, binary)
            self.threshold_models_.append(clf)
        return self

    def predict_proba(self, X: List[str]) -> np.ndarray:
        X_vec = self.vectorizer_.transform(list(X))
        threshold_probs = []
        for clf in self.threshold_models_:
            p = clf.predict_proba(X_vec)[:, 1]
            threshold_probs.append(p)
        cum = np.vstack(threshold_probs).T
        # Enforce monotonic decreasing P(y > k)
        cum = np.minimum.accumulate(cum, axis=1)

        out = np.zeros((len(X), len(BLOOM_LEVELS)), dtype=np.float64)
        out[:, 0] = 1.0 - cum[:, 0]
        for idx in range(1, len(BLOOM_LEVELS) - 1):
            out[:, idx] = np.clip(cum[:, idx - 1] - cum[:, idx], 0.0, 1.0)
        out[:, -1] = np.clip(cum[:, -1], 0.0, 1.0)
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
        return out

    def predict(self, X: List[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return np.array([BLOOM_LEVELS[i] for i in idx])


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.clip(exp_scores.sum(axis=1, keepdims=True), 1e-9, None)
