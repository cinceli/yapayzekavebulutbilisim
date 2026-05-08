"""Eğitim: TF-IDF + Logistic Regression veya TF-IDF + SVD + Random Forest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .preprocessing import TextPreprocessor

DEFAULT_LABEL_NAMES: Tuple[str, str] = ("Ham", "Spam")


@dataclass
class TrainResult:
    """Eğitim çıktısı: tahmin için nesneler ve test metrikleri."""

    preprocessor: TextPreprocessor
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    classifier: Any
    label_names: Tuple[str, str]
    model_type: str
    metrics: Dict[str, float]
    y_test: np.ndarray
    pred: np.ndarray


def _min_df_for_corpus(n: int) -> int:
    return 1 if n < 500 else 2


def _svd_components(n_samples: int, n_features: int) -> int:
    upper = min(n_samples, n_features)
    return max(1, min(200, upper - 1))


def build_tfidf(n_rows: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=_min_df_for_corpus(n_rows),
        sublinear_tf=True,
    )


def train_logistic_regression(
    pre: TextPreprocessor,
    X_clean: pd.Series,
    y: pd.Series,
    label_names: Tuple[str, str] = DEFAULT_LABEL_NAMES,
    random_state: int = 42,
) -> TrainResult:
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=random_state, stratify=y
    )
    vectorizer = build_tfidf(len(X_clean))
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)

    metrics = _compute_metrics(y_test.to_numpy(), pred)
    return TrainResult(
        preprocessor=pre,
        vectorizer=vectorizer,
        svd=None,
        classifier=clf,
        label_names=label_names,
        model_type="Logistic Regression + TF-IDF",
        metrics=metrics,
        y_test=y_test.to_numpy(),
        pred=pred,
    )


def train_random_forest_svd(
    pre: TextPreprocessor,
    X_clean: pd.Series,
    y: pd.Series,
    label_names: Tuple[str, str] = DEFAULT_LABEL_NAMES,
    random_state: int = 42,
) -> TrainResult:
    """Random Forest için önce TruncatedSVD ile yoğunlaştırma."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=random_state, stratify=y
    )
    vectorizer = build_tfidf(len(X_clean))
    X_tr_sp = vectorizer.fit_transform(X_train)
    X_te_sp = vectorizer.transform(X_test)

    n_comp = _svd_components(X_tr_sp.shape[0], X_tr_sp.shape[1])
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_tr = svd.fit_transform(X_tr_sp)
    X_te = svd.transform(X_te_sp)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)

    metrics = _compute_metrics(y_test.to_numpy(), pred)
    return TrainResult(
        preprocessor=pre,
        vectorizer=vectorizer,
        svd=svd,
        classifier=clf,
        label_names=label_names,
        model_type="Random Forest + TF-IDF + TruncatedSVD",
        metrics=metrics,
        y_test=y_test.to_numpy(),
        pred=pred,
    )


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_result_to_joblib_dict(result: TrainResult) -> Dict[str, Any]:
    return {
        "preprocessor": result.preprocessor,
        "vectorizer": result.vectorizer,
        "svd": result.svd,
        "classifier": result.classifier,
        "label_names": result.label_names,
        "model_type": result.model_type,
    }


def print_report(result: TrainResult) -> None:
    m = result.metrics
    n0, n1 = result.label_names[0], result.label_names[1]
    print(f"--- {result.model_type} ---")
    print(f"Accuracy:  {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1-score:  {m['f1']:.4f}")
    print(
        "\n",
        classification_report(result.y_test, result.pred, target_names=[f"{n0}(0)", f"{n1}(1)"]),
    )
