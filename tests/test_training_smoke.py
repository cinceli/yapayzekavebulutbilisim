"""Eğitim işlem akışı duman testi (küçük veri, metrikler üretilir)."""

import pandas as pd

from spam_detector.preprocessing import TextPreprocessor
from spam_detector.training import train_logistic_regression


def test_logreg_train_produces_metrics() -> None:
    pre = TextPreprocessor()
    X = pd.Series(["hello friend", "win prize now"] * 15)
    y = pd.Series([0, 1] * 15)
    Xc = X.map(pre.clean)
    res = train_logistic_regression(pre, Xc, y)
    assert "accuracy" in res.metrics
    assert "precision" in res.metrics
    assert "recall" in res.metrics
    assert "f1" in res.metrics
    assert 0.0 <= res.metrics["f1"] <= 1.0
