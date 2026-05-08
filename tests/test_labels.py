"""Etiketlerin 0/1 (ham/spam) olması — hocanın ikili sınıflandırma beklentisi."""

import pandas as pd

import train as train_module


def test_ham_spam_to_01() -> None:
    s = pd.Series(["ham", "spam", "ham"])
    y = train_module._normalize_labels(s)
    assert list(y) == [0, 1, 0]


def test_numeric_labels_passthrough() -> None:
    s = pd.Series([0, 1, 0])
    y = train_module._normalize_labels(s)
    assert list(y.astype(int)) == [0, 1, 0]
