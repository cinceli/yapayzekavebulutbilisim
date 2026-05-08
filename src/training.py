"""Uyumluluk: nadir pickle referansları için."""

from spam_detector.training import (
    DEFAULT_LABEL_NAMES,
    TrainResult,
    build_tfidf,
    print_report,
    train_logistic_regression,
    train_random_forest_svd,
    train_result_to_joblib_dict,
)

__all__ = [
    "DEFAULT_LABEL_NAMES",
    "TrainResult",
    "build_tfidf",
    "print_report",
    "train_logistic_regression",
    "train_random_forest_svd",
    "train_result_to_joblib_dict",
]
