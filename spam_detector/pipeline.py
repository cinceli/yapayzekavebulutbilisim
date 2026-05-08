"""Kayıtlı model yükleme ve Ham / Spam tahmini."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import joblib

from .preprocessing import TextPreprocessor

_DEFAULT_LABELS = ("Ham", "Spam")


class SmsClassifierPipeline:
    """Ön işleme + TF-IDF (+ isteğe bağlı SVD) + sınıflandırıcı."""

    __slots__ = ("preprocessor", "vectorizer", "svd", "classifier", "label_names", "model_type")

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        vectorizer: Any,
        classifier: Any,
        label_names: Tuple[str, str] = _DEFAULT_LABELS,
        svd: Any | None = None,
        model_type: str = "Classifier",
    ) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.svd = svd
        self.classifier = classifier
        self.label_names = label_names
        self.model_type = model_type

    def _transform(self, cleaned: str) -> Any:
        X = self.vectorizer.transform([cleaned])
        if self.svd is not None:
            X = self.svd.transform(X)
        return X

    def predict(self, raw_text: str) -> Tuple[str, float]:
        """(etiket, spam_olasılığı) döner."""
        cleaned = self.preprocessor.clean(raw_text)
        if not cleaned.strip():
            return "Belirsiz (metin boş veya çok kısa)", 0.0
        X = self._transform(cleaned)
        proba = self.classifier.predict_proba(X)[0]
        classes = list(self.classifier.classes_)
        if 1 in classes:
            idx_spam = classes.index(1)
        elif "spam" in classes:
            idx_spam = classes.index("spam")
        elif "fake" in classes:
            idx_spam = classes.index("fake")
        else:
            idx_spam = 1 if len(classes) > 1 else 0
        spam_p = float(proba[idx_spam])
        pred = self.classifier.predict(X)[0]
        is_spam = pred == 1 or str(pred).lower() in ("spam", "fake")
        label = self.label_names[1] if is_spam else self.label_names[0]
        return label, spam_p

    @classmethod
    def load(
        cls,
        models_dir: str | Path,
        artifact_file: str = "pipeline_artifacts.joblib",
    ) -> "SmsClassifierPipeline":
        base = Path(models_dir)
        path = base / artifact_file if not Path(artifact_file).is_absolute() else Path(artifact_file)
        data = joblib.load(path)
        return cls(
            preprocessor=data["preprocessor"],
            vectorizer=data["vectorizer"],
            classifier=data["classifier"],
            label_names=tuple(data.get("label_names", _DEFAULT_LABELS)),
            svd=data.get("svd"),
            model_type=data.get("model_type", "Classifier"),
        )


NewsClassifierPipeline = SmsClassifierPipeline
