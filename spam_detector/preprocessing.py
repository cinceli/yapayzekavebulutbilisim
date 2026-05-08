"""SMS / kısa metin ön işleme (UCI İngilizce SMS ile uyumlu). O(n) tek geçişte temizleme."""

from __future__ import annotations

import re
import string
from typing import Iterable, Set

_DEFAULT_STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now",
}


class TextPreprocessor:
    """SMS metnini küçük harfe çevirir, noktalama ve stopword temizliği yapar."""

    __slots__ = ("stopwords", "_punct_trans")

    def __init__(self, stopwords: Iterable[str] | None = None) -> None:
        self.stopwords: Set[str] = set(stopwords) if stopwords else set(_DEFAULT_STOPWORDS)
        self._punct_trans = str.maketrans("", "", string.punctuation)

    def clean(self, text: str) -> str:
        """Tek string için ön işleme; boş veya çok kısa metinlerde boş string döner."""
        if not text or not isinstance(text, str):
            return ""
        lowered = text.lower()
        lowered = re.sub(r"http\S+|www\.\S+", " ", lowered)
        lowered = re.sub(r"\S+@\S+", " ", lowered)
        lowered = lowered.translate(self._punct_trans)
        lowered = re.sub(r"\d+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        tokens = [t for t in lowered.split() if t and t not in self.stopwords]
        return " ".join(tokens)

    def transform_series(self, texts: "pandas.Series") -> "pandas.Series":
        """DataFrame sütunu için uygulama (pandas gerekir)."""
        import pandas as pd

        if not isinstance(texts, pd.Series):
            raise TypeError("texts bir pandas.Series olmalıdır")
        return texts.astype(str).map(self.clean)


def preprocess_text(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Modül düzeyinde kısayol."""
    return TextPreprocessor(stopwords=stopwords).clean(text)
