"""Uyumluluk: eski kayıtlar src.model_pipeline bekleyebilir."""

from spam_detector.pipeline import NewsClassifierPipeline, SmsClassifierPipeline

__all__ = ["NewsClassifierPipeline", "SmsClassifierPipeline"]
