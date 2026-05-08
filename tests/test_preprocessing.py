"""Ön işleme birim testleri."""

from spam_detector.preprocessing import TextPreprocessor


def test_clean_lowercase_and_strip() -> None:
    pre = TextPreprocessor()
    assert "hello" in pre.clean("HELLO world!!!")


def test_clean_empty_returns_empty() -> None:
    pre = TextPreprocessor()
    assert pre.clean("") == ""
    assert pre.clean("   ") == ""


def test_clean_removes_url() -> None:
    pre = TextPreprocessor()
    out = pre.clean("see http://evil.com for prize")
    assert "http" not in out
