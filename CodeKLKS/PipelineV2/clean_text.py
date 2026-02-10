# -*- coding: utf-8 -*-
"""
Unified text cleaning module.

Combines best practices from all existing scripts:
  - Unicode NFC normalization       (from fake_news_detection.py)
  - Emoji → text conversion         (from CrossAttentionFusion)
  - URL / email / phone tokenisation (from fake_news_detection.py)
  - HTML tag removal                 (from V1 / Gating)
"""

import re
import unicodedata
from typing import Optional

import pandas as pd

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False


# ── regex patterns (compiled once) ──────────────────────────────

_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_RE_WWW = re.compile(
    r"www\.(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_RE_EMAIL = re.compile(r"\S+@\S+")
_RE_PHONE = re.compile(r"(\+84|0)[0-9]{1,3}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}")
_RE_MULTI_SPACE = re.compile(r"\s+")


# ── public API ──────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """NFC normalization — chuẩn hóa ký tự tiếng Việt."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def emoji_to_text(text: str) -> str:
    """Convert emojis to readable text descriptions (e.g. 😀 → 'grinning face')."""
    if not text or not HAS_EMOJI:
        return text
    t = emoji.demojize(text, delimiters=(" ", " "))
    t = t.replace("_", " ")
    t = t.replace("\uFE0F", " ")
    return _RE_MULTI_SPACE.sub(" ", t).strip()


def clean_text(text: Optional[str]) -> str:
    """
    Full cleaning pipeline:
      1. NaN / type check
      2. Unicode NFC
      3. Emoji → text
      4. Remove HTML tags
      5. URL  → [URL]
      6. Email → [EMAIL]
      7. Phone → [PHONE]
      8. Normalize whitespace
    """
    if pd.isna(text):
        return ""
    text = str(text)

    # 1. Unicode normalization
    text = normalize_unicode(text)

    # 2. Emoji → descriptive text
    text = emoji_to_text(text)

    # 3. Remove HTML tags
    text = _RE_HTML.sub(" ", text)

    # 4. Tokenize URLs
    text = _RE_URL.sub(" [URL] ", text)
    text = _RE_WWW.sub(" [URL] ", text)

    # 5. Tokenize emails
    text = _RE_EMAIL.sub(" [EMAIL] ", text)

    # 6. Tokenize phone numbers
    text = _RE_PHONE.sub(" [PHONE] ", text)

    # 7. Collapse whitespace
    text = _RE_MULTI_SPACE.sub(" ", text).strip()

    return text


def is_valid(text: str, min_words: int = 8, min_chars: int = 10) -> bool:
    """Check if cleaned text is long enough to be useful for classification."""
    if not isinstance(text, str):
        return False
    if len(text) < min_chars:
        return False
    if len(text.split()) < min_words:
        return False
    return True
