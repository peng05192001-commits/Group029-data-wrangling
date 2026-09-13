"""Published FIT5196 A1 text-function interface for Group029.

The functions are deliberately pure: they perform no file I/O, network access,
or row-specific lookup. JSON and XML structure must be parsed before values are
passed into this module.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Final


LITERAL_NAN: Final = "NaN"

ORDER_REFERENCE_RE: Final = re.compile(
    r"(?<!\w)([HC]ORD[0-9]{6})(?!\w)",
    re.IGNORECASE,
)
PRODUCT_SKU_RE: Final = re.compile(
    r"(?<![\w-])(SKU-[A-Za-z0-9]+)(?![\w-])",
    re.IGNORECASE,
)
PROMO_CODE_RE: Final = re.compile(
    r"(?<![\w-])(B[1-5]SAVE-[0-9]{2})(?![\w-])",
    re.IGNORECASE,
)

HTML_TAG_RE: Final = re.compile(r"<[^>]*>")
FIXED_MARKER_RE: Final = re.compile(
    r"\[(?:SYSTEM|CATALOGUE|VERIFIED_PURCHASE)\]",
    re.IGNORECASE,
)
SOURCE_MARKER_RE: Final = re.compile(r"\[SOURCE\s*:[^\]]*\]", re.IGNORECASE)
RATING_MARKER_RE: Final = re.compile(
    r"\[RATING\s*:\s*[1-5]\s*/\s*5\]",
    re.IGNORECASE,
)
SOCIAL_MARKER_RE: Final = re.compile(
    r"(?<![A-Za-z0-9_-])(?:#verified-buyer|@store_support)(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
URL_RE: Final = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
REFERENCE_WRAPPER_RE: Final = re.compile(
    r"(?<!\w)Reference\s*:\s*[HC]ORD[0-9]{6}(?!\w)"
    r"(?:\s*[-–—;,|/]\s*|\s+)"
    r"SKU\s*:\s*SKU-[A-Za-z0-9]+(?![\w-])",
    re.IGNORECASE,
)
PROMO_WRAPPER_RE: Final = re.compile(
    r"(?<!\w)PROMO\s*:\s*B[1-5]SAVE-[0-9]{2}(?![\w-])",
    re.IGNORECASE,
)
WHITESPACE_RE: Final = re.compile(r"\s+")


def _as_text(value: object) -> str:
    """Return an accepted scalar as text while treating None as empty."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError("Text functions accept only None or a string.")
    return value


def _is_emoji_character(character: str) -> bool:
    """Recognise common Unicode emoji blocks without deleting all symbols."""
    codepoint = ord(character)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or 0x2300 <= codepoint <= 0x23FF
        or 0x2B00 <= codepoint <= 0x2BFF
        or 0x1F1E6 <= codepoint <= 0x1F1FF
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or codepoint in {0x200D, 0xFE0E, 0xFE0F, 0x20E3}
    )


def _remove_emoji(text: str) -> str:
    return "".join(character for character in text if not _is_emoji_character(character))


def clean_narrative_text(value):
    """Accept None or a string; return cleaned text or the string 'NaN'."""
    text = _as_text(value)
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = HTML_TAG_RE.sub(" ", text)
    text = FIXED_MARKER_RE.sub(" ", text)
    text = SOURCE_MARKER_RE.sub(" ", text)
    text = RATING_MARKER_RE.sub(" ", text)
    text = SOCIAL_MARKER_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = _remove_emoji(text)
    text = REFERENCE_WRAPPER_RE.sub(" ", text)
    text = PROMO_WRAPPER_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text if text else LITERAL_NAN


def extract_order_reference(value):
    """Accept None or a string; return the upper-case reference or 'NaN'."""
    match = ORDER_REFERENCE_RE.search(_as_text(value))
    return match.group(1).upper() if match else LITERAL_NAN


def extract_product_sku(value):
    """Accept None or a string; return the upper-case SKU or 'NaN'."""
    match = PRODUCT_SKU_RE.search(_as_text(value))
    return match.group(1).upper() if match else LITERAL_NAN


def extract_promo_code(value):
    """Accept None or a string; return the upper-case code or 'NaN'."""
    match = PROMO_CODE_RE.search(_as_text(value))
    return match.group(1).upper() if match else LITERAL_NAN


def _is_latin_letter(character: str) -> bool:
    return (
        unicodedata.category(character).startswith("L")
        and "LATIN" in unicodedata.name(character, "")
    )


def build_latin_analysis(value):
    """Accept cleaned multilingual text; return Latin analysis or 'NaN'."""
    text = unicodedata.normalize("NFC", _as_text(value))
    if not text or text == LITERAL_NAN:
        return LITERAL_NAN

    output: list[str] = []
    has_latin_letter = False
    keep_combining_mark = False
    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L"):
            if _is_latin_letter(character):
                output.append(character)
                has_latin_letter = True
                keep_combining_mark = True
            else:
                output.append(" ")
                keep_combining_mark = False
        elif category.startswith("M"):
            if keep_combining_mark:
                output.append(character)
        elif category.startswith("N") or category.startswith("P") or character.isspace():
            output.append(character)
            keep_combining_mark = False
        else:
            output.append(" ")
            keep_combining_mark = False

    cleaned = WHITESPACE_RE.sub(" ", "".join(output)).strip()
    return cleaned if has_latin_letter and cleaned else LITERAL_NAN


def contains_non_latin_script(value):
    """Accept cleaned multilingual text; return a Python bool."""
    text = unicodedata.normalize("NFC", _as_text(value))
    if not text or text == LITERAL_NAN:
        return False
    return any(
        unicodedata.category(character).startswith("L")
        and not _is_latin_letter(character)
        for character in text
    )


__all__ = [
    "clean_narrative_text",
    "extract_order_reference",
    "extract_product_sku",
    "extract_promo_code",
    "build_latin_analysis",
    "contains_non_latin_script",
]
