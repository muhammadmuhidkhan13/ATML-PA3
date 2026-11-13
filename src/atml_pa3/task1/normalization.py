"""Comparison-only normalization, approved by the user on 2026-11-01."""
import unicodedata

NORMALIZATION_VERSION = "nfkc-casefold-whitespace-v1"


def normalize_question(text: str) -> str:
    """Preserve punctuation; do not replace original training text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_leakage_set(rows):
    return {normalize_question(row["question"]) for row in rows}
