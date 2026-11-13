"""VQA official punctuation/digit/article/contraction normalization.

Lookup tables are attributed to GT-Vision-Lab/VQA in vqa_tables.json.
Apply the same normalization unconditionally to every answer, as required here.
"""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
from .io import read_json


@lru_cache(maxsize=1)
def _tables():
    return read_json(Path(__file__).with_name("vqa_tables.json"))


@lru_cache(maxsize=100000)
def normalize_answer(text: str) -> str:
    tables = _tables()
    text = text.replace("\n", " ").replace("\t", " ").strip()
    original = text
    for punctuation in tables["punct"]:
        text = text.replace(punctuation, "" if (
            punctuation + " " in original or " " + punctuation in original
            or re.search(r"(\d)(,)(\d)", original)) else " ")
    # Preserve the official period expression, including decimal handling.
    text = re.sub(r"(?!<=\d)(\.)(?!\d)", "", text)
    words = [tables["manualMap"].get(word, word) for word in text.lower().split()]
    return " ".join(tables["contractions"].get(word, word) for word in words
                    if word not in tables["articles"])


def consensus(answers: list[dict]) -> dict:
    if len(answers) != 10 or any(not isinstance(a.get("answer"), str) for a in answers):
        raise ValueError("Natural VQA requires ten original annotator answers")
    normalized = [normalize_answer(a["answer"]) for a in answers]
    frequencies = Counter(normalized)
    winner, count = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[0]
    target = next(a["answer"] for a, norm in zip(answers, normalized) if norm == winner)
    return {"canonical_answer": winner, "agreement": count, "target": target,
            "normalized_answer_frequencies": dict(frequencies)}
