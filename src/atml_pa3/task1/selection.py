"""Deterministic first-eligible selection. No shuffling or truncation."""
from collections import Counter
import hashlib
import json

from .formatting import format_example
from .normalization import normalize_question


def canonical_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def select_examples(
    rows,
    tokenizer,
    leakage_set,
    count=40000,
    max_length=2048,
    max_scanned=None,
    on_accept=None,
    retain_examples=True,
):
    if count <= 0 or max_length <= 0:
        raise ValueError("count and max_length must be positive")

    accepted = []
    accepted_count = 0
    accepted_source_indices = []
    rejected = Counter()
    scanned = 0
    source_hash = hashlib.sha256()

    for source_index, row in enumerate(rows):
        if max_scanned is not None and scanned >= max_scanned:
            break

        scanned += 1
        source_hash.update(canonical_bytes(row) + b"\n")

        query = row["query"]
        response = row["response"]

        if not isinstance(query, str) or not isinstance(response, str):
            raise TypeError(
                f"Non-string query/response at source index {source_index}"
            )

        if not query.strip():
            rejected["empty_query"] += 1
            continue

        if not response.strip():
            rejected["empty_response"] += 1
            continue

        original_question = row.get("original_question") or ""

        if any(
            normalize_question(text) in leakage_set
            for text in (query, original_question)
        ):
            rejected["gsm8k_overlap"] += 1
            continue

        encoded = format_example(
            tokenizer,
            query,
            response,
        )

        if len(encoded["input_ids"]) > max_length:
            rejected["over_length"] += 1
            continue

        example = {
            "source_index": source_index,
            "source": row,
            **encoded,
        }

        accepted_count += 1
        accepted_source_indices.append(source_index)

        if on_accept is not None:
            on_accept(example)

        if retain_examples:
            accepted.append(example)

        if accepted_count == count:
            break

    if accepted_count != count:
        raise ValueError(
            f"Only {accepted_count}/{count} eligible after {scanned} rows; "
            f"rejections={dict(rejected)}"
        )

    stats = {
        "scanned": scanned,
        "accepted": accepted_count,
        "rejected": dict(rejected),
        "scanned_source_sha256": source_hash.hexdigest(),
        "accepted_source_indices": accepted_source_indices,
    }

    return accepted, stats
