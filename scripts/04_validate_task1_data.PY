"""Read-only independent validation of the completed Task 1 dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

EOT = "<|im_end|>"
IGNORE = -100

MODEL_SHA = "c1899de289a04d12100db370d81485cdf75e47ca"
META_SHA = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
GSM_SHA = "740312add88f781978c0658806c59bc2815b9866"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text):
    import unicodedata
    return " ".join(
        unicodedata.normalize("NFKC", text).casefold().split()
    )


def decode(tokenizer, ids):
    return tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def independent_tokens(tokenizer, query, response):
    """Reconstruct boundaries without importing the training formatter."""
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    begin = rendered.rfind(response + EOT)
    require(begin >= 0, "Chat template did not preserve response")
    end = begin + len(response)

    eot_id = tokenizer.convert_tokens_to_ids(EOT)

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    positions = [
        i for i, (a, b) in enumerate(offsets)
        if b > begin and a < end
    ]

    exact = bool(positions) and (
        offsets[positions[0]][0] == begin
        and offsets[positions[-1]][1] == end
        and positions == list(
            range(positions[0], positions[-1] + 1)
        )
    )

    if exact:
        stop = positions[-1] + 1
        exact = (
            stop < len(ids)
            and ids[stop] == eot_id
            and tuple(offsets[stop]) == (end, end + len(EOT))
            and decode(
                tokenizer,
                [ids[i] for i in positions] + [ids[stop]]
            ) == response + EOT
        )

    if exact:
        labels = [IGNORE] * len(ids)
        for i in positions + [stop]:
            labels[i] = ids[i]
    else:
        prefix = tokenizer(
            rendered[:begin], add_special_tokens=False
        )["input_ids"]
        answer = tokenizer(
            response, add_special_tokens=False
        )["input_ids"]
        suffix = tokenizer(
            rendered[end + len(EOT):],
            add_special_tokens=False,
        )["input_ids"]

        require(bool(answer), "Response produced no token IDs")

        ids = prefix + answer + [eot_id] + suffix
        labels = (
            [IGNORE] * len(prefix)
            + answer
            + [eot_id]
            + [IGNORE] * len(suffix)
        )

    require(
        decode(tokenizer, ids) == rendered,
        "Full decoded chat changed",
    )
    require(
        decode(
            tokenizer,
            [v for v in labels if v != IGNORE]
        ) == response + EOT,
        "Supervision is not response plus EOT",
    )

    return ids, labels, not exact


def validate(folder, tokenizer, gsm_rows, metamath_rows):
    expected_count = 40000

    meta_path = folder / "metadata.json"
    data_path = folder / "examples.jsonl"
    sums_path = folder / "SHA256SUMS"

    require(
        all(p.is_file() for p in (meta_path, data_path, sums_path)),
        "Missing artifact file",
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    require(meta["mode"] == "full_build", "Wrong build mode")
    require(meta["seed"] == 6304, "Wrong seed")
    require(
        meta["normalization"]["version"]
        == "nfkc-casefold-whitespace-v1",
        "Unexpected normalization",
    )
    require(
        meta["rules"]["max_formatted_tokens"] == 2048,
        "Wrong sequence-length limit",
    )
    require(
        meta["rules"]["shuffle"] is False
        and meta["rules"]["truncate"] is False,
        "Shuffling or truncation enabled",
    )

    require(
        meta["model"]["snapshot_sha"] == MODEL_SHA,
        "Wrong Qwen revision",
    )
    require(
        meta["datasets"]["metamath"]["revision"] == META_SHA,
        "Wrong MetaMathQA revision",
    )
    require(
        meta["datasets"]["gsm8k"]["revision"] == GSM_SHA,
        "Wrong GSM8K revision",
    )

    hashes = {
        "examples.jsonl": file_sha(data_path),
        "metadata.json": file_sha(meta_path),
    }

    declared = {}
    for line in sums_path.read_text(
        encoding="utf-8"
    ).splitlines():
        digest, name = line.split(maxsplit=1)
        declared[name.strip()] = digest

    require(
        declared == hashes,
        "SHA256SUMS disagrees with saved files",
    )
    require(
        meta["output_sha256"]["examples.jsonl"]
        == hashes["examples.jsonl"],
        "Dataset hash disagrees with metadata",
    )

    require(len(gsm_rows) == 1319, "Wrong GSM8K test-set size")

    leakage = {
        normalize(row["question"])
        for row in gsm_rows
    }

    gsm_meta = meta["datasets"]["gsm8k"]

    require(gsm_meta["rows"] == 1319, "GSM8K rows mismatch")
    require(
        gsm_meta["normalized_unique_questions"] == len(leakage),
        "GSM8K unique count mismatch",
    )
    require(
        gsm_meta["leakage_set_sha256"]
        == hashlib.sha256(
            canonical(sorted(leakage))
        ).hexdigest(),
        "GSM8K overlap-set hash mismatch",
    )
    require(
        gsm_meta["raw_rows_sha256"]
        == hashlib.sha256(canonical(gsm_rows)).hexdigest(),
        "GSM8K raw hash mismatch",
    )

    accepted = 0
    scanned = 0
    rejected = Counter()
    source_hash = hashlib.sha256()
    token_lengths = []
    supervised_lengths = []
    fallback_count = 0
    source_indices = []

    with data_path.open("rb") as saved:

        def next_saved():
            line = saved.readline()
            return json.loads(line) if line else None

        record = next_saved()

        for index, source in enumerate(metamath_rows):
            scanned += 1
            source_hash.update(canonical(source) + b"\n")

            query = source["query"]
            response = source["response"]

            require(
                isinstance(query, str)
                and isinstance(response, str),
                f"Non-string source at {index}",
            )

            if not query.strip():
                rejected["empty_query"] += 1
                continue

            if not response.strip():
                rejected["empty_response"] += 1
                continue

            original = source.get("original_question") or ""

            if (
                normalize(query) in leakage
                or normalize(original) in leakage
            ):
                rejected["gsm8k_overlap"] += 1
                continue

            ids, labels, fallback = independent_tokens(
                tokenizer, query, response
            )

            if len(ids) > 2048:
                rejected["over_length"] += 1
                continue

            require(
                record is not None,
                f"Saved data ended before source {index}",
            )
            require(
                record["source_index"] == index,
                f"Source-order mismatch at index {index}",
            )
            require(
                record["source"] == source,
                f"Raw source mismatch at index {index}",
            )
            require(
                record["input_ids"] == ids,
                f"Token IDs mismatch at source {index}",
            )
            require(
                record["labels"] == labels,
                f"Response masking mismatch at source {index}",
            )
            require(
                record["attention_mask"] == [1] * len(ids),
                f"Attention mask mismatch at source {index}",
            )
            require(
                record.get("boundary_retokenized") == fallback,
                f"Boundary flag mismatch at source {index}",
            )
            require(
                any(v != IGNORE for v in labels),
                f"No supervised tokens at source {index}",
            )
            require(
                all(
                    v == IGNORE or v == t
                    for v, t in zip(labels, ids)
                ),
                f"Label/token mismatch at source {index}",
            )

            accepted += 1
            source_indices.append(index)
            token_lengths.append(len(ids))
            supervised_lengths.append(
                sum(v != IGNORE for v in labels)
            )
            fallback_count += int(fallback)

            if accepted % 10000 == 0:
                print(
                    f"Independently checked "
                    f"{accepted}/{expected_count} examples",
                    flush=True,
                )

            record = next_saved()

            if accepted == expected_count:
                break

        require(
            accepted == expected_count,
            f"Recovered only {accepted}/{expected_count}",
        )
        require(
            record is None,
            "Extra saved examples after first 40000",
        )

    stats = meta["selection"]

    require(
        stats["accepted"] == accepted,
        "Accepted-count mismatch",
    )
    require(
        stats["scanned"] == scanned,
        "Scanned-count mismatch",
    )
    require(
        stats["accepted_source_indices"] == source_indices,
        "Source-index metadata mismatch",
    )
    require(
        stats["rejected"] == dict(rejected),
        "Rejection counts mismatch",
    )
    require(
        accepted + sum(rejected.values()) == scanned,
        "Selection totals do not reconcile",
    )
    require(
        stats["scanned_source_sha256"] == source_hash.hexdigest(),
        "Scanned-source hash mismatch",
    )

    return {
        "accepted": accepted,
        "scanned": scanned,
        "rejected": dict(rejected),
        "first_source_index": source_indices[0],
        "last_source_index": source_indices[-1],
        "boundary_retokenized": fallback_count,
        "formatted_tokens": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": round(statistics.mean(token_lengths), 2),
            "median": statistics.median(token_lengths),
        },
        "supervised_tokens": {
            "min": min(supervised_lengths),
            "max": max(supervised_lengths),
            "mean": round(statistics.mean(supervised_lengths), 2),
        },
        "examples_sha256": hashes["examples.jsonl"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(
        "Loading pinned tokenizer and GSM8K test set...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        revision=MODEL_SHA,
        use_fast=True,
    )

    gsm_rows = list(
        load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            streaming=True,
            revision=GSM_SHA,
        )
    )

    print(
        "Rescanning pinned MetaMathQA source order...",
        flush=True,
    )

    metamath_rows = load_dataset(
        "meta-math/MetaMathQA",
        split="train",
        streaming=True,
        revision=META_SHA,
    )

    result = validate(
        args.output_dir,
        tokenizer,
        gsm_rows,
        metamath_rows,
    )

    print("INDEPENDENT VALIDATION PASSED")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
