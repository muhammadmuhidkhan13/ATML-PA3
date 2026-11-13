"""Read-only real-tokenizer validation across every fixed selected example."""
from collections import Counter
from pathlib import Path
from .io import read_jsonl, atomic_json, sha256
from .model import encode_example, QWEN_ID, QWEN_REVISION


def validate_tokenization(data_dir, output):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                               local_files_only=True, use_fast=True)
    data_dir = Path(data_dir)
    result = {"qwen_revision": QWEN_REVISION, "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
              "caps": {"question": 80, "caption": 64, "answer": 16}, "files": {}}
    for file in sorted(data_dir.glob("*.jsonl")):
        counts = Counter()
        caption = file.name.startswith("captions_")
        for row in read_jsonl(file):
            members = [row] if caption else row["members"]
            for member in members:
                encoded = encode_example(tokenizer, member["question"], member["target"], caption=caption)
                counts["examples"] += 1
                counts["question_truncated"] += encoded.question_truncated
                counts["target_truncated"] += encoded.target_truncated
                counts["supervised_tokens_including_eot"] += encoded.target_tokens + 1
                if len(encoded.input_ids) != len(encoded.labels):
                    raise AssertionError("Input/label mismatch")
                supervised = [i for i in encoded.labels if i != -100]
                if supervised[-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
                    raise AssertionError("Assistant EOT not supervised")
                if not encoded.target_truncated:
                    actual = tokenizer.decode(supervised, skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
                    if actual != member["target"] + "<|im_end|>":
                        raise AssertionError("Original target text changed")
        result["files"][file.name] = dict(counts)
        print(f"Tokenization verified: {file.name} ({counts['examples']})", flush=True)
    result["status"] = "passed"
    atomic_json(output, result)
    return result
