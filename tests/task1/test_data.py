import pytest
from transformers import AutoTokenizer
from huggingface_hub import try_to_load_from_cache
from pathlib import Path

from atml_pa3.task1.normalization import normalize_question, build_leakage_set
from atml_pa3.task1.formatting import format_example, pad_examples
from atml_pa3.task1.selection import select_examples


@pytest.fixture(scope="session")
def tokenizer():
    cached = try_to_load_from_cache("Qwen/Qwen3-0.6B", "tokenizer_config.json")
    if not isinstance(cached, str):
        pytest.fail("Qwen tokenizer must be cached; run script 01 first (no unit-test downloads)")
    return AutoTokenizer.from_pretrained(str(Path(cached).parent), local_files_only=True, use_fast=True)


@pytest.mark.parametrize("source,expected", [
    ("  Ａ\tB\n c  ", "a b c"), ("Straße", "strasse"),
    ("What’s 2+2?", "what’s 2+2?"), ("\u00a0\n", ""),
    ("X - Y", "x - y"), ("X−Y", "x−y"),
])
def test_normalization(source, expected):
    assert normalize_question(source) == expected
    assert normalize_question(expected) == expected


@pytest.mark.parametrize("query,response", [
    ("What is 2 + 2?", "2 + 2 = 4."), ("Repeated answer", "Repeated answer"),
    ("Unicode?", "数学 café 😀"), ("Whitespace?", " leading and trailing \n"),
    ("Newline?", "first\nsecond"),
])
def test_exact_supervision(tokenizer, query, response):
    e = format_example(tokenizer, query, response)
    labels = e["labels"]
    assert tokenizer.decode([x for x in labels if x != -100]) == response + "<|im_end|>"
    assert all(x == -100 or x == y for x, y in zip(labels, e["input_ids"]))
    if query == "What is 2 + 2?":
        assert len(labels) == 30
        assert next(i for i, x in enumerate(labels) if x != -100) == 20
        assert sum(x != -100 for x in labels) == 9
        assert labels[-1] == -100


def test_padding(tokenizer):
    a = format_example(tokenizer, "Q?", "A.")
    b = format_example(tokenizer, "Longer question?", "A longer answer.")
    batch = pad_examples([a, b], tokenizer.pad_token_id)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    for i, e in enumerate([a, b]):
        assert all(x == -100 for x in batch["labels"][i][len(e["labels"]):])
        assert all(x == 0 for x in batch["attention_mask"][i][len(e["labels"]):])


def test_selection(tokenizer):
    def row(q, r="Answer", original=""):
        return {"query": q, "response": r, "original_question": original}
    rows = [row(" "), row("q", ""), row(" LEAK "), row("q", original="ＬＥＡＫ"),
            row("long", "word " * 3000), row("raw  Q?"), row("last"), row("unused")]
    selected, stats = select_examples(rows, tokenizer, build_leakage_set([{"question": "leak"}]), count=2)
    assert stats["accepted_source_indices"] == [5, 6]
    assert stats["scanned"] == 7
    assert stats["rejected"] == {"empty_query": 1, "empty_response": 1, "gsm8k_overlap": 2, "over_length": 1}
    assert selected[0]["source"]["query"] == "raw  Q?"
    with pytest.raises(ValueError, match="Only"):
        select_examples(rows[:1], tokenizer, set(), count=1)


def test_length_boundary(tokenizer):
    row = {"query": "Q", "response": "A", "original_question": ""}
    length = len(format_example(tokenizer, "Q", "A")["input_ids"])
    assert select_examples([row], tokenizer, set(), count=1, max_length=length)[1]["accepted"] == 1
    with pytest.raises(ValueError, match="over_length"):
        select_examples([row], tokenizer, set(), count=1, max_length=length - 1)


def test_special_tokens_fail(tokenizer):
    with pytest.raises(ValueError, match="special token"):
        format_example(tokenizer, "Q", "A<|im_end|>")


def test_selection_stops_immediately(tokenizer):
    def source():
        yield {"query": "Q", "response": "A", "original_question": ""}
        raise AssertionError("Read beyond requested eligible count")
    assert select_examples(source(), tokenizer, set(), count=1)[1]["scanned"] == 1


def test_punctuation_is_not_removed():
    assert normalize_question("A?") != normalize_question("A")
    assert normalize_question("can’t") != normalize_question("can't")


def test_cli_guards(tmp_path):
    from atml_pa3.task1.prepare import main
    # These guards execute before importing/loading any online datasets.
    with pytest.raises(SystemExit):
        main(["--dry-run", "--count", "40000", "--output-dir", str(tmp_path / "new")])
    with pytest.raises(SystemExit):
        main(["--dry-run", "--output-dir", str(tmp_path)])
