"""Regression tests for Qwen reply boundaries."""

import pytest
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

from atml_pa3.task1.formatting import format_example


@pytest.fixture(scope="module")
def tokenizer():
    cached = try_to_load_from_cache(
        "Qwen/Qwen3-0.6B",
        "tokenizer_config.json",
    )
    if not isinstance(cached, str):
        pytest.fail("Qwen tokenizer must already be cached")

    return AutoTokenizer.from_pretrained(
        str(Path(cached).parent),
        local_files_only=True,
        use_fast=True,
    )


@pytest.mark.parametrize("response", [
    " 42",
    "  answer with leading spaces",
    "answer with trailing spaces  ",
    "\nThe answer is 42.\n",
    "  First line\n second line \n",
    "\tIndented text\t",
    "数学 café 😀",
    "### Solution\n\\boxed{42}",
])
def test_exact_training_text_and_mask(tokenizer, response):
    example = format_example(
        tokenizer,
        "Solve this question, please.",
        response,
    )

    supervised = [
        x for x in example["labels"] if x != -100
    ]

    assert tokenizer.decode(
        supervised,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == response + "<|im_end|>"

    assert len(example["input_ids"]) == len(example["labels"])

    assert all(
        x == -100 or x == y
        for x, y in zip(
            example["labels"],
            example["input_ids"],
        )
    )


def test_forced_boundary_fallback(tokenizer):
    response = "42 is the answer."

    class BoundarySimulation:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, name):
            return getattr(self.actual, name)

        def __call__(self, text, **kwargs):
            out = self.actual(text, **kwargs)

            if kwargs.get("return_offsets_mapping"):
                start = text.rfind(response + "<|im_end|>")
                offsets = list(out["offset_mapping"])

                candidates = [
                    i for i, (a, b) in enumerate(offsets)
                    if b > start and a < start + len(response)
                ]

                assert candidates
                idx = candidates[0]
                _, b = offsets[idx]

                offsets[idx] = (start - 1, b)
                out["offset_mapping"] = offsets

            return out

    example = format_example(
        BoundarySimulation(tokenizer),
        "Q?",
        response,
    )

    assert example["boundary_retokenized"] is True

    supervised = [
        x for x in example["labels"] if x != -100
    ]

    assert tokenizer.decode(
        supervised,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ) == response + "<|im_end|>"
