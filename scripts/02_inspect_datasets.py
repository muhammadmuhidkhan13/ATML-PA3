from itertools import islice

from datasets import load_dataset


def preview_dataset(name, dataset, rows=3):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Features:", dataset.features)
    print()

    examples = list(islice(dataset, rows))

    for index, example in enumerate(examples):
        print(f"Example {index}")
        print("Fields:", list(example.keys()))

        for key, value in example.items():
            text = repr(value)

            if len(text) > 300:
                text = text[:300] + "..."

            print(f"  {key}: {text}")

        print()


print("Loading MetaMathQA in streaming mode...")

metamath = load_dataset(
    "meta-math/MetaMathQA",
    split="train",
    streaming=True,
)

print("Loading GSM8K test split in streaming mode...")

gsm8k = load_dataset(
    "openai/gsm8k",
    "main",
    split="test",
    streaming=True,
)

preview_dataset(
    "MetaMathQA train",
    metamath,
)

preview_dataset(
    "GSM8K main/test",
    gsm8k,
)

expected_metamath_fields = {
    "query",
    "response",
    "original_question",
}

expected_gsm8k_fields = {
    "question",
    "answer",
}

metamath_fields = set(metamath.features.keys())
gsm8k_fields = set(gsm8k.features.keys())

missing_metamath = expected_metamath_fields - metamath_fields
missing_gsm8k = expected_gsm8k_fields - gsm8k_fields

if missing_metamath:
    raise AssertionError(
        f"MetaMathQA is missing expected fields: {sorted(missing_metamath)}"
    )

if missing_gsm8k:
    raise AssertionError(
        f"GSM8K is missing expected fields: {sorted(missing_gsm8k)}"
    )

print("=" * 80)
print("Dataset schema check passed.")
print("=" * 80)
