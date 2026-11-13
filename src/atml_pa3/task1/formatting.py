"""Exact assistant-only supervision with boundary-safe tokenization."""

EOT = "<|im_end|>"
IGNORE = -100


def _decode(tokenizer, ids):
    return tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _tokenize(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def format_example(tokenizer, query: str, response: str):
    if not query.strip() or not response.strip():
        raise ValueError("Empty query or response")

    if any(
        token in text
        for text in (query, response)
        for token in tokenizer.all_special_tokens
    ):
        raise ValueError("Source text contains a tokenizer special token")

    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for offsets")

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    start = rendered.rfind(response + EOT)

    if start < 0:
        raise AssertionError(
            "Template did not preserve supplied response verbatim"
        )

    end = start + len(response)
    eot_id = tokenizer.convert_tokens_to_ids(EOT)

    if eot_id is None or eot_id == tokenizer.unk_token_id:
        raise AssertionError("Tokenizer does not recognize assistant EOT")

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    positions = [
        i
        for i, (a, b) in enumerate(offsets)
        if b > start and a < end
    ]

    clean = (
        bool(positions)
        and offsets[positions[0]][0] == start
        and offsets[positions[-1]][1] == end
        and positions == list(range(positions[0], positions[-1] + 1))
    )

    if clean:
        eot_pos = positions[-1] + 1
        clean = (
            eot_pos < len(ids)
            and ids[eot_pos] == eot_id
            and tuple(offsets[eot_pos]) == (end, end + len(EOT))
            and _decode(
                tokenizer,
                [ids[i] for i in positions] + [ids[eot_pos]],
            ) == response + EOT
        )

    if clean:
        labels = [IGNORE] * len(ids)

        for i in positions + [eot_pos]:
            labels[i] = ids[i]

        retokenized = False

    else:
        # Avoid token boundaries that mix response and template text.
        prefix = rendered[:start]
        suffix = rendered[end + len(EOT):]

        prefix_ids = _tokenize(tokenizer, prefix)
        response_ids = _tokenize(tokenizer, response)
        suffix_ids = _tokenize(tokenizer, suffix)

        if not response_ids:
            raise AssertionError("Nonempty response produced no tokens")

        ids = prefix_ids + response_ids + [eot_id] + suffix_ids

        labels = (
            [IGNORE] * len(prefix_ids)
            + response_ids
            + [eot_id]
            + [IGNORE] * len(suffix_ids)
        )

        if _decode(tokenizer, ids) != rendered:
            raise AssertionError(
                "Boundary-safe tokenization changed chat text"
            )

        retokenized = True

    supervised = [x for x in labels if x != IGNORE]

    if _decode(tokenizer, supervised) != response + EOT:
        raise AssertionError(
            "Supervised tokens differ from response plus EOT"
        )

    if not (
        len(ids) == len(labels)
        and all(
            label == IGNORE or label == token
            for token, label in zip(ids, labels)
        )
    ):
        raise AssertionError("Invalid response-only supervision")

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "boundary_retokenized": retokenized,
    }


def pad_examples(examples, pad_token_id):
    """Right-padding with ignored padding labels."""
    length = max(len(e["input_ids"]) for e in examples)

    return {
        key: [
            e[key] + [fill] * (length - len(e[key]))
            for e in examples
        ]
        for key, fill in (
            ("input_ids", pad_token_id),
            ("attention_mask", 0),
            ("labels", IGNORE),
        )
    }
