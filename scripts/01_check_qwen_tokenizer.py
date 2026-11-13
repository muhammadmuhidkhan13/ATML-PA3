from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision="main",
    use_fast=True,
)

if not tokenizer.is_fast:
    raise AssertionError(
        "This check requires the fast tokenizer for offset mappings."
    )

question = "What is 2 + 2?"
response = "2 + 2 = 4."

messages = [
    {
        "role": "user",
        "content": question,
    },
    {
        "role": "assistant",
        "content": response,
    },
]

# First render the exact text produced by Qwen's current chat template.
complete_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)

# Tokenize that rendered text while retaining character-to-token offsets.
encoded = tokenizer(
    complete_text,
    add_special_tokens=False,
    return_offsets_mapping=True,
)

complete_ids = encoded["input_ids"]
offsets = encoded["offset_mapping"]

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

if im_end_id is None or im_end_id == tokenizer.unk_token_id:
    raise AssertionError(
        "The tokenizer does not recognize <|im_end|>."
    )

# Because the assistant is the final message, its response should be followed
# immediately by its end-of-turn token.
assistant_suffix = response + "<|im_end|>"
response_start_char = complete_text.rfind(assistant_suffix)

if response_start_char == -1:
    print("Rendered conversation:")
    print(repr(complete_text))

    raise AssertionError(
        "Could not locate the assistant response followed by <|im_end|>."
    )

response_end_char = response_start_char + len(response)

# Find every token whose character span overlaps the supplied response.
response_token_positions = [
    index
    for index, (start, end) in enumerate(offsets)
    if end > response_start_char and start < response_end_char
]

if not response_token_positions:
    raise AssertionError(
        "No tokens were mapped to the assistant response."
    )

first_response_token = response_token_positions[0]
last_response_token = response_token_positions[-1]

# Verify that the first response token does not also contain prompt text.
first_start, first_end = offsets[first_response_token]

if first_start < response_start_char:
    raise AssertionError(
        "The first response token overlaps the assistant header. "
        f"Token offset: {(first_start, first_end)}, "
        f"response starts at: {response_start_char}"
    )

# Locate the assistant's closing end-of-turn token.
eot_positions = [
    index
    for index, token_id in enumerate(complete_ids)
    if token_id == im_end_id and index > last_response_token
]

if not eot_positions:
    raise AssertionError(
        "Could not locate <|im_end|> after the assistant response."
    )

# The first <|im_end|> after the response closes the assistant message.
assistant_eot_position = eot_positions[0]

# Mask every token by default.
labels = [-100] * len(complete_ids)

# Supervise the supplied assistant-response tokens.
for index in response_token_positions:
    labels[index] = complete_ids[index]

# Also supervise the assistant end-of-turn token.
labels[assistant_eot_position] = complete_ids[assistant_eot_position]

supervised_ids = [
    label
    for label in labels
    if label != -100
]

decoded_complete = tokenizer.decode(
    complete_ids,
    skip_special_tokens=False,
)

decoded_supervised = tokenizer.decode(
    supervised_ids,
    skip_special_tokens=False,
)

if len(labels) != len(complete_ids):
    raise AssertionError("Input and label lengths do not match.")

if supervised_ids[-1] != im_end_id:
    raise AssertionError(
        "The supervised region does not end with <|im_end|>."
    )

if response not in decoded_supervised:
    raise AssertionError(
        "The supplied assistant response is missing from the supervised tokens."
    )

thinking_scaffold_present = (
    "<think>" in complete_text or "</think>" in complete_text
)

if thinking_scaffold_present:
    if "<think>" not in complete_text or "</think>" not in complete_text:
        raise AssertionError(
            "The thinking scaffold is incomplete."
        )

    think_start = complete_text.find("<think>") + len("<think>")
    think_end = complete_text.find("</think>", think_start)
    thinking_content = complete_text[think_start:think_end]

    if thinking_content.strip():
        raise AssertionError(
            "Thinking is not disabled: the thinking block contains text."
        )

# Confirm that prompt and template tokens remain masked.
for index in range(first_response_token):
    if labels[index] != -100:
        raise AssertionError(
            "A token before the assistant response is being supervised."
        )

print()
print("Tokenizer revision:", tokenizer.init_kwargs.get("_commit_hash"))
print("Empty thinking scaffold present:", thinking_scaffold_present)
print("Total formatted tokens:", len(complete_ids))
print("Masked tokens before response:", first_response_token)
print("Assistant response tokens:", len(response_token_positions))
print("Supervised response + EOT tokens:", len(supervised_ids))
print(
    "Masked trailing tokens:",
    len(complete_ids) - assistant_eot_position - 1,
)

print()
print("Template text immediately before the response:")
print(repr(complete_text[max(0, response_start_char - 120):response_start_char]))

print()
print("Complete formatted conversation:")
print(repr(decoded_complete))

print()
print("Supervised region:")
print(repr(decoded_supervised))

print()
print("Qwen tokenizer and assistant-loss masking check passed.")

