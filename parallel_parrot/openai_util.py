import tiktoken


def openai_token_truncate(input: str, model: str, tokens_to_remove: int):
    encoding = tiktoken.encoding_for_model(model)
    encoded_tokens = encoding.encode(input)
    max_tokens = len(encoded_tokens) - tokens_to_remove
    truncated_tokens = encoded_tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
