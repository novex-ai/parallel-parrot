import tiktoken


def openai_token_truncate(input: str, model: str, max_tokens: int):
    encoding = tiktoken.encoding_for_model(model)
    encoded_tokens = encoding.encode(input)
    truncated_tokens = encoded_tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
