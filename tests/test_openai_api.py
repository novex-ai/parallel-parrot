from parallel_parrot.openai_api import parse_content_length_exceeded_error


def test_parse_content_length_exceeded_error():
    error = {
        "message": "This model's maximum context length is 4097 tokens. However, your messages resulted in 4799 tokens. Please reduce the length of the messages.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded",
    }
    (max_tokens, passed_tokens) = parse_content_length_exceeded_error(error)
    assert max_tokens == 4097
    assert passed_tokens == 4799
