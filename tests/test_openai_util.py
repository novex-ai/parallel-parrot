from parallel_parrot.openai_util import openai_token_truncate


def test_openai_token_truncate():
    input = "This is a test.  It is a very simple test.  But we think its useful."
    model = "gpt-3.5-turbo"
    max_tokens = 5
    truncated = openai_token_truncate(input, model, max_tokens)
    assert truncated == "This is a test."
