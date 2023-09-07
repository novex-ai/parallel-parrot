from aioresponses import aioresponses
import pytest

import parallel_parrot as pp


@pytest.fixture
def mock_aioresponse():
    with aioresponses(passthrough=[]) as m:
        yield m


@pytest.fixture
def openai_chat_completion_config():
    return pp.OpenAIChatCompletionConfig(
        openai_api_key="*suupersekret*", model="gpt-3.5-turbo-0613"
    )


def test_parrot_openai_chat_completion_dictlist(
    mock_aioresponse, openai_chat_completion_config
):
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "x-ratelimit-limit-requests": "3500",
        },
        payload={
            "id": "chatcmpl-7wGexgOcfdurdLgbolOd6xMV2vqUB",
            "object": "chat.completion",
            "created": 1694121419,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "2"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 37, "completion_tokens": 1, "total_tokens": 38},
        },
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        status=429,
        headers={
            "retry-after": "0.01",
        },
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "id": "chatcmpl-7wGex22aPD9J2ZViFSzLucaPruzMH",
            "object": "chat.completion",
            "created": 1694121419,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "4"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 37, "completion_tokens": 1, "total_tokens": 38},
        },
    )
    (output_list, usage_stats_sum) = pp.sync_run(
        pp.parrot_openai_chat_completion_dictlist(
            config=openai_chat_completion_config,
            input_list=[
                {"input": "what is 1+1?"},
                {"input": "what is 2+2?"},
            ],
            prompt_template="Q: ${input}\nA:",
            output_key="output",
            system_message="you are a super-precise calculator that returns correct answers in integer form",
        )
    )
    assert output_list == [
        {"input": "what is 1+1?", "output": "2"},
        {"input": "what is 2+2?", "output": "4"},
    ]
    assert usage_stats_sum == {
        "completion_tokens": 2,
        "prompt_tokens": 74,
        "total_tokens": 76,
    }
