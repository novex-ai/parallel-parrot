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
        openai_api_key="*no-so-seekret*",
        model="gpt-3.5-turbo-0613"
    )


def test_parrot_openai_chat_completion_dictlist(
    mock_aioresponse,
    openai_chat_completion_config
):
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            'x-ratelimit-limit-requests': '3500',
        },
        payload={
            'id': 'chatcmpl-7wG8BdZa9qGmkovOqT1e9ZmNDCusV',
            'object': 'chat.completion',
            'created': 1694119387,
            'model': 'gpt-3.5-turbo-0613',
            'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': '2'}, 'finish_reason': 'stop'}
            ],
            'usage': {
                'prompt_tokens': 34,
                'completion_tokens': 1,
                'total_tokens': 35
            }
        },
    )
    (output_list, usage_stats_sum) = pp.sync_run(pp.parrot_openai_chat_completion_dictlist(
        config=openai_chat_completion_config,
        input_list=[
            {"input": "what is 1+1?"},
        ],
        prompt_template="Q: ${input}\nA:",
        output_key="output",
        system_message="you are a super-precise calculator that only returns answers"
    ))
    assert output_list == [
        {'input': 'what is 1+1?', 'output': '2'},
    ]
    assert usage_stats_sum == {
        'completion_tokens': 1,
        'prompt_tokens': 34,
        'total_tokens': 35
    }
