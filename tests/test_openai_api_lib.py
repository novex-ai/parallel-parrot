import pytest

from parallel_parrot.types import ParallelParrotError
from parallel_parrot.openai_api_lib import (
    OPENAI_EMPTY_USAGE_STATS,
    prep_openai_function_list_of_objects,
    parse_chat_completion_message_and_usage,
    parse_content_length_exceeded_error,
    parse_seconds_from_header,
)


def test_prep_openai_function_list_of_objects():
    function_name = "test_function"
    parameter_name = "test_parameter"
    output_key_names = ["key1", "key2", "key3"]
    expected_functions = [
        {
            "name": function_name,
            "parameters": {
                "type": "object",
                "properties": {
                    parameter_name: {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "key1": {"type": "string"},
                                "key2": {"type": "string"},
                                "key3": {"type": "string"},
                            },
                            "required": ["key1", "key2", "key3"],
                        },
                    },
                },
            },
        },
    ]
    expected_function_call = {"name": function_name}

    functions, function_call = prep_openai_function_list_of_objects(
        function_name, parameter_name, output_key_names
    )

    assert functions == expected_functions
    assert function_call == expected_function_call

    with pytest.raises(ParallelParrotError):
        prep_openai_function_list_of_objects(function_name, parameter_name, [])


def test_parse_chat_completion_message_and_usage_simple():
    empty_response = (
        None,
        OPENAI_EMPTY_USAGE_STATS,
    )
    assert parse_chat_completion_message_and_usage({}) == empty_response
    assert (
        parse_chat_completion_message_and_usage(
            {"object": "chat.completion", "choices": []}
        )
        == empty_response
    )
    assert (
        parse_chat_completion_message_and_usage(
            {
                "error": {
                    "message": "You exceeded your current quota, please check your plan and billing details.",
                    "type": "insufficient_quota",
                    "param": None,
                    "code": "insufficient_quota",
                }
            }
        )
        == empty_response
    )


def test_parse_chat_completion_message_and_usage_text():
    assert parse_chat_completion_message_and_usage(
        {
            "id": "chatcmpl-7wMJoagRJIyE35cVIMTzEtHCnlqCa",
            "object": "chat.completion",
            "created": 1694143172,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "NEGATIVE"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35},
        },
    ) == ("NEGATIVE", {"prompt_tokens": 33, "completion_tokens": 2, "total_tokens": 35})
    assert parse_chat_completion_message_and_usage(
        {
            "id": "chatcmpl-7wMJoagRJIyE35cVIMTzEtHCnlqCa",
            "object": "chat.completion",
            "created": 1694143172,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "NEGATIVE"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "NEGATIVE"},
                    "finish_reason": "stop",
                },
                {
                    "index": 2,
                    "message": {"role": "assistant", "content": "NEGATIVE"},
                    "finish_reason": "stop",
                },
            ],
            "usage": {"prompt_tokens": 33, "completion_tokens": 6, "total_tokens": 39},
        }
    ) == (
        ["NEGATIVE"],
        {"prompt_tokens": 33, "completion_tokens": 6, "total_tokens": 39},
    )


def test_parse_chat_completion_message_and_usage_function_call():
    assert parse_chat_completion_message_and_usage(
        {
            "id": "chatcmpl-7wMjGKIjYVB5KxUEWixvT2AZzPAuA",
            "object": "chat.completion",
            "created": 1694144750,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "f",
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was George Washington?",\n      "answer": "George Washington was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797."\n    },\n    {\n      "question": "When did George Washington serve as the first president of the United States?",\n      "answer": "George Washington served as the first president of the United States from 1789 to 1797."\n    },\n    {\n      "question": "What role did George Washington play in the American Revolutionary War?",\n      "answer": "George Washington led Patriot forces to victory in the American Revolutionary War."\n    },\n    {\n      "question": "What role did George Washington play in drafting the Constitution of the United States?",\n      "answer": "George Washington served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States."\n    },\n    {\n      "question": "What is George Washington often referred to as?",\n      "answer": "George Washington is often referred to as the \\"Father of his Country\\"."\n    }\n  ]\n}',
                        },
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "f",
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was George Washington?",\n      "answer": "George Washington was an American military officer, statesman, and the first president of the United States."\n    },\n    {\n      "question": "When was George Washington born?",\n      "answer": "George Washington was born on February 22, 1732."\n    },\n    {\n      "question": "When did George Washington die?",\n      "answer": "George Washington died on December 14, 1799."\n    },\n    {\n      "question": "What role did George Washington play in the American Revolutionary War?",\n      "answer": "George Washington led Patriot forces to victory in the American Revolutionary War."\n    },\n    {\n      "question": "What role did George Washington play in the Constitutional Convention?",\n      "answer": "George Washington served as the president of the Constitutional Convention in 1787."\n    },\n    {\n      "question": "Why is George Washington called the \'Father of his Country\'?",\n      "answer": "George Washington is called the \'Father of his Country\' because of his significant contributions to the founding of the United States and establishment of the American federal government."\n    }\n  ]\n}',
                        },
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 229,
                "completion_tokens": 491,
                "total_tokens": 720,
            },
        },
        "f",
        "p",
    ) == (
        [
            {
                "question": "Who was George Washington?",
                "answer": "George Washington was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797.",
            },
            {
                "question": "When did George Washington serve as the first president of the United States?",
                "answer": "George Washington served as the first president of the United States from 1789 to 1797.",
            },
            {
                "answer": "George Washington led Patriot forces to victory in the American "
                "Revolutionary War.",
                "question": "What role did George Washington play in the American "
                "Revolutionary War?",
            },
            {
                "answer": "George Washington served as president of the Constitutional "
                "Convention in 1787, which drafted and ratified the Constitution "
                "of the United States.",
                "question": "What role did George Washington play in drafting the "
                "Constitution of the United States?",
            },
            {
                "answer": 'George Washington is often referred to as the "Father of his '
                'Country".',
                "question": "What is George Washington often referred to as?",
            },
            {
                "answer": "George Washington was an American military officer, statesman, "
                "and the first president of the United States.",
                "question": "Who was George Washington?",
            },
            {
                "answer": "George Washington was born on February 22, 1732.",
                "question": "When was George Washington born?",
            },
            {
                "answer": "George Washington died on December 14, 1799.",
                "question": "When did George Washington die?",
            },
            {
                "answer": "George Washington led Patriot forces to victory in the American "
                "Revolutionary War.",
                "question": "What role did George Washington play in the American "
                "Revolutionary War?",
            },
            {
                "answer": "George Washington served as the president of the Constitutional "
                "Convention in 1787.",
                "question": "What role did George Washington play in the Constitutional "
                "Convention?",
            },
            {
                "answer": "George Washington is called the 'Father of his Country' because "
                "of his significant contributions to the founding of the United "
                "States and establishment of the American federal government.",
                "question": "Why is George Washington called the 'Father of his Country'?",
            },
        ],
        {
            "prompt_tokens": 229,
            "completion_tokens": 491,
            "total_tokens": 720,
        },
    )


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


def test_parse_seconds_from_header():
    assert parse_seconds_from_header(None) is None
    assert parse_seconds_from_header("0.123s") == 0.123
    assert parse_seconds_from_header("1m20s") == 80.0
    assert parse_seconds_from_header("1m") == 60.0
