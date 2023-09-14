import dataclasses

from aioresponses import aioresponses
import pytest

import parallel_parrot as pp
from parallel_parrot.openai import (
    parallel_openai_chat_completion_dictlist,
    parallel_openai_chat_completion_pandas,
    parallel_openai_chat_completion_exploding_function_dictlist,
    parallel_openai_chat_completion_exploding_function_pandas,
)

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None


@pytest.fixture(scope="function")
def mock_aioresponse():
    with aioresponses(passthrough=[]) as m:
        yield m


@pytest.fixture
def openai_chat_completion_config():
    return pp.OpenAIChatCompletionConfig(
        openai_api_key="*suupersekret*", model="gpt-3.5-turbo-0613"
    )


def test_parallel_openai_chat_completion_dictlist(
    mock_aioresponse, openai_chat_completion_config
):
    config = dataclasses.replace(
        openai_chat_completion_config,
        system_message="you are a super-precise calculator that returns correct answers in integer form",
    )
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
        status=409,
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
    (output_list, usage_stats_sum) = pp.run_async(
        parallel_openai_chat_completion_dictlist(
            config=config,
            input_list=[
                {"input": "what is 1+1?"},
                {"input": "what is 2+2?"},
            ],
            prompt_template="Q: ${input}\nA:",
            output_key="output",
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


@pytest.mark.skipif(pd is None, reason="requires pandas")
def test_parallel_openai_chat_completion_pandas(
    mock_aioresponse, openai_chat_completion_config
):
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "x-ratelimit-limit-requests": "3500",
        },
        payload={
            "id": "chatcmpl-7wMJooPRFYJLk1PKLQLAEt2kPmAHF",
            "object": "chat.completion",
            "created": 1694143172,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "POSITIVE"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 41, "completion_tokens": 2, "total_tokens": 43},
        },
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
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
    )
    input_df = pd.DataFrame(
        {
            "input": [
                "this is a super duper product that will change the world",
                "do not buy this",
            ],
            "source": [
                "amazon",
                "shopify",
            ],
        },
        index=[100, 101],
    )
    (output_df, usage_stats_sum) = pp.run_async(
        parallel_openai_chat_completion_pandas(
            config=openai_chat_completion_config,
            input_df=input_df,
            prompt_template="""
What is the sentiment of this product review?
POSITIVE or NEGATIVE?
product review: ${input}
sentiment:""",
            output_key="sentiment",
        )
    )
    assert output_df["sentiment"].tolist() == ["POSITIVE", "NEGATIVE"]
    assert usage_stats_sum == {
        "completion_tokens": 4,
        "prompt_tokens": 74,
        "total_tokens": 78,
    }


def test_parallel_openai_chat_completion_exploding_function_dictlist(
    mock_aioresponse, openai_chat_completion_config
):
    config = dataclasses.replace(
        openai_chat_completion_config,
        n=2,
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "x-ratelimit-limit-requests": "3500",
        },
        payload={
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
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "id": "chatcmpl-7wMjMwUJfE63ZifYkcxJUZbtkSwD1",
            "object": "chat.completion",
            "created": 1694144756,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "f",
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was John Adams?",\n      "answer": "John Adams was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States."\n    },\n    {\n      "question": "What role did John Adams play in the American Revolution?",\n      "answer": "Before his presidency, John Adams was a leader of the American Revolution that achieved independence from Great Britain."\n    },\n    {\n      "question": "What positions did John Adams hold in the U.S. government?",\n      "answer": "John Adams served as the vice president of the United States from 1789 to 1797, and as the second president of the United States from 1797 to 1801."\n    },\n    {\n      "question": "Who did John Adams correspond with regularly?",\n      "answer": "John Adams regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson."\n    }\n  ]\n}',
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
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was John Adams?",\n      "answer": "John Adams was an American statesman, attorney, diplomat, writer, and Founding Father."\n    },\n    {\n      "question": "What position did John Adams hold in the U.S. government?",\n      "answer": "John Adams was the first person to hold the office of vice president of the United States."\n    },\n    {\n      "question": "Who were some of the important contemporaries John Adams corresponded with?",\n      "answer": "John Adams regularly corresponded with his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson."\n    }\n  ]\n}',
                        },
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 270,
                "completion_tokens": 358,
                "total_tokens": 628,
            },
        },
    )
    (output_list, usage_stats_sum) = pp.run_async(
        parallel_openai_chat_completion_exploding_function_dictlist(
            config=config,
            input_list=[
                {
                    "input": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
                """.strip()
                },
                {
                    "input": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
               """.strip()
                },
            ],
            prompt_template="""
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.  If there are no questions and answers, output an empty list.
document: ${input}
        """,
            output_key_names=["question", "answer"],
        )
    )
    assert output_list[0] == {
        "input": 'George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".',
        "question": "Who was George Washington?",
        "answer": "George Washington was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797.",
    }
    assert usage_stats_sum == {
        "completion_tokens": 849,
        "total_tokens": 1348,
        "prompt_tokens": 499,
    }


@pytest.mark.skipif(pd is None, reason="requires pandas")
def test_parallel_openai_chat_completion_exploding_function_pandas(
    mock_aioresponse, openai_chat_completion_config
):
    config = dataclasses.replace(
        openai_chat_completion_config,
        n=2,
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "x-ratelimit-limit-requests": "3500",
        },
        payload={
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
    )
    mock_aioresponse.post(
        "https://api.openai.com/v1/chat/completions",
        payload={
            "id": "chatcmpl-7wMjMwUJfE63ZifYkcxJUZbtkSwD1",
            "object": "chat.completion",
            "created": 1694144756,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "f",
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was John Adams?",\n      "answer": "John Adams was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States."\n    },\n    {\n      "question": "What role did John Adams play in the American Revolution?",\n      "answer": "Before his presidency, John Adams was a leader of the American Revolution that achieved independence from Great Britain."\n    },\n    {\n      "question": "What positions did John Adams hold in the U.S. government?",\n      "answer": "John Adams served as the vice president of the United States from 1789 to 1797, and as the second president of the United States from 1797 to 1801."\n    },\n    {\n      "question": "Who did John Adams correspond with regularly?",\n      "answer": "John Adams regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson."\n    }\n  ]\n}',
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
                            "arguments": '{\n  "p": [\n    {\n      "question": "Who was John Adams?",\n      "answer": "John Adams was an American statesman, attorney, diplomat, writer, and Founding Father."\n    },\n    {\n      "question": "What position did John Adams hold in the U.S. government?",\n      "answer": "John Adams was the first person to hold the office of vice president of the United States."\n    },\n    {\n      "question": "Who were some of the important contemporaries John Adams corresponded with?",\n      "answer": "John Adams regularly corresponded with his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson."\n    }\n  ]\n}',
                        },
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 270,
                "completion_tokens": 358,
                "total_tokens": 628,
            },
        },
    )
    (output_df, usage_stats_sum) = pp.run_async(
        parallel_openai_chat_completion_exploding_function_pandas(
            config=config,
            input_df=pd.DataFrame(
                [
                    {
                        "input": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
                """.strip()
                    },
                    {
                        "input": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
               """.strip()
                    },
                ]
            ),
            prompt_template="""
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.  If there are no questions and answers, output an empty list.
document: ${input}
        """,
            output_key_names=["question", "answer"],
        )
    )
    assert output_df.loc[0, "question"] == "Who was George Washington?"
    assert usage_stats_sum == {
        "completion_tokens": 849,
        "total_tokens": 1348,
        "prompt_tokens": 499,
    }
