from parallel_parrot.util import (
    input_list_to_prompts,
    sum_usage_stats,
)


def test_input_list_to_prompts():
    input_list = [
        {"a": "alpha", "b": "beta"},
        {"a": "ALPHA", "b": "BETA"},
    ]
    prompt_template = "${a}--${b}"
    prompts = input_list_to_prompts(input_list, prompt_template)
    assert prompts == [
        "alpha--beta",
        "ALPHA--BETA",
    ]


def test_sum_usage_stats():
    usage_stats_list = [
        {
            "total_tokens": 100,
            "prompt_tokens": 30,
            "completion_tokens": 70,
        },
        {
            "total_tokens": 200,
            "prompt_tokens": 60,
            "completion_tokens": 140,
        },
    ]
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    assert usage_stats_sum == {
        "total_tokens": 300,
        "prompt_tokens": 90,
        "completion_tokens": 210,
    }
