from parallel_parrot.util import (
    make_curried_prompt_template,
    sum_usage_stats,
)


def test_make_curried_prompt_template():
    prompt_template = "${a}--${b}"
    curried_prompt_template = make_curried_prompt_template(prompt_template)
    assert curried_prompt_template({"a": "alpha", "b": "beta"}) == "alpha--beta"


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
