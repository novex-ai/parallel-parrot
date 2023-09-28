from parallel_parrot.util import (
    sum_usage_stats,
)


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
