from functools import reduce
import logging
from typing import List


logger = logging.getLogger(__name__.split(".")[0])
logger.addHandler(logging.NullHandler())


def sum_usage_stats(usage_stats_list: List[dict]) -> dict:
    return reduce(
        lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
        usage_stats_list,
    )
