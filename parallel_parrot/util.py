from collections.abc import Callable
from functools import reduce
import logging
from string import Template
import sys
from typing import List

from .types import ParallelParrotError


logger = logging.getLogger(__name__.split(".")[0])
logger.addHandler(logging.NullHandler())


def make_curried_prompt_template(prompt_template: str) -> Callable:
    prompt_template = prompt_template.strip()
    t = Template(prompt_template)
    if sys.version_info >= (3, 11):
        if not t.is_valid():
            raise ParallelParrotError(f"Invalid template {prompt_template=}")
        identifiers_set = set(t.get_identifiers())
    else:
        identifiers_set = None

    def f(input_dict: dict) -> str:
        if identifiers_set and identifiers_set > set(input_dict.keys()):
            raise ParallelParrotError(
                f"Template identifiers not in input dict {input_dict=} {prompt_template=}"
            )
        return t.substitute(input_dict)

    return f


def sum_usage_stats(usage_stats_list: List[dict]) -> dict:
    return reduce(
        lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
        usage_stats_list,
    )
