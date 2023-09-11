from functools import reduce
import logging
from string import Template
import sys

from .types import ParallelParrotError


logger = logging.getLogger(__name__.split(".")[0])
logger.addHandler(logging.NullHandler())


def input_list_to_prompts(input_list: list[dict], prompt_template: str) -> list[str]:
    if len(input_list) == 0:
        raise ParallelParrotError(
            f"Input data must not be empty: input={repr(input_list)}"
        )
    prompt_template = prompt_template.strip()
    t = Template(prompt_template)
    if sys.version_info >= (3, 11):
        if not t.is_valid():
            raise ParallelParrotError(f"Invalid template {prompt_template=}")
        identifiers = t.get_identifiers()
        first_input_dict = input_list[0]
        if set(identifiers) > set(first_input_dict.keys()):
            raise ParallelParrotError(
                f"Template identifiers not in input list {first_input_dict=} {prompt_template=}"
            )
    prompts = [t.substitute(input_dict) for input_dict in input_list]
    return prompts


def sum_usage_stats(usage_stats_list: list[dict]) -> dict:
    return reduce(
        lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
        usage_stats_list,
    )
