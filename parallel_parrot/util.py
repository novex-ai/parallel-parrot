import copy
from functools import reduce
import logging
from string import Template
import sys
from typing import Optional


logger = logging.getLogger("parallel_parrot")
logger.addHandler(logging.NullHandler())


def input_list_to_prompts(input_list: list[dict], prompt_template: str) -> list[str]:
    if len(input_list) == 0:
        raise ValueError(f"Empty {input_list=}")
    t = Template(prompt_template)
    if sys.version_info >= (3, 11):
        if not t.is_valid():
            raise ValueError(f"Invalid template {prompt_template=}")
        identifiers = t.get_identifiers()
        first_input_dict = input_list[0]
        if set(identifiers) > set(first_input_dict.keys()):
            raise ValueError(
                f"Template identifiers not in input list {first_input_dict=} {prompt_template=}"
            )
    prompts = [t.substitute(input_dict) for input_dict in input_list]
    return prompts


def append_model_outputs_dictlist(
    input_list: list[dict], model_outputs: list[Optional[str]], output_key: str
):
    output_list = [copy.copy(input_dict) for input_dict in input_list]
    for output_dict, model_output in zip(output_list, model_outputs):
        output_dict[output_key] = model_output
    return output_list


def sum_usage_stats(usage_stats_list: list[dict]) -> dict:
    return reduce(
        lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
        usage_stats_list,
    )
