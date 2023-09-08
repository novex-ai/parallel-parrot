import copy
import json
from functools import reduce
import logging
from string import Template
import sys
from typing import Optional

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


def append_model_outputs_dictlist(
    input_list: list[dict], model_outputs: list[Optional[str]], output_key: str
):
    output_list = [copy.copy(input_dict) for input_dict in input_list]
    for output_dict, model_output in zip(output_list, model_outputs):
        output_dict[output_key] = model_output
    return output_list


def append_one_to_many_model_outputs_dictlist(
    input_list: list[dict], model_outputs: list[list[Optional[str]]], output_key: str
):
    output_list = []
    for input_dict, model_output in zip(input_list, model_outputs):
        for model_output_item in model_output:
            output_dict = copy.copy(input_dict)
            output_dict[output_key] = model_output_item
            output_list.append(output_dict)
    return output_list


def append_one_to_many_objlist_outputs_dictlist(
    input_list: list[dict],
    objlist_outputs: list[list[dict]],
    output_key_names: list[str],
):
    output_list = []
    for input_dict, objlist_output in zip(input_list, objlist_outputs):
        for obj in objlist_output:
            output_dict = copy.copy(input_dict)
            for key in output_key_names:
                output_dict[key] = obj[key]
            output_list.append(output_dict)
    return output_list


def auto_explode_json_dictlist(
    data_dictlist: list[dict], key: str, delete_source_data: bool = True
) -> list[dict]:
    """
    If the value of a key is a list, explode the list into multiple rows
    """
    output_list = []
    for data_dict in data_dictlist:
        value = data_dict.get(key)
        appended_data = False
        if isinstance(value, str):
            try:
                row_dictlist = json.loads(value)
                if isinstance(row_dictlist, list):
                    for row_dict in row_dictlist:
                        output_dict = copy.copy(data_dict)
                        output_dict.update(row_dict)
                        if delete_source_data:
                            del output_dict[key]
                        output_list.append(output_dict)
                    appended_data = True
            except Exception:
                pass
        if not appended_data:
            output_list.append(data_dict)
    return output_list


def sum_usage_stats(usage_stats_list: list[dict]) -> dict:
    return reduce(
        lambda x, y: {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)},
        usage_stats_list,
    )
