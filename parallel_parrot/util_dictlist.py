import copy
import json
from typing import List, Optional


def append_model_outputs_dictlist(
    input_list: List[dict], model_outputs: List[Optional[str]], output_key: str
) -> List:
    output_list = [copy.copy(input_dict) for input_dict in input_list]
    for output_dict, model_output in zip(output_list, model_outputs):
        output_dict[output_key] = model_output
    return output_list


def append_one_to_many_model_outputs_dictlist(
    input_list: List[dict], model_outputs: List[List[Optional[str]]], output_key: str
) -> List[dict]:
    output_list = []
    for input_dict, model_output in zip(input_list, model_outputs):
        if isinstance(model_output, list) and len(model_output) > 0:
            for model_output_item in model_output:
                output_dict = copy.copy(input_dict)
                output_dict[output_key] = model_output_item
                output_list.append(output_dict)
        else:
            output_dict = copy.copy(input_dict)
            output_dict[output_key] = None
            output_list.append(output_dict)
    return output_list


def append_one_to_many_objlist_outputs_dictlist(
    input_list: List[dict],
    objlist_outputs: List[List[dict]],
    output_key_names: List[str],
) -> List[dict]:
    output_list = []
    for input_dict, objlist_output in zip(input_list, objlist_outputs):
        if isinstance(objlist_output, list) and len(objlist_output) > 0:
            for obj in objlist_output:
                output_dict = copy.copy(input_dict)
                for key in output_key_names:
                    output_dict[key] = obj[key]
                output_list.append(output_dict)
        else:
            output_dict = copy.copy(input_dict)
            for key in output_key_names:
                output_dict[key] = None
            output_list.append(output_dict)
    return output_list


def auto_explode_json_dictlist(
    data_dictlist: List[dict], key: str, delete_source_data: bool = True
) -> List[dict]:
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
            output_dict = copy.copy(data_dict)
            if delete_source_data:
                del output_dict[key]
            output_list.append(output_dict)
    return output_list
