try:
    import pandas as pd  # type: ignore
except ImportError:
    pandas_installed = False
else:
    pandas_installed = True

from typing import List, Optional, Union

from .openai_api import (
    single_setup_openai_chat_completion,
    parallel_openai_chat_completion,
)
from .types import ParallelParrotError, ParallelParrotOutput, OpenAIChatCompletionConfig
from .util import (
    logger,
    sum_usage_stats,
)
from .util_template import (
    make_curried_prompt_template,
)
from .util_dictlist import (
    append_model_outputs_dictlist,
    append_one_to_many_model_outputs_dictlist,
    append_one_to_many_objlist_outputs_dictlist,
)
from .util_pandas import (
    append_model_outputs_pandas,
    append_one_to_many_model_outputs_pandas,
    append_one_to_many_objlist_outputs_pandas,
)


async def parallel_openai_chat_completion_dictlist(
    config: OpenAIChatCompletionConfig,
    input_list: List[dict],
    prompt_template: str,
    output_key: str,
) -> ParallelParrotOutput:
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        input=input_list,
        prompt_template=prompt_template,
    )
    if config.n is not None and config.n > 1:
        output_list = append_one_to_many_model_outputs_dictlist(
            input_list, model_outputs, output_key
        )
        input_num_rows = len(input_list)
        output_num_rows = len(output_list)
        logger.info(
            "Output may have more rows than input"
            f" because {config.n=} is greater than 1."
            f" {input_num_rows=} {output_num_rows=}"
        )
    else:
        output_list = append_model_outputs_dictlist(
            input_list, model_outputs, output_key
        )
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return ParallelParrotOutput(output=output_list, usage_stats=usage_stats_sum)


async def parallel_openai_chat_completion_pandas(
    config: OpenAIChatCompletionConfig,
    input_df: "pd.DataFrame",
    prompt_template: str,
    output_key: str,
) -> ParallelParrotOutput:
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        input=input_df,
        prompt_template=prompt_template,
    )
    if config.n is not None and config.n > 1:
        output_df = append_one_to_many_model_outputs_pandas(
            input_df, model_outputs, output_key
        )
        input_num_rows = len(input_df)
        output_num_rows = len(output_df)
        logger.info(
            "Output may have more rows than input"
            f" because {config.n=} is greater than 1."
            f" {input_num_rows=} {output_num_rows=}"
        )
    else:
        output_df = append_model_outputs_pandas(input_df, model_outputs, output_key)
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return ParallelParrotOutput(output=output_df, usage_stats=usage_stats_sum)


async def parallel_openai_chat_completion_exploding_function_dictlist(
    config: OpenAIChatCompletionConfig,
    input_list: List[dict],
    prompt_template: str,
    output_key_names: List[str],
) -> ParallelParrotOutput:
    """
    Process a prompt which generates a list of objects.
    Explode those outputs into multiple rows with the object keys as column names
    """
    (functions, function_call) = _prep_function_list_of_objects(
        function_name="f",
        parameter_name="p",
        output_key_names=output_key_names,
    )
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        input=input_list,
        prompt_template=prompt_template,
        functions=functions,
        function_call=function_call,
    )
    output_list = append_one_to_many_objlist_outputs_dictlist(
        input_list, model_outputs, output_key_names
    )
    input_num_rows = len(input_list)
    output_num_rows = len(output_list)
    logger.info(
        "Output may have more rows than input because we are asking for a list of objects."
        f" {input_num_rows=} {output_num_rows=}"
    )
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return ParallelParrotOutput(output=output_list, usage_stats=usage_stats_sum)


async def parallel_openai_chat_completion_exploding_function_pandas(
    config: OpenAIChatCompletionConfig,
    input_df: "pd.DataFrame",
    prompt_template: str,
    output_key_names: List[str],
) -> ParallelParrotOutput:
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    (functions, function_call) = _prep_function_list_of_objects(
        function_name="f",
        parameter_name="p",
        output_key_names=output_key_names,
    )
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        input=input_df,
        prompt_template=prompt_template,
        functions=functions,
        function_call=function_call,
    )
    output_df = append_one_to_many_objlist_outputs_pandas(
        input_df, model_outputs, output_key_names
    )
    input_num_rows = len(input_df)
    output_num_rows = len(output_df)
    logger.info(
        "Output may have more rows than input because we are asking for a list of objects. Note that the index is also reset."
        f" {input_num_rows=} {output_num_rows=}"
    )
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return ParallelParrotOutput(output=output_df, usage_stats=usage_stats_sum)


async def _parrot_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input: Union[List[dict], "pd.DataFrame"],
    prompt_template: str,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> ParallelParrotOutput:
    curried_prompt_template = make_curried_prompt_template(prompt_template)
    # process a single row first, both to check for errors and to get the ratelimit_limit_requests
    if isinstance(input, list):
        first_row = input[0]
    elif isinstance(input, pd.DataFrame):
        first_row = input.iloc[0]
    else:
        raise ParallelParrotError(f"Unexpected type {type(input)=}")
    (
        model_output,
        usage_stats,
        response_headers,
    ) = await single_setup_openai_chat_completion(
        config=config,
        input_row=first_row,
        curried_prompt_template=curried_prompt_template,
        functions=functions,
        function_call=function_call,
    )
    model_outputs = [model_output]
    usage_stats_list = [usage_stats]
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    if len(input) >= 2:
        if isinstance(input, list):
            nonfirst_rows = input[1:]
        elif isinstance(input, pd.DataFrame):
            nonfirst_rows = input.iloc[1:, :]
        (_model_outputs, _usage_stats_list) = await parallel_openai_chat_completion(
            config=config,
            input_table=nonfirst_rows,
            curried_prompt_template=curried_prompt_template,
            functions=functions,
            function_call=function_call,
            ratelimit_limit_requests=ratelimit_limit_requests,
        )
        model_outputs += _model_outputs
        usage_stats_list += _usage_stats_list
    return ParallelParrotOutput(output=model_outputs, usage_stats=usage_stats_list)


def _prep_function_list_of_objects(
    function_name: str, parameter_name: str, output_key_names: List[str]
):
    if len(output_key_names) == 0:
        raise ParallelParrotError(f"{output_key_names=} must not be empty")
    output_item_json_schema_properties = {
        key: {
            "type": "string",
        }
        for key in output_key_names
    }
    parameter_json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": output_item_json_schema_properties,
            "required": output_key_names,
        },
    }
    function_json_schema = {
        "name": function_name,
        "parameters": {
            "type": "object",
            "properties": {
                parameter_name: parameter_json_schema,
            },
        },
    }
    functions = [function_json_schema]
    function_call = {
        "name": function_name,
    }
    return (functions, function_call)
