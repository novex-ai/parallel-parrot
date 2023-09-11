try:
    import pandas as pd  # type: ignore
except ImportError:
    pandas_installed = False
else:
    pandas_installed = True

from typing import Optional, Union

from .openai_api import single_openai_chat_completion, parallel_openai_chat_completion
from .types import ParallelParrotError, ParallelParrotOutput, OpenAIChatCompletionConfig
from .util import (
    logger,
    input_list_to_prompts,
    sum_usage_stats,
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
    input_list: list[dict],
    prompt_template: str,
    output_key: str,
) -> ParallelParrotOutput:
    prompts = input_list_to_prompts(input_list, prompt_template)
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        prompts=prompts,
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
    prompts = input_list_to_prompts(input_df.to_dict(orient="records"), prompt_template)
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        prompts=prompts,
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
    input_list: list[dict],
    prompt_template: str,
    output_key_names: list[str],
) -> ParallelParrotOutput:
    """
    Process a prompt which generates a list of objects.
    Explode those outputs into multiple rows with the object keys as column names
    """
    prompts = input_list_to_prompts(input_list, prompt_template)
    (functions, function_call) = _prep_function_list_of_objects(
        function_name="f",
        parameter_name="p",
        output_key_names=output_key_names,
    )
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        prompts=prompts,
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
    output_key_names: list[str],
) -> ParallelParrotOutput:
    if not pandas_installed:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    prompts = input_list_to_prompts(input_df.to_dict(orient="records"), prompt_template)
    (functions, function_call) = _prep_function_list_of_objects(
        function_name="f",
        parameter_name="p",
        output_key_names=output_key_names,
    )
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        prompts=prompts,
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
    prompts: list[str],
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> ParallelParrotOutput:
    # process a single row first, both to check for errors and to get the ratelimit_limit_requests
    (model_output, usage_stats, response_headers) = await single_openai_chat_completion(
        config=config,
        prompt=prompts[0],
        functions=functions,
        function_call=function_call,
    )
    model_outputs = [model_output]
    usage_stats_list = [usage_stats]
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    if len(prompts) >= 2:
        (_model_outputs, _usage_stats_list) = await parallel_openai_chat_completion(
            config=config,
            prompts=prompts[1:],
            functions=functions,
            function_call=function_call,
            ratelimit_limit_requests=ratelimit_limit_requests,
        )
        model_outputs += _model_outputs
        usage_stats_list += _usage_stats_list
    return ParallelParrotOutput(output=model_outputs, usage_stats=usage_stats_list)


def _prep_function_list_of_objects(
    function_name: str, parameter_name: str, output_key_names: list[str]
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
