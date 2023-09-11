try:
    import pandas as pd  # type: ignore
except ImportError:
    pass


from typing import Union

from .openai import (
    parallel_openai_chat_completion_dictlist,
    parallel_openai_chat_completion_pandas,
    parallel_openai_chat_completion_exploding_function_dictlist,
    parallel_openai_chat_completion_exploding_function_pandas,
)
from .types import LLMConfig, OpenAIChatCompletionConfig
from .util_pandas import is_pandas_dataframe


async def parallel_text_generation(
    config: LLMConfig,
    input_data: Union[list[dict], "pd.DataFrame"],
    prompt_template: str,
    output_key: str,
):
    """
    This function executes text generation/completion using a LLM.

    It does so by:
    - taking in a dataframe or list of dictionaries
    - applying the python prompt template to each row.  Column names are used as the variable names in the template.
    - calling the LLM API with the prompt for each row
    - appending the output to the input dataframe or list of dictionaries using the output_key

    Note:
    - If the LLM generates multiple outputs (n > 1 for OpenAI), the output may have more rows than the input.
    - If no output is generated, then None of math.nan is returned.
    """
    if not isinstance(config, OpenAIChatCompletionConfig):
        raise Exception("Only OpenAIChatCompletionConfig is supported for now")
    if isinstance(input_data, list):
        return await parallel_openai_chat_completion_dictlist(
            config=config,
            input_list=input_data,
            prompt_template=prompt_template,
            output_key=output_key,
        )
    elif is_pandas_dataframe(input_data):
        return await parallel_openai_chat_completion_pandas(
            config=config,
            input_df=input_data,
            prompt_template=prompt_template,
            output_key=output_key,
        )
    else:
        raise Exception(
            "Only lists of dictionaries and pd.DataFrame are supported for now"
        )


async def parallel_data_generation(
    config: LLMConfig,
    input_data: Union[list[dict], "pd.DataFrame"],
    prompt_template: str,
    output_key_names: list[str],
):
    """
    This function uses an LLM to generate structured data.

    It does so by:
    - taking in a dataframe or list of dictionaries
    - applying the python prompt template to each row.  Column names are used as the variable names in the template.
    - generating a modified prompt / API call to specify that we want a list of objects,
      with each object containing values for each of the output_key_names.
    - calling the LLM API with the prompt for each row
    - parsing the returned JSON data into a list of dictionaries
    - mapping each returned dictionary to a row in the output dataframe or list of dictionaries

    Note:
    - If no output is generated, then None of math.nan is returned.
    """
    if not isinstance(config, OpenAIChatCompletionConfig):
        raise Exception("Only OpenAIChatCompletionConfig is supported for now")
    if isinstance(input_data, list):
        return await parallel_openai_chat_completion_exploding_function_dictlist(
            config=config,
            input_list=input_data,
            prompt_template=prompt_template,
            output_key_names=output_key_names,
        )
    elif is_pandas_dataframe(input_data):
        return await parallel_openai_chat_completion_exploding_function_pandas(
            config=config,
            input_df=input_data,
            prompt_template=prompt_template,
            output_key_names=output_key_names,
        )
    else:
        raise Exception(
            "Only lists of dictionaries and pd.DataFrame are supported for now"
        )
