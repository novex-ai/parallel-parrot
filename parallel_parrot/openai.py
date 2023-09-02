try:
    import pandas as pd
except ImportError:
    pd = None


from .openai_api import parallel_openai_chat_completion
from .types import OpenAIChatCompletionConfig
from .util import input_list_to_prompts, append_model_outputs_dictlist, sum_usage_stats


async def parrot_openai_chat_completion_pandas(
    config: OpenAIChatCompletionConfig,
    input_df: "pd.DataFrame",
    prompt_template: str,
    output_key: str,
    system_message: str = None,
):
    if not pd:
        raise ImportError("pandas is not installed. Please install pandas to use this function.")
    prompts = input_list_to_prompts(input_df.to_dict(orient="records"), prompt_template)
    (model_outputs, usage_stats_list) = await parallel_openai_chat_completion(
        config=config,
        prompts=prompts,
        system_message=system_message,
    )
    output_df = input_df.copy()
    output_df[output_key] = model_outputs
    output_df = output_df.astype({output_key: "string"})
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return output_df, usage_stats_sum


async def parrot_openai_chat_completion_dictlist(
    config: OpenAIChatCompletionConfig,
    input_list: list[dict],
    prompt_template: str,
    output_key: str,
    system_message: str = None,
):
    prompts = input_list_to_prompts(input_list, prompt_template)
    (model_outputs, usage_stats_list) = await parallel_openai_chat_completion(
        config=config,
        prompts=prompts,
        system_message=system_message,
    )
    output_list = append_model_outputs_dictlist(input_list, model_outputs, output_key)
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return output_list, usage_stats_sum
