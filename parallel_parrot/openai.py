try:
    import pandas as pd
except ImportError:
    pd = None


from .openai_api import single_openai_chat_completion, parallel_openai_chat_completion
from .types import ParallelParrotError, OpenAIChatCompletionConfig
from .util import input_list_to_prompts, append_model_outputs_dictlist, sum_usage_stats


async def parrot_openai_chat_completion_pandas(
    config: OpenAIChatCompletionConfig,
    input_df: "pd.DataFrame",
    prompt_template: str,
    output_key: str,
    system_message: str = None,
) -> tuple["pd.DataFrame", dict]:
    if not pd:
        raise ParallelParrotError(
            "pandas is not installed. Please install pandas to use this function."
        )
    if input_df.empty:
        raise ParallelParrotError(f"{input_df=} must not be empty")
    prompts = input_list_to_prompts(input_df.to_dict(orient="records"), prompt_template)
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
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
) -> tuple[list[dict], dict]:
    if len(input_list) == 0:
        raise ParallelParrotError(f"{input_list=} must not be empty")
    prompts = input_list_to_prompts(input_list, prompt_template)
    (model_outputs, usage_stats_list) = await _parrot_openai_chat_completion(
        config=config,
        prompts=prompts,
        system_message=system_message,
    )
    output_list = append_model_outputs_dictlist(input_list, model_outputs, output_key)
    usage_stats_sum = sum_usage_stats(usage_stats_list)
    return output_list, usage_stats_sum


async def _parrot_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    prompts: list[str],
    system_message: str,
):
    (model_output, usage_stats, response_headers) = await single_openai_chat_completion(
        config=config,
        prompt=prompts[0],
        system_message=system_message,
    )
    model_outputs = [model_output]
    usage_stats_list = [usage_stats]
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    if len(prompts) >= 2:
        (_model_outputs, _usage_stats_list) = await parallel_openai_chat_completion(
            config=config,
            prompts=prompts[1:],
            system_message=system_message,
            ratelimit_limit_requests=ratelimit_limit_requests,
        )
        model_outputs += _model_outputs
        usage_stats_list += _usage_stats_list
    return (model_outputs, usage_stats_list)
