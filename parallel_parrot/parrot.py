import asyncio
import copy
from string import Template
import sys

import uvloop

from .openai_api import (
    OpenAIChatCompletionConfig,
    parallel_openai_chat_completion,
)


def sync_run(runnable_coro):
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(runnable_coro)
    finally:
        loop.close()



async def parrot_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input_list: list[dict],
    prompt_template: str,
    output_key: str,
    system_message: str = None
):
    t = Template(prompt_template)
    prompts = [
        t.substitute(input_dict) for input_dict in input_list
    ]
    (model_outputs, usage_stats_sum) = await parallel_openai_chat_completion(
        config=config,
        prompts=prompts,
        system_message=system_message,
    )
    output_list = [
        copy.copy(input_dict)
        for input_dict in input_list
    ]
    for input_dict, model_output in zip(output_list, model_outputs):
        input_dict[output_key] = model_output
    return output_list, usage_stats_sum
