import asyncio
import copy
from string import Template
import sys

import uvloop

from .openai_api import (
    OpenAIChatCompletionConfig,
    create_chat_completion_client_session,
    do_chat_completion,
    parse_chat_completion_result,
)


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
    with create_chat_completion_client_session(config) as client_session:
        tasks = [
            asyncio.create_task(do_chat_completion(client_session, config, prompt, system_message))
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
    output_list = [
        copy.copy(input_dict)
        for input_dict in input_list
    ]
    for input_dict, result in zip(output_list, results):
        input_dict[output_key] = parse_chat_completion_result(result)
    return output_list
