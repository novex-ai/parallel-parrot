import os
import asyncio
from parallel_parrot.openai_api import OpenAIChatCompletionConfig
from parallel_parrot.parrot import sync_run, parrot_openai_chat_completion


config = OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
)


if __name__ == '__main__':
    output = sync_run(parrot_openai_chat_completion(
        config=config,
        input_list=[
            {"input": "what is 1+1?"},
            {"input": "what is 2+2?"},
            {"input": "what is 3+3?"},
        ],
        prompt_template="Q: $input\nA:",
        output_key="output",
        system_message="you are a super-precise calculator that only returns answers"
    ))
    print(repr(output))
