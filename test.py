import os

from parallel_parrot import sync_run
from parallel_parrot.openai_api import OpenAIChatCompletionConfig, parrot_openai_chat_completion_dictlist


config = OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
)


if __name__ == '__main__':
    output = sync_run(parrot_openai_chat_completion_dictlist(
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
