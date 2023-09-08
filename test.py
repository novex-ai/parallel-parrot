import os
import logging

import pandas as pd  # type: ignore
import parallel_parrot as pp


config = pp.OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-0613",
)

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    input_df = pd.DataFrame(
        {
            "input": [
                "this is a super duper product that will change the world",
                "do not buy this",
            ],
            "source": [
                "amazon",
                "shopify",
            ],
        },
        index=[100, 101],
    )
    (output_df, usage_stats_sum) = pp.sync_run(
        pp.parrot_openai_chat_completion_pandas(
            config=config,
            input_df=input_df,
            prompt_template="""
What is the sentiment of this product review?
POSITIVE or NEGATIVE?
product review: ${input}
sentiment:""",
            output_key="sentiment",
        )
    )
    print(repr(output_df["sentiment"]))
    print(repr(usage_stats_sum))
