import json
import os
import logging

import parallel_parrot as pp
from parallel_parrot.async_util import sync_run

config = pp.OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-0613",
    n=3,
)

logging.basicConfig(level=logging.DEBUG)


async def main():
    result = await pp.parrot_openai_chat_completion_exploding_function_dictlist(
        config=config,
        input_list=[
            {
                "input": """
Power your business with Retina
Get fast, reliable Customer Lifetime Value (CLV)
Every deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.
                """.strip()
            },
            {
                "input": """
The Predictive Customer Insights Platform
Retina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you’re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.
                """.strip()
            }
        ],
        prompt_template="""
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.  If there are no questions and answers, output an empty list.
document: ${input}
        """,
        output_key_names=["question", "answer"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    sync_run(main())
