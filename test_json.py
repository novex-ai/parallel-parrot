import json
import os
import logging

import parallel_parrot as pp
from parallel_parrot.prompt_templates import JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT


config = pp.OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    (result_dictlist, stats) = pp.sync_run(
        pp.parrot_openai_chat_completion_dictlist(
            config=config,
            input_list=[
                {
                    "input": """
            Predict Customer Lifetime Value From Day 1
Leading e-commerce and subscription brands use Retina AI to predict customer lifetime value and make more profitable decisions.
"""
                },
                {
                    "input": """
            Power your business with Retina
Get fast, reliable CLV
Every deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.

 

Gain a deeper understanding of your customers
Access dashboards and reports that tell you which segments, customers, and attributes are high-performing and which are dragging down your business profitability.

 

Improve customer profitability
Retina’s unique combination of predictive insights and integrations make it easy to identify and act to acquire and retain more high-value customers.

 

Generate more accurate forecasts
Go beyond past customer behavior data and start building forecasts based on accurate predictions of future customer behavior and revenue.
"""
                },
            ],
            prompt_template=JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT,
            output_key="output",
        )
    )
    print("-------------------------------")
    qa_list = pp.auto_explode_json_dictlist(result_dictlist, "output")
    print(json.dumps(qa_list, indent=2))
