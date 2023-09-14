import json
import logging

import parallel_parrot as pp


logging.basicConfig(level=logging.DEBUG)


input_data = [
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What can Retina help businesses with?",
        "answer": "Retina can help businesses with powering their business and getting fast, reliable Customer Lifetime Value (CLV).",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What is delivered related to CLV?",
        "answer": "Every deliverable related to CLV (scores, analytics, backtest & data explorer) is delivered within hours.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "How long does it take to deliver CLV?",
        "answer": "CLV is delivered within hours, not months or years.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "How much does it cost to deliver CLV?",
        "answer": "The cost of delivering CLV is a fraction of the cost of traditional methods.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What can Retina help with?",
        "answer": "Retina can help with Customer Lifetime Value (CLV)",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "How fast is the delivery of CLV-related deliverables?",
        "answer": "CLV-related deliverables are delivered within hours",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What is the accuracy of CLV?",
        "answer": "CLV is 90%+ accurate",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What is the cost of CLV-related deliverables?",
        "answer": "CLV-related deliverables cost a fraction of the cost",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What can Retina help with?",
        "answer": "Retina can help with fast and reliable Customer Lifetime Value (CLV) calculations.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "How fast can Retina deliver CLV-related deliverables?",
        "answer": "Retina can deliver CLV-related deliverables within hours.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "How accurate are the CLV scores delivered by Retina?",
        "answer": "The CLV scores delivered by Retina are 90%+ accurate.",
    },
    {
        "input": "Power your business with Retina\nGet fast, reliable Customer Lifetime Value (CLV)\nEvery deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.",
        "question": "What is the advantage of using Retina for CLV calculations?",
        "answer": "The advantage of using Retina for CLV calculations is that it is much faster and more cost-effective compared to other methods that can take months or years.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What is Retina?",
        "answer": "Retina is a complete, self-serve solution for predicting, understanding, and acting on future customer behavior.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What can Retina help with?",
        "answer": "Retina can help optimize customer acquisition costs and better understand which high-value customers are at risk.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What is Retina?",
        "answer": "Retina is a complete, self-serve solution for predicting, understanding, and acting on future customer behavior.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What can Retina help with?",
        "answer": "Retina can help optimize customer acquisition costs and better understand which high-value customers are at risk.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What is Retina?",
        "answer": "Retina is a complete, self-serve solution for predicting, understanding, and acting on future customer behavior.",
    },
    {
        "input": "The Predictive Customer Insights Platform\nRetina is your complete, self-serve solution for predicting, understanding, and acting on future customer behavior. Whether you\u2019re trying to optimize customer acquisition costs or better understand which high-value customers are at risk, Retina has you covered.",
        "question": "What can Retina help with?",
        "answer": "Retina can help optimize customer acquisition costs and better understand high-value customers at risk.",
    },
]


async def main():
    paths = pp.write_openai_fine_tuning_jsonl(
        input_data=input_data,
        prompt_key="question",
        completion_key="answer",
        system_message="",
        model="gpt-3.5-turbo-0613",
        output_file_prefix="/tmp/parallel_parrot/test_fine_tuning_jsonl",
    )
    print(json.dumps(paths, indent=2, default=str))


if __name__ == "__main__":
    pp.run_async(main())
