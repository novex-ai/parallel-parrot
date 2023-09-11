# parallel-parrot

A Python library for easily and efficiently using LLMs on tabular data.

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

Some use cases:
- generate questions from documents for better Retrieval Augmented Generation (match questions to questions, not documents)
- sentiment analysis on a large number of documents
- data extraction and summarization
- removal of personal identifiers
- generate instructions from documents for fine tuning of LLMs

Features:
- Operates on both pandas dataframes and native python lists of dictionaries
- Supports OpenAI Chat Completion API, with structured output ("functions") - more LLMs planned in the future
- Automatically determines your rate limit from the API and executes API requests in parallel, as fast as possible
- Automatic retries, with exponential backoff and jitter
- Uses simple/robust Python [string.Template](https://docs.python.org/3/library/string.html#string.Template) for prompt templates
- create fine-tuning jsonl files from data that also avoids truncation due to token limits
- tracking of API usage statistics, to support cost governance
- fast asynchronous (concurrent) requests using aiohttp and uvloop

## Getting Started

```python
pip install parallel-parrot
```

Define an API configuration object:
```python
import parallel_parrot as pp

config = pp.OpenAIChatCompletionConfig(
    openai_api_key="*your API key*"
)
```

see the [declaration](./parallel_parrot/types.py) of `OpenAIChatCompletionConfig` for more available parameters, including the `system_message`.  All parameters that can be passed to the OpenAI API are available.

## Generate Text - pp.parallel_text_generation()

LLM text generation can be used for a wide variety of tasks:
- Sentiment analysis: for understanding large amounts of customer input.
- Summarization: for making a large number of documents easier to digest/use.
- Text transformation: such as the removal of PII from text

see the [prompt_templates](./parallel_parrot/prompt_templates.py) for some pre-engineered templates.

Example of `pp.parallel_text_generation()`:
```python

input_data = [
    {
        "input": "this is a super duper product that will change the world",
        "source": "shopify",
    },
    {
        "input": "this is a horrible product that does not work",
        "source": "amazon"
    }
]

(output, usage_stats) = pp.sync_run(
    pp.parallel_text_generation(
        config=config,
        input_data=input_data,
        prompt_template="""
What is the sentiment of this product review?
POSITIVE, NEUTRAL or NEGATIVE?
product review: ${input}
sentiment:""",
        output_key="sentiment",
    )
)
print(f"{output=}")
```

## Generate Data - pp.parallel_data_generation()

Some use-cases are more demanding than the above, and require multiple structured outputs.  This package handles all of that API and data wrangling for you.

Some examples of these use cases include:
- generating multiple question/answer pairs from each input document
- generating multiple title/summary paris from each input document

see the [prompt_templates](./parallel_parrot/prompt_templates.py) for some pre-engineered templates.

Example of `pp.parallel_data_generation()`:
```python

input_data = [
    {
        "input": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
        """
    },
    {
        "input": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
        """
    },
]

(output, usage_stats) = pp.sync_run(
    pp.parallel_data_generation(
        config=config,
        input_data=input_data,
        prompt_template="""
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.  If there are no questions and answers, output an empty list.
document: ${input}
        """,
        output_key_names: ["question", "answer"]
    )
)
print(f"{output=}")
```
