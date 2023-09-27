# parallel-parrot

A Python library for easily and quickly using LLMs on tabular data.  Because synchronous for-loops are too slow, and parallelism can be a pain.

[![PyPI version](https://badge.fury.io/py/parallel-parrot.svg)](https://badge.fury.io/py/parallel-parrot)
[![Release Notes](https://img.shields.io/github/release/novex-ai/parallel-parrot)](https://github.com/novex-ai/parallel-parrot/releases)
[![pytest](https://github.com/novex-ai/parallel-parrot/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/novex-ai/parallel-parrot/actions/workflows/pytest.yml)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

![flock of parrots](https://res.cloudinary.com/dn7sohze7/image/upload/v1695421900/parallel-parrot/0002-gareth-davies-EGcfyDiUv58-unsplash.jpg)

*Photo by [Gareth Davies](https://unsplash.com/@gdfoto?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)
 on [Unsplash](https://unsplash.com/photos/EGcfyDiUv58?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)* 

See [blog post](https://bradito.me/blog/parallel-parrot/) on why we built this.

Use cases:
- Generate questions from documents for better Retrieval Augmented Generation (match questions to questions, not documents)
- Sentiment analysis or summarization on a large number of documents
- Data extraction and summarization
- Removal of personal identifiers
- Generate instructions from documents for fine tuning of LLMs

Main Features:
- Supports both pandas dataframes and native python lists of dictionaries
- Supports OpenAI Chat Completion API, with structured output "functions" (more LLMs planned in the future)
- Output formatted data for fine-tuning

Other Features:
- Fast asynchronous (parallel) requests using aiohttp and uvloop, with support for notebook environments that have an existing event loop
- Python logging support (e.g. `logging.basicConfig(level=logging.DEBUG)`)
- Automatic retries, with exponential backoff, jitter, and dynamic header-based delays
- Flexible prompt templates using standard Python [string.Template](https://docs.python.org/3/library/string.html#string.Template).  e.g. `"summarize: ${input}"`
- "Batteries included" with pre-engineered prompt templates
- Programmatic tracking of token usage to support cost control measures.
- Supports `pandas` 1.x and 2.x APIs


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

see the [declaration](https://github.com/novex-ai/parallel-parrot/blob/v0.3.2/parallel_parrot/types.py#L27) of `OpenAIChatCompletionConfig` for more available parameters, including the `system_message`.
All [Open API parameters](https://platform.openai.com/docs/api-reference/chat/create) can be passed.  Note that only models supported by the [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/gpt/gpt-models) can be used with this configuration.

## Generate Text - pp.parallel_text_generation()

This function executes parallel text generation/completion using a LLM.

It does so by:
- Taking in a dataframe or list of dictionaries.
- Applying the python prompt template to each row.  Column names are used as the variable names in the template.
- Calling the LLM API with the prompt for each row.  Runs a single request first, for two reasons:
  - Test access to the API, including credentials, without retries or complicated calling mechanics.
  - Uses that request to automatically obtain [rate limit information](https://platform.openai.com/docs/guides/rate-limits) from the OpenAI API to configure the parallel requests to run with maximum concurrency.
- Appending the output to the input dataframe or list of dictionaries using the output_key.
- Input values are passed through to the outputs to permit custom logic.

Example of `pp.parallel_text_generation()`:
```python
import json
import parallel_parrot as pp


config = pp.OpenAIChatCompletionConfig(
    openai_api_key="*your API key*",
    system_message="you are a very precise assistant",
)


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

(output, usage_stats) = pp.run_async(
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

print(json.dumps(output, indent=2))
print(json.dumps(usage_stats, indent=2))
```

example output:
```json
[
    {
        "input": "this is a super duper product that will change the world",
        "source": "shopify",
        "sentiment": "POSITIVE",
    },
    {
        "input": "this is a horrible product that does not work",
        "source": "amazon",
        "sentiment": "NEGATIVE",
    }
]
```

- If the LLM generates multiple outputs (n > 1 for OpenAI), outputs are deduped, then exploded.  Outputs may then contain more rows than the input.
- If no output is generated, then `None` (for lists of dictionaries) or `math.nan` (for pandas) is returned.
- See the [prompt_templates](https://github.com/novex-ai/parallel-parrot/blob/v0.3.2/parallel_parrot/prompt_templates.py) for some pre-engineered templates.

## Generate Data - pp.parallel_data_generation()

Some use-cases are more demanding than the above, and require more complicated outputs.
This function supports prompts which expect to generate lists of dictionaries.

Some examples of these use cases include:
- Generating multiple question/answer pairs from each input document
- Generating multiple title/summary pairs from each input document

It does so by:
- Taking in a pandas dataframe or list of dictionaries
- Applying the python prompt template to each row.  Column names are used as the variable names in the template.
- Generating an API call specifying that we want a list of objects,
  with each object containing values for each of the output_key_names. (OpenAI "functions")
- Calling the LLM API with the prompt for each row
- Parsing the returned JSON data into a list of dictionaries
- Mapping each returned dictionary to a row in the output dataframe or list of dictionaries.  This will result in "exploded" output, where
  the output will contain more than one row for a given input.

Example of `pp.parallel_data_generation()`:
```python
import json
import parallel_parrot as pp

config = pp.OpenAIChatCompletionConfig(
    openai_api_key="*your API key*",
    n=3,  # tip: to generate many creative outputs, it often makes sense to use n > 1
)

input_data = [
    {
        "text": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
        """,
        "source_url": "https://en.wikipedia.org/wiki/George_Washington",
    },
    {
        "text": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
        """,
        "source_url": "https://en.wikipedia.org/wiki/John_Adams",
    },
]

(output, usage_stats) = pp.run_async(
    pp.parallel_data_generation(
        config=config,
        input_data=input_data,
        prompt_template="""
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.
If there are no questions and answers, output an empty list.
document: ${text}
        """,
        output_key_names=["question", "answer"]
    )
)

print(json.dumps(output, indent=2))
print(json.dumps(usage_stats, indent=2))
```

example output:
```json
[
  {
    "text": "...",
    "source_url": "https://en.wikipedia.org/wiki/George_Washington",
    "question": "Who was the first president of the United States?",
    "answer": "George Washington"
  },
  {
    "text": "...",
    "source_url": "https://en.wikipedia.org/wiki/George_Washington",
    "question": "What position did George Washington hold during the American Revolutionary War?",
    "answer": "Commander of the Continental Army"
  },
  {
    "text": "...",
    "source_url": "https://en.wikipedia.org/wiki/George_Washington",
    "question": "What document did George Washington help draft and ratify?",
    "answer": "The Constitution of the United States"
  },
  // ...
  {
    "text": "...",
    "source_url": "https://en.wikipedia.org/wiki/John_Adams",
    "question": "Who were some important contemporaries that John Adams corresponded with?",
    "answer": "Adams regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson."
  },
  {
    "text": "...",
    "source_url": "https://en.wikipedia.org/wiki/John_Adams",
    "question": "Who was John Adams?",
    "answer": "John Adams was an American statesman, attorney, diplomat, writer, and Founding Father."
  },
]
```

Notice that multiple output rows are created for each input, based on what the LLM returns.  All input columns/keys are retained, to permit integration (joining) with other code.

If more than one LLM continuation/response is generated per prompt (e.g. `n` > 1 for OpenAI), then these
outputs are put into additional rows.

If no output is generated (an empty list, or an empty string, or malformed JSON), then `None` (for lists of dictionaries) or `math.nan` (for pandas dataframes) is returned for each key in `output_key_names`.

## Prepare Fine-Tuning Data for OpenAI - pp.write_openai_fine_tuning_jsonl()

If you need to do [OpenAI Fine Tuning](https://platform.openai.com/docs/guides/fine-tuning) - but find it a pain to
split your data at appropriate token counts in jsonl format, the `parrallel-parrot` can help with this as well.

```python
import json
import parallel_parrot as pp

input_data = [
  {
    "question": "Who was the first president of the United States?",
    "answer": "George Washington"
  },
  {
    "question": "What position did George Washington hold during the American Revolutionary War?",
    "answer": "Commander of the Continental Army"
  },
  {
    "question": "What document did George Washington help draft and ratify?",
    "answer": "The Constitution of the United States"
  },
]

paths = pp.write_openai_fine_tuning_jsonl(
    input_data=input_data,
    prompt_key="question",
    completion_key="answer",
    system_message="",
    model="gpt-3.5-turbo-0613",  # used to calculate token counts
    output_file_prefix="/tmp/parallel_parrot/test_fine_tuning",
)
print(json.dumps(paths, indent=2, default=str))
```

This will create files that can be sent directly to the [OpenAI Fine Tuning API](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset).  Doing so with this example will result in an LLM which knows more than the average parrot about presidents of the USA.

example output paths:
```
/tmp/parallel_parrot/test_fine_tuning.00001.jsonl
/tmp/parallel_parrot/test_fine_tuning.00002.jsonl
```

---

_Note on the name of the package: It's an alliterative animal name that combines the main functionality: parallelism, with the animal that can sort-of talk: parrots (like LLMs)_
