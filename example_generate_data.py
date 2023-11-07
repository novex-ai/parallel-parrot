import json
import os
import logging

import parallel_parrot as pp
from parallel_parrot import run_async
from parallel_parrot.prompt_templates import (
    JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT,
    JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT_KEY_NAMES,
)

config = pp.OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-1106",
    n=2,
)

logging.basicConfig(level=logging.DEBUG)


async def main():
    (output, usage_stats) = await pp.parallel_data_generation(
        config=config,
        input_data=[
            {
                "input": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
                """,
                "source_url": "https://en.wikipedia.org/wiki/George_Washington",
            },
            {
                "input": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
                """,
                "source_url": "https://en.wikipedia.org/wiki/John_Adams",
            },
        ],
        prompt_template=JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT,
        output_key_names=JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT_KEY_NAMES,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_async(main())
