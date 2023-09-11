import json
import os
import logging

import parallel_parrot as pp
from parallel_parrot import sync_run

config = pp.OpenAIChatCompletionConfig(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-0613",
    n=2,
)

logging.basicConfig(level=logging.DEBUG)


async def main():
    result = await pp.parrot_openai_chat_completion_exploding_function_dictlist(
        config=config,
        input_list=[
            {
                "input": """
George Washington (February 22, 1732 - December 14, 1799) was an American military officer, statesman, and Founding Father who served as the first president of the United States from 1789 to 1797. Appointed by the Second Continental Congress as commander of the Continental Army in June 1775, Washington led Patriot forces to victory in the American Revolutionary War and then served as president of the Constitutional Convention in 1787, which drafted and ratified the Constitution of the United States and established the American federal government. Washington has thus been called the "Father of his Country".
                """.strip()
            },
            {
                "input": """
John Adams (October 30, 1735 - July 4, 1826) was an American statesman, attorney, diplomat, writer, and Founding Father who served as the second president of the United States from 1797 to 1801. Before his presidency, he was a leader of the American Revolution that achieved independence from Great Britain. During the latter part of the Revolutionary War and in the early years of the new nation, he served the U.S. government as a senior diplomat in Europe. Adams was the first person to hold the office of vice president of the United States, serving from 1789 to 1797. He was a dedicated diarist and regularly corresponded with important contemporaries, including his wife and adviser Abigail Adams and his friend and political rival Thomas Jefferson.
               """.strip()
            },
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
