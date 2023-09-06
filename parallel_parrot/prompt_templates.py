ECHO_INPUT = "${input}"

QUESTION_FROM_DOCUMENT = """
Come up with a single question which is best answered by the following document: ${input}
"""

JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT = """
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
Only output questions and answers clearly described in the document.  If there are no questions and answers, output an empty list.
document: ${input}
"""
JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT_KEY_NAMES = ["question", "answer"]

JSON_TITLE_AND_SUMMARY_FROM_DOCUMENT = """
Generate a title and summary from the following document.
Output a list of JSON objects with keys "title" and "summary".
Only output summaries that are accurate and reasonably comprehensive.  If there are no good titles and summaries, output an empty list.
document: ${input}
"""
JSON_TITLE_AND_SUMMARY_FROM_DOCUMENT_KEY_NAMES = ["title", "summary"]
