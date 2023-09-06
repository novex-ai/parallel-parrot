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
