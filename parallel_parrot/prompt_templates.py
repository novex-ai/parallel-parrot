ECHO_INPUT = "${input}"

QUESTION_FROM_DOCUMENT = """
Come up with a single question which is best answered by the following document: ${input}
"""

JSON_QUESTION_AND_ANSWER_FROM_DOCUMENT = """
Generate question and answer pairs from the following document.
Encode each pair as a list of JSON object with keys "question" and "answer".
example:
{"question": "What is the capital of France?", "answer": "Paris"}
document: ${input}
"""
