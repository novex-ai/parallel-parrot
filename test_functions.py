import json
import os

from aiohttp import ClientSession

from parallel_parrot.async_util import sync_run


openai_api_key=os.environ["OPENAI_API_KEY"]


async def to_request():
    async with ClientSession() as client_session:
        async with client_session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_api_key}",
            },
            json={
                "model": "gpt-3.5-turbo-0613",
                "messages": [
                    {
                        "role": "user",
                        "content": """
Generate question and answer pairs from the following document.
Output a list of JSON objects with keys "question" and "answer".
example:
[{"question": "What is the capital of France?", "answer": "Paris"},{"question": "What is the capital of Spain?", "answer": "Madrid"}]
document: Power your business with Retina
Get fast, reliable CLV
Every deliverable related to 90%+ accurate CLV (scores, analytics, backtest & data explorer) is delivered within hours vs. months or years and at a fraction of the cost.
                        """
                    }
                ],
                "functions": [
                    {
                        "name": "handle_qa_json",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "qa_objects": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question": {
                                                "type": "string",
                                            },
                                            "answer": {
                                                "type": "string",
                                            },
                                        },
                                        "required": ["question", "answer"],
                                    },
                                },
                            },
                        }
                    }
                ],
                "function_call": {
                    "name": "handle_qa_json"
                },
            }
        ) as response:
            print(response.status)
            print(await response.text())
            data = await response.json()
            choices = data.get("choices")
            choice = choices[0]
            message = choice.get("message", {})
            function_call = message.get("function_call")
            arg_obj = json.loads(function_call["arguments"])
            print(json.dumps(arg_obj, indent=2))


if __name__ == "__main__":
    sync_run(to_request())
