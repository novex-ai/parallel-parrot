import io
import json
from typing import Optional


def write_openai_fine_tuning_jsonl(
    input_dictlist: list[dict],
    prompt_key: str,
    completion_key: str,
    system_message: Optional[str],
    output_file: io.TextIOWrapper
):
    if not isinstance(output_file, io.TextIOWrapper):
        raise ValueError(f"{output_file=} must be a io.TextIOWrapper")
    for input_dict in input_dictlist:
        prompt = input_dict[prompt_key]
        completion = input_dict[completion_key]
        messages = []
        if system_message is not None:
            messages.append({
                "role": "system",
                "content": system_message,
            })
        messages.append({
            "role": "user",
            "content": prompt,
        })
        messages.append({
            "role": "assistant",
            "content": completion,
        })
        json.dump(messages, output_file)
        output_file.write("\n")

