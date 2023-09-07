import json
from pathlib import Path
from string import Template
import tiktoken
from typing import Optional, Union

from .util import logger


# https://platform.openai.com/docs/guides/fine-tuning/token-limits
FINE_TUNING_MAX_TOKENS = 4096


def write_openai_fine_tuning_jsonl(
    input_dictlist: list[dict],
    prompt_key: str,
    completion_key: str,
    system_message: Optional[str],
    model: str,
    output_file_prefix: Union[str, Path],
) -> list[Path]:
    jsonl_generator = openai_fine_tuning_jsonl_generator(
        input_dictlist=input_dictlist,
        prompt_key=prompt_key,
        completion_key=completion_key,
        system_message=system_message,
    )
    encoding = tiktoken.encoding_for_model(model)
    output_file_prefix_path = Path(output_file_prefix).resolve()
    output_file_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    current_num_tokens = 0
    file_index = 1
    current_output_file_path = output_file_prefix_path.with_suffix(f".{file_index:0>6}.jsonl")
    current_open_filehandle = current_output_file_path.open("w")
    output_file_paths = []
    for line in jsonl_generator:
        num_tokens = len(encoding.encode(line))
        if num_tokens > FINE_TUNING_MAX_TOKENS:
            logger.warn(f"skipping line with {num_tokens=} {line=}")
            continue
        if current_num_tokens + num_tokens > FINE_TUNING_MAX_TOKENS:
            current_open_filehandle.close()
            output_file_paths.append(current_output_file_path)
            logger.info(f"wrote {current_num_tokens} tokens to {str(current_output_file_path)}")
            file_index += 1
            current_output_file_path = output_file_prefix_path.with_suffix(f"_{file_index:0>6}.jsonl")
            current_open_filehandle = current_output_file_path.open("w")
            current_num_tokens = 0
        current_open_filehandle.write(line)
        current_num_tokens += num_tokens
    current_open_filehandle.close()
    output_file_paths.append(current_output_file_path)
    logger.info(f"wrote {current_num_tokens} tokens to {str(current_output_file_path)}")
    return [
        str(output_file_path)
        for output_file_path in output_file_paths
    ]


def openai_fine_tuning_jsonl_generator(
    input_dictlist: list[dict],
    prompt_key: str,
    completion_key: str,
    system_message: Optional[str] = None,
):
    for input_dict in input_dictlist:
        prompt = input_dict[prompt_key]
        completion = input_dict[completion_key]
        messages = []
        if system_message:
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
        line = json.dumps(messages, separators=(",", ":")) + "\n"
        yield line
