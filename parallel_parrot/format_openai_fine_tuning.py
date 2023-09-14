try:
    import pandas as pd  # type: ignore
except ImportError:
    pandas_installed = False
else:
    pandas_installed = True

import json
from pathlib import Path
import tiktoken
from typing import List, Optional, Union

from .util import logger
from .util_pandas import is_pandas_dataframe, pandas_row_reader


# https://platform.openai.com/docs/guides/fine-tuning/token-limits
FINE_TUNING_MAX_TOKENS = 4096


def write_openai_fine_tuning_jsonl(
    input_data: Union[List[dict], "pd.DataFrame"],
    prompt_key: str,
    completion_key: str,
    system_message: Optional[str],
    model: str,
    output_file_prefix: Union[str, Path],
) -> List[str]:
    """
    Take a list of dictionaries or a pandas dataframe and generate JSONL data which
    can be used for instruction fine-tuning an OpenAI model.
    - input_data: a list of dictionaries or a pandas dataframe
    - prompt_key: the key in the dictionary or dataframe which contains the prompt
    - completion_key: the key in the dictionary or dataframe which contains the completion
    - system_message: an optional message to be sent to the assistant before the prompt
    - model: the model to use for token counting
    - output_file_prefix: the prefix for the output file(s) to write to, e.g. "/tmp/fine_tuning"
    """
    jsonl_generator = openai_fine_tuning_jsonl_generator(
        input_data=input_data,
        prompt_key=prompt_key,
        completion_key=completion_key,
        system_message=system_message,
        model=model,
    )
    output_file_prefix_path = Path(output_file_prefix).resolve()
    output_file_prefix_path.parent.mkdir(parents=True, exist_ok=True)
    current_num_tokens = 0
    total_billable_num_tokens = 0
    file_index = 1
    current_output_file_path = _make_jsonl_path(output_file_prefix_path, file_index)
    current_open_filehandle = current_output_file_path.open("w")
    output_file_paths = []
    try:
        for line, num_tokens in jsonl_generator:
            if num_tokens > FINE_TUNING_MAX_TOKENS:
                logger.warn(
                    f"example line too long, will be truncated {num_tokens=} {line=}"
                )
                num_tokens = FINE_TUNING_MAX_TOKENS
            if current_num_tokens + num_tokens > FINE_TUNING_MAX_TOKENS:
                current_open_filehandle.close()
                output_file_paths.append(current_output_file_path)
                logger.info(
                    f"wrote {current_num_tokens} tokens to {str(current_output_file_path)}"
                )
                file_index += 1
                current_output_file_path = _make_jsonl_path(
                    output_file_prefix_path, file_index
                )
                current_open_filehandle = current_output_file_path.open("w")
                current_num_tokens = 0
            current_open_filehandle.write(line)
            current_num_tokens += num_tokens
            total_billable_num_tokens += num_tokens
    finally:
        current_open_filehandle.close()
        output_file_paths.append(current_output_file_path)
    logger.info(f"wrote {current_num_tokens} tokens to {str(current_output_file_path)}")
    logger.info(
        f"openai will charge for {total_billable_num_tokens=} * <number of epochs>"
    )
    return [str(output_file_path) for output_file_path in output_file_paths]


def openai_fine_tuning_jsonl_generator(
    input_data: Union[List[dict], "pd.DataFrame"],
    prompt_key: str,
    completion_key: str,
    system_message: Optional[str],
    model: str,
):
    if is_pandas_dataframe(input_data):
        reader = pandas_row_reader(input_data)
    elif isinstance(input_data, list):
        reader = _dictlist_reader(input_data)
    else:
        raise Exception(f"Invalid {type(input_data)=}")
    # use the token configuration for models that can be fine-tuned
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    tokens_per_message = 3
    encoding = tiktoken.encoding_for_model(model)
    for input_dict in reader:
        prompt = input_dict[prompt_key]
        completion = input_dict[completion_key]
        num_tokens = tokens_per_message
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        messages = []
        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )
            num_tokens += len(encoding.encode(system_message))
        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        num_tokens += len(encoding.encode(prompt))
        messages.append(
            {
                "role": "assistant",
                "content": completion,
            }
        )
        num_tokens += len(encoding.encode(completion))
        data = {
            "messages": messages,
        }
        line = json.dumps(data, separators=(",", ":")) + "\n"
        yield (line, num_tokens)


def _dictlist_reader(input_dictlist: List[dict]):
    for input_dict in input_dictlist:
        yield input_dict


def _make_jsonl_path(output_file_prefix_path: Path, file_index: int) -> Path:
    output_file_path = output_file_prefix_path.with_suffix(f".{file_index:0>6}.jsonl")
    if output_file_path.exists():
        logger.warn(f"overwriting existing file {str(output_file_path)}")
    return output_file_path
