from dataclasses import dataclass
import json
import re
from typing import List, Optional, Tuple, Union


from .types import (
    ParallelParrotError,
    OpenAIChatCompletionConfig,
)
from .util import logger


OPENAI_EMPTY_USAGE_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


@dataclass()
class OpenAIResponseData:
    status: int
    reason: str
    headers: dict
    body_from_json: dict
    complete: bool = False


def parse_chat_completion_message_and_usage(
    response_result: dict,
) -> Tuple[Union[None, str, list], dict]:
    """
    https://platform.openai.com/docs/api-reference/chat/object
    """
    if response_result.get("object") != "chat.completion":
        logger.warn(f"Unexpected {response_result=}")
        return (None, OPENAI_EMPTY_USAGE_STATS)
    choices = response_result.get("choices", [])
    usage = response_result.get("usage", OPENAI_EMPTY_USAGE_STATS)
    if len(choices) == 0:
        return (None, usage)
    elif len(choices) == 1:
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        if finish_reason != "stop":
            logger.warning(f"Unexpected {finish_reason=} in {choice=}")
        content = message.get("content")
        if content:
            return (content, usage)
        function_call = message.get("function_call")
        if function_call:
            parsed_arguments = _parse_json_arguments_from_function_call(function_call)
            single_function_param = len(parsed_arguments.keys()) == 1
            if single_function_param:
                # de-nest a single parameter
                param_name = next(iter(parsed_arguments))
                return (parsed_arguments.get(param_name), usage)
            else:
                return (parsed_arguments, usage)
        return (None, usage)
    else:
        content_set = set()
        function_calls = list()
        for choice in choices:
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
            if finish_reason != "stop":
                logger.warning(f"Unexpected {finish_reason=} in {choice=}")
            content = message.get("content")
            if content:
                content_set.add(content)
            else:
                function_call = message.get("function_call")
                if function_call:
                    function_calls.append(function_call)
        if len(content_set) > 0:
            # return a deduped list of string outputs
            return (list(content_set), usage)
        elif len(function_calls) > 0:
            parsed_arguments_list = [
                _parse_json_arguments_from_function_call(function_call)
                for function_call in function_calls
            ]
            first_parsed_arguments = parsed_arguments_list[0]
            single_function_param = len(first_parsed_arguments.keys()) == 1
            if single_function_param:
                param_name = next(iter(first_parsed_arguments))
                param_value = first_parsed_arguments.get(param_name)
                if isinstance(param_value, list):
                    # reduce all of the list parameter outputs into a single list
                    output_list = []
                    for parsed_arguments in parsed_arguments_list:
                        output_list += parsed_arguments.get(param_name, [])
                    # de-nest a single list-valued parameter
                    output = output_list
                    return (output, usage)
                else:
                    # de-nest a single parameter
                    output = [
                        parsed_arguments.get(param_name)
                        for parsed_arguments in parsed_arguments_list
                    ]
                    return (output, usage)
            return (function_calls, usage)
        else:
            return (None, usage)


def parse_content_length_exceeded_error(error: dict):
    message = error.get("message", "")
    match = re.search(
        r"maximum context length is (\d+) tokens. However, your messages resulted in (\d+) tokens",
        message,
    )
    if match:
        return (int(match.group(1)), int(match.group(2)))
    else:
        raise ParallelParrotError(f"Unexpected {message=}")


def create_chat_completion_request_payload(
    config: OpenAIChatCompletionConfig,
    prompt: str,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> dict:
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """
    payload = config.to_payload_dict()
    payload["stream"] = False
    messages = []
    if config.system_message:
        messages.append({"role": "system", "content": config.system_message})
    messages.append({"role": "user", "content": prompt})
    payload["messages"] = messages
    if functions is not None:
        payload["functions"] = functions
    if function_call is not None:
        payload["function_call"] = function_call
    return payload


def parse_seconds_from_header(header_value: Optional[str]) -> Optional[float]:
    if header_value is None:
        return None
    match = re.match(r"([0-9\.]+m)?([0-9\.]+)s", header_value)
    if match:
        minutes = match.group(1)
        seconds = match.group(2)
        if minutes:
            return float(minutes.replace("m", "")) * 60.0 + float(seconds)
        else:
            return float(seconds)
    else:
        return None


def _parse_json_arguments_from_function_call(function_call: dict):
    arguments = function_call.get("arguments")
    if not arguments:
        return None
    try:
        parsed_arguments = json.loads(arguments)
        return parsed_arguments
    except Exception as e:
        logger.warn(f"Could not parse arguments in {function_call=} {e=}")
    return None
