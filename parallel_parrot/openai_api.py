try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import re
from typing import List, Optional, Tuple, Union

from aiohttp import ClientSession, ClientTimeout
from aiohttp_retry import ExponentialRetry, RetryClient, JitterRetry

from .types import (
    ParallelParrotError,
    TokenLimitMode,
    ClientSessionType,
    OpenAIChatCompletionConfig,
)
from .util import logger
from .openai_util import openai_token_truncate


OPENAI_REQUEST_TIMEOUT_SECONDS = 120.0
OPENAI_TOTAL_RETRIES = 16
OPENAI_TOTAL_TIMEOUT_SECONDS = 600.0
RATELIMIT_RETRY_SLEEP_SECONDS = 5
MAX_NUM_CONCURRENT_REQUESTS = 1000
MAX_NUM_RATELIMIT_RETRIES = 10
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
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


async def single_setup_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> Tuple[Union[None, str, list], dict, Optional[str]]:
    async with create_chat_completion_client_session(
        config, is_setup_request=True
    ) as client_session:
        response_data = await do_openai_chat_completion(
            client_session=client_session,
            config=config,
            input_row=input_row,
            curried_prompt_template=curried_prompt_template,
            functions=functions,
            function_call=function_call,
        )
    if response_data.status != 200:
        raise ParallelParrotError(f"error in single_setup request: {response_data=}")
    response_result = response_data.body_from_json
    response_headers = response_data.headers
    (model_output, usage) = parse_chat_completion_message_and_usage(response_result)
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    return (model_output, usage, ratelimit_limit_requests)


async def parallel_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input_table: Union[List[dict], "pd.DataFrame"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
    ratelimit_limit_requests: Optional[str] = None,
) -> Tuple[list, List[dict]]:
    if ratelimit_limit_requests:
        # use half of the available capacity at a time, up until the fileshandle system limit
        # https://platform.openai.com/docs/guides/rate-limits/overview
        num_concurrent_requests = min(
            round(int(ratelimit_limit_requests) / 2), MAX_NUM_CONCURRENT_REQUESTS
        )
    else:
        num_concurrent_requests = MAX_NUM_CONCURRENT_REQUESTS
    async with create_chat_completion_client_session(
        config, is_setup_request=False
    ) as client_session:
        semaphore = asyncio.Semaphore(num_concurrent_requests)
        if isinstance(input_table, list):
            input_rows = input_table
        elif isinstance(input_table, pd.DataFrame):
            input_rows = [input_table.iloc[i] for i in range(len(input_table))]
        else:
            raise ParallelParrotError(f"Unexpected type {type(input_table)=}")
        tasks = [
            asyncio.create_task(
                do_chat_completion_with_semaphore_and_ratelimit(
                    client_session=client_session,
                    semaphore=semaphore,
                    config=config,
                    input_row=input_row,
                    curried_prompt_template=curried_prompt_template,
                    functions=functions,
                    function_call=function_call,
                )
            )
            for input_row in input_rows
        ]
        response_data_list = await asyncio.gather(*tasks)
    result_tuples = [
        parse_chat_completion_message_and_usage(response_data.body_from_json)
        for response_data in response_data_list
    ]
    unzipped_results = list(zip(*result_tuples))
    model_outputs = list(unzipped_results[0])
    usage_stats_list = list(unzipped_results[1])
    return (model_outputs, usage_stats_list)


def create_chat_completion_client_session(
    config: OpenAIChatCompletionConfig,
    is_setup_request: bool,
) -> ClientSessionType:
    headers = create_openai_http_headers(config)
    client_timeout = ClientTimeout(total=OPENAI_REQUEST_TIMEOUT_SECONDS)
    client_session = ClientSession(
        headers=headers,
        timeout=client_timeout,
    )
    # Retry error codes which do not indicate a problem with the request itself. Using jitter to avoid thundering herd.
    # The 409 code (openai.error.TryAgain) is returned when the model needs to warm up.
    # https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
    # https://github.com/inyutin/aiohttp_retry/blob/master/aiohttp_retry/retry_options.py#L158
    # https://platform.openai.com/docs/guides/error-codes/api-errors
    retry_statuses = {409, 500, 502, 503}
    if is_setup_request:
        retry_options = ExponentialRetry(
            attempts=OPENAI_TOTAL_RETRIES,
            start_timeout=0.25,
            max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS,
            factor=1.5,
            statuses=retry_statuses,
            exceptions={asyncio.TimeoutError},
            retry_all_server_errors=False,
        )
    else:
        retry_options = JitterRetry(
            attempts=OPENAI_TOTAL_RETRIES,
            start_timeout=1,
            max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS,
            factor=2.0,
            statuses=retry_statuses,
            exceptions={asyncio.TimeoutError},
            random_interval_size=1.5,
            retry_all_server_errors=False,
        )
    retry_client_session = RetryClient(
        client_session=client_session, retry_options=retry_options
    )
    return retry_client_session


async def do_chat_completion_with_semaphore_and_ratelimit(
    client_session: ClientSessionType,
    semaphore: asyncio.Semaphore,
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> OpenAIResponseData:
    async with semaphore:
        response_data = await _chat_completion_with_ratelimit(
            client_session=client_session,
            config=config,
            input_row=input_row,
            curried_prompt_template=curried_prompt_template,
            functions=functions,
            function_call=function_call,
        )
    if response_data.status != 200:
        raise ParallelParrotError(f"error in parallel request: {response_data=}")
    return response_data


async def _chat_completion_with_ratelimit(
    client_session: ClientSessionType,
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
    num_ratelimit_retries: int = 0,
) -> OpenAIResponseData:
    response_data = await do_openai_chat_completion(
        client_session=client_session,
        config=config,
        input_row=input_row,
        curried_prompt_template=curried_prompt_template,
        functions=functions,
        function_call=function_call,
        log_level=logging.DEBUG,
    )
    if response_data.status == 429:
        if "exceeded your current quota" in response_data.reason:
            raise ParallelParrotError(
                f"{response_data.status=} {response_data.reason=}"
            )
        if num_ratelimit_retries >= MAX_NUM_RATELIMIT_RETRIES:
            raise ParallelParrotError(
                f"Too many ratelimit retries: {num_ratelimit_retries=} for {input_row=}"
            )
        sleep_seconds = None
        headers = response_data.headers
        if "error" in response_data.body_from_json:
            error = response_data.body_from_json.get("error", {})
            if error.get("code") == "rate_limit_exceeded":
                # https://platform.openai.com/docs/guides/rate-limits/overview
                if error.get("type") == "tokens":
                    reset_seconds_str = headers.get("x-ratelimit-reset-tokens")
                elif error.get("type") == "requests":
                    reset_seconds_str = headers.get("x-ratelimit-reset-requests")
                else:
                    raise ParallelParrotError(f"Unexpected {error=}")
                reset_seconds = _parse_seconds_from_header(reset_seconds_str)
                if reset_seconds is not None:
                    sleep_seconds = float(reset_seconds)
        else:
            retry_after = headers.get("retry-after")
            if retry_after:
                sleep_seconds = float(retry_after)
        if sleep_seconds is None:
            sleep_seconds = RATELIMIT_RETRY_SLEEP_SECONDS
        logger.warn(
            f"Sleeping for {sleep_seconds=} due to ratelimit "
            f" {response_data.status=} {response_data.reason=} {headers=}"
        )
        await asyncio.sleep(sleep_seconds)
        return await _chat_completion_with_ratelimit(
            client_session=client_session,
            config=config,
            input_row=input_row,
            curried_prompt_template=curried_prompt_template,
            functions=functions,
            function_call=function_call,
            num_ratelimit_retries=(num_ratelimit_retries + 1),
        )
    return response_data


async def do_openai_chat_completion(
    client_session: ClientSessionType,
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Union[None, dict, str] = None,
    log_level: int = logging.INFO,
) -> OpenAIResponseData:
    prompt = curried_prompt_template(input_row)
    payload = create_chat_completion_request_payload(
        config=config,
        prompt=prompt,
        functions=functions,
        function_call=function_call,
    )
    response_data = await _do_openai_chat_completion(
        client_session=client_session,
        payload=payload,
        log_level=log_level,
    )
    print(f"{response_data=}")
    if "error" in response_data.body_from_json:
        error = response_data.body_from_json.get("error", {})
        if error.get("code") == "context_length_exceeded":
            if config.token_limit_mode == TokenLimitMode.RAISE_ERROR:
                raise ParallelParrotError(
                    f"Context length exceeded: {error=} {payload=}"
                )
            elif config.token_limit_mode == TokenLimitMode.TRUNCATE:
                (max_tokens, supplied_tokens) = parse_content_length_exceeded_error(
                    error
                )
                tokens_to_remove = int(supplied_tokens - (max_tokens / 2))
                logger.warn(
                    f"truncating prompt {tokens_to_remove=} {error=}",
                )
                truncated_prompt = openai_token_truncate(
                    prompt, config.model, tokens_to_remove
                )
                payload = create_chat_completion_request_payload(
                    config=config,
                    prompt=truncated_prompt,
                    functions=functions,
                    function_call=function_call,
                )
                response_data = await _do_openai_chat_completion(
                    client_session=client_session,
                    payload=payload,
                    log_level=log_level,
                )
            elif config.token_limit_mode == TokenLimitMode.IGNORE:
                logger.warn(
                    f"Ignoring context length exceeded error: {error=} {payload=}"
                )
    return response_data


async def _do_openai_chat_completion(
    client_session: ClientSessionType,
    payload: dict,
    log_level: int,
):
    logger.log(log_level, f"POST to {OPENAI_CHAT_COMPLETIONS_URL} with {payload=}")
    # https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientResponse
    async with client_session.post(
        OPENAI_CHAT_COMPLETIONS_URL, json=payload
    ) as response:
        if response.content_type == "application/json":
            body_from_json = await response.json()
        else:
            body_from_json = {
                "text": await response.text(),
            }
        if body_from_json is None:
            body_from_json = {}
        response_data = OpenAIResponseData(
            status=response.status,
            reason=str(response.reason),
            headers=dict(response.headers),
            body_from_json=body_from_json,
        )
    logger.log(log_level, f"Response {response_data=} from {payload=}")
    return response_data


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


def parse_chat_completion_message_and_usage(
    response_result: dict,
) -> Tuple[Union[None, str, list], dict]:
    """
    https://platform.openai.com/docs/api-reference/chat/object
    """
    if response_result.get("object") != "chat.completion":
        raise ParallelParrotError(
            f"Unexpected object type: {response_result.get('object')}"
        )
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


def create_openai_http_headers(config: OpenAIChatCompletionConfig) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }
    if config.openai_org_id:
        headers["OpenAI-Organization"] = config.openai_org_id
    return headers


def _parse_seconds_from_header(header_value: Optional[str]) -> Optional[float]:
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
