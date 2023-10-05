try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None

import asyncio
from collections.abc import Callable

import logging
import math
import time
from typing import List, Optional, Tuple, Union

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector
from aiohttp_retry import ExponentialRetry, RetryClient, JitterRetry

from .types import (
    ParallelParrotError,
    TokenLimitMode,
    ClientSessionType,
    OpenAIChatCompletionConfig,
)
from .util import logger, sum_usage_stats
from .openai_util import openai_token_truncate
from .openai_api_lib import (
    OpenAIResponseData,
    prep_openai_function_list_of_objects,
    create_chat_completion_request_payload,
    parse_chat_completion_message_and_usage,
    parse_content_length_exceeded_error,
    parse_seconds_from_header,
    parse_json_arguments_from_function_call,
)

try:
    import resource

    # maximize the number of concurrent connections for this process
    rlimit_soft, rlimit_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit_hard, rlimit_hard))
    except Exception as e:
        logger.warning(f"Could not set rlimit: {e=}")
    rlimit_soft, rlimit_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    MAX_NUM_CONCURRENT_REQUESTS = max(120, rlimit_soft - 80)
except ImportError:
    MAX_NUM_CONCURRENT_REQUESTS = 120


OPENAI_REQUEST_TIMEOUT_SECONDS = 120.0
MAX_HTTP_RETRIES = 16
OPENAI_TOTAL_TIMEOUT_SECONDS = 600.0
RATELIMIT_RETRY_SLEEP_SECONDS = 5
MAX_NUM_RATELIMIT_RETRIES = 20
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_FUNCTION_NAME = "f"
OPENAI_FUNCTION_PARAMETER_NAME = "p"


throttle_until_time = 0.0


async def single_setup_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    function_output_key_names: Optional[List[str]],
) -> Tuple[Union[None, str, list], dict, Optional[str]]:
    if function_output_key_names is not None:
        function_name = OPENAI_FUNCTION_NAME
        parameter_name = OPENAI_FUNCTION_PARAMETER_NAME
        (functions, function_call) = prep_openai_function_list_of_objects(
            function_name=function_name,
            parameter_name=parameter_name,
            output_key_names=function_output_key_names,
        )
    else:
        function_name = None
        parameter_name = None
        functions = None
        function_call = None
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
    if not response_data.complete:
        raise ParallelParrotError(f"error in single_setup request: {response_data=}")
    response_result = response_data.body_from_json
    response_headers = response_data.headers
    (model_output, usage) = parse_chat_completion_message_and_usage(
        response_result,
        function_name=function_name,
        parameter_name=parameter_name,
    )
    ratelimit_limit_requests = response_headers.get("x-ratelimit-limit-requests")
    return (model_output, usage, ratelimit_limit_requests)


async def parallel_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    input_table: Union[List[dict], "pd.DataFrame"],
    curried_prompt_template: Callable,
    function_output_key_names: Optional[List[str]],
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
    logger.info(f"using {num_concurrent_requests=}")
    if function_output_key_names is not None:
        function_name = OPENAI_FUNCTION_NAME
        parameter_name = OPENAI_FUNCTION_PARAMETER_NAME
        (functions, function_call) = prep_openai_function_list_of_objects(
            function_name=function_name,
            parameter_name=parameter_name,
            output_key_names=function_output_key_names,
        )
    else:
        function_name = None
        parameter_name = None
        functions = None
        function_call = None
    async with create_chat_completion_client_session(
        config, is_setup_request=False
    ) as client_session:
        num_chunks = math.ceil(len(input_table) / num_concurrent_requests)
        response_data_list = []
        for chunk in range(num_chunks):
            start_index = chunk * num_concurrent_requests
            end_index = min(start_index + num_concurrent_requests, len(input_table))
            logger.info(f"processing chunk of data {start_index=} {end_index=}")
            if isinstance(input_table, list):
                input_rows = input_table[start_index:end_index]
            elif isinstance(input_table, pd.DataFrame):
                input_rows = [
                    input_table.iloc[i] for i in range(start_index, end_index)
                ]
            else:
                raise ParallelParrotError(f"Unexpected type {type(input_table)=}")
            tasks = [
                asyncio.create_task(
                    _chat_completion_with_ratelimit(
                        client_session=client_session,
                        config=config,
                        input_row=input_row,
                        curried_prompt_template=curried_prompt_template,
                        functions=functions,
                        function_call=function_call,
                    )
                )
                for input_row in input_rows
            ]
            response_data_list += await asyncio.gather(*tasks)
    result_tuples = [
        parse_chat_completion_message_and_usage(
            response_data.body_from_json,
            function_name=function_name,
            parameter_name=parameter_name,
        )
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
        connector=TCPConnector(limit=MAX_NUM_CONCURRENT_REQUESTS),
        headers=headers,
        timeout=client_timeout,
    )
    # Retry error codes which do not indicate a problem with the request itself. Using jitter to avoid thundering herd.
    # The 409 code (openai.error.TryAgain) is returned when the model needs to warm up.
    # https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
    # https://github.com/inyutin/aiohttp_retry/blob/master/aiohttp_retry/retry_options.py#L158
    # https://platform.openai.com/docs/guides/error-codes/api-errors
    retry_statuses = {409, 500, 502, 503}
    retry_exceptions = {asyncio.TimeoutError, ClientError}
    if is_setup_request:
        retry_options = ExponentialRetry(
            attempts=MAX_HTTP_RETRIES,
            start_timeout=0.25,
            max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS,
            factor=1.5,
            statuses=retry_statuses,
            exceptions=retry_exceptions,
            retry_all_server_errors=False,
        )
    else:
        retry_options = JitterRetry(
            attempts=MAX_HTTP_RETRIES,
            start_timeout=1,
            max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS,
            factor=2.0,
            statuses=retry_statuses,
            exceptions=retry_exceptions,
            random_interval_size=1.5,
            retry_all_server_errors=False,
        )
    retry_client_session = RetryClient(
        client_session=client_session, retry_options=retry_options
    )
    return retry_client_session


async def _chat_completion_with_ratelimit(
    client_session: ClientSessionType,
    config: OpenAIChatCompletionConfig,
    input_row: Union[dict, "pd.Series"],
    curried_prompt_template: Callable,
    functions: Optional[List[dict]] = None,
    function_call: Optional[dict] = None,
    num_ratelimit_retries: int = 0,
) -> OpenAIResponseData:
    global throttle_until_time
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
                reset_seconds = parse_seconds_from_header(reset_seconds_str)
                if reset_seconds is not None:
                    sleep_seconds = float(reset_seconds)
            elif error.get("code") == "insufficient_quota":
                raise ParallelParrotError(f"Insufficient quota: {response_data=}")
        else:
            retry_after = headers.get("retry-after")
            if retry_after:
                sleep_seconds = float(retry_after)
        if sleep_seconds is None:
            sleep_seconds = RATELIMIT_RETRY_SLEEP_SECONDS
        throttle_until_time = max(
            time.monotonic() + sleep_seconds + RATELIMIT_RETRY_SLEEP_SECONDS,
            throttle_until_time,
        )
        logger.warning(
            f"Sleeping for {sleep_seconds=} due to ratelimit "
            f" {throttle_until_time=}"
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
    function_call: Optional[dict] = None,
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
    retry_usage_list = []
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
                logger.warning(
                    f"truncating prompt and re-doing request {tokens_to_remove=} {error=}",
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
                retry_usage_list.append(response_data.body_from_json.get("usage", {}))
                response_data = await _do_openai_chat_completion(
                    client_session=client_session,
                    payload=payload,
                    log_level=log_level,
                )
            elif config.token_limit_mode == TokenLimitMode.IGNORE:
                logger.warning(
                    f"Ignoring context length exceeded error: {error=} {payload=}"
                )
                response_data.complete = True
    elif function_call is not None:
        choices = response_data.body_from_json.get("choices", [])
        found_invalid_function_response = False
        for choice in choices:
            message = choice.get("message", {})
            response_function_call = message.get("function_call")
            if response_function_call is None:
                logger.warning(
                    f"Function not called.  Re-doing request {response_function_call=} in {choice=}"
                )
                found_invalid_function_response = True
            elif response_function_call.get("name") != function_call.get("name"):
                logger.warning(
                    f"Mismatched function name. Re-doing request {response_function_call=} in {choice=}"
                )
                found_invalid_function_response = True
            elif (
                parse_json_arguments_from_function_call(response_function_call) is None
            ):
                logger.warning(
                    f"Invalid JSON arguments. Re-doing request {response_function_call=} in {choice=}"
                )
                found_invalid_function_response = True
        if found_invalid_function_response:
            retry_usage_list.append(response_data.body_from_json.get("usage", {}))
            response_data = await _do_openai_chat_completion(
                client_session=client_session,
                payload=payload,
                log_level=log_level,
            )
    if len(retry_usage_list) > 0:
        last_usage = response_data.body_from_json["usage"]
        total_usage = sum_usage_stats(retry_usage_list + [last_usage])
        response_data.body_from_json["usage"] = total_usage
    return response_data


async def _do_openai_chat_completion(
    client_session: ClientSessionType,
    payload: dict,
    log_level: int,
) -> OpenAIResponseData:
    global throttle_until_time
    throttle_seconds = throttle_until_time - time.monotonic()
    if throttle_seconds > 0:
        logger.info(f"Throttling for {throttle_seconds=}")
        await asyncio.sleep(throttle_seconds)
    logger.log(log_level, f"POST to {OPENAI_CHAT_COMPLETIONS_URL} with {payload=}")
    # https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.ClientResponse
    async with client_session.post(
        OPENAI_CHAT_COMPLETIONS_URL, json=payload
    ) as response:
        if response.content_type == "application/json":
            body_from_json = await response.json()
            if body_from_json is None:
                body_from_json = {}
        else:
            body_from_json = {
                "text": await response.text(),
            }
        response_data = OpenAIResponseData(
            status=response.status,
            reason=str(response.reason),
            headers=dict(response.headers),
            body_from_json=body_from_json,
            complete=(response.status == 200),
        )
    logger.log(log_level, f"Response {response_data=} from {payload=}")
    return response_data


def create_openai_http_headers(config: OpenAIChatCompletionConfig) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }
    if config.openai_org_id:
        headers["OpenAI-Organization"] = config.openai_org_id
    return headers
