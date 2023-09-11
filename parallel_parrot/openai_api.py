import asyncio
import json
from typing import Optional, Union

from aiohttp import ClientSession, ClientTimeout
from aiohttp_retry import RetryClient, JitterRetry

from .types import ParallelParrotError, ClientSessionType, OpenAIChatCompletionConfig
from .util import logger


OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0
OPENAI_TOTAL_RETRIES = 10
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


async def single_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    prompt: str,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> tuple[Union[None, str, list], dict, dict]:
    async with create_chat_completion_client_session(
        config, use_retries=False
    ) as client_session:
        (response_result, response_headers) = await do_chat_completion_simple(
            client_session=client_session,
            config=config,
            prompt=prompt,
            functions=functions,
            function_call=function_call,
        )
        (model_output, usage) = parse_chat_completion_message_and_usage(response_result)
    return (model_output, usage, response_headers)


async def parallel_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    prompts: list[str],
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
    ratelimit_limit_requests: Optional[str] = None,
) -> tuple[list, list[dict]]:
    if ratelimit_limit_requests:
        # use half of the available capacity at a time, up until the fileshandle system limit
        # https://platform.openai.com/docs/guides/rate-limits/overview
        num_concurrent_requests = min(
            round(int(ratelimit_limit_requests) / 2), MAX_NUM_CONCURRENT_REQUESTS
        )
    else:
        num_concurrent_requests = MAX_NUM_CONCURRENT_REQUESTS
    async with create_chat_completion_client_session(
        config, use_retries=True
    ) as client_session:
        semaphore = asyncio.Semaphore(num_concurrent_requests)
        tasks = [
            asyncio.create_task(
                do_chat_completion_with_semaphore(
                    client_session=client_session,
                    semaphore=semaphore,
                    config=config,
                    prompt=prompt,
                    functions=functions,
                    function_call=function_call,
                )
            )
            for prompt in prompts
        ]
        response_results = await asyncio.gather(*tasks)
    result_tuples = [
        parse_chat_completion_message_and_usage(response_result)
        for response_result in response_results
    ]
    unzipped_results = list(zip(*result_tuples))
    model_outputs = list(unzipped_results[0])
    usage_stats_list = list(unzipped_results[1])
    return (model_outputs, usage_stats_list)


def create_chat_completion_client_session(
    config: OpenAIChatCompletionConfig,
    use_retries: bool,
) -> ClientSessionType:
    headers = create_openai_http_headers(config)
    client_timeout = ClientTimeout(total=OPENAI_REQUEST_TIMEOUT_SECONDS)
    client_session = ClientSession(
        headers=headers,
        timeout=client_timeout,
    )
    if not use_retries:
        return client_session
    # Retry error codes which do not indicate a problem with the request itself. Using jitter to avoid thundering herd.
    # The 409 code (openai.error.TryAgain) is returned when the model needs to warm up.
    # https://github.com/openai/openai-python/blob/1be14ee34a0f8e42d3f9aa5451aa4cb161f1781f/openai/api_requestor.py#L401
    # https://github.com/inyutin/aiohttp_retry/blob/master/aiohttp_retry/retry_options.py#L158
    retry_options = JitterRetry(
        attempts=OPENAI_TOTAL_RETRIES,
        start_timeout=1,
        max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS,
        factor=2.0,
        statuses={409, 500},
        exceptions={asyncio.TimeoutError},
        random_interval_size=1.5,
        retry_all_server_errors=True,
    )
    retry_client_session = RetryClient(
        client_session=client_session, retry_options=retry_options
    )
    return retry_client_session


async def do_chat_completion_with_semaphore(
    client_session: ClientSessionType,
    semaphore: asyncio.Semaphore,
    config: OpenAIChatCompletionConfig,
    prompt: str,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> Optional[dict]:
    if not prompt:
        return None
    payload = create_chat_completion_request_payload(
        config=config,
        prompt=prompt,
        functions=functions,
        function_call=function_call,
    )
    async with semaphore:
        return await _chat_completion_with_ratelimit(
            client_session=client_session,
            payload=payload,
        )


async def _chat_completion_with_ratelimit(
    client_session: ClientSessionType,
    payload: dict,
    num_ratelimit_retries: int = 0,
) -> dict:
    logger.debug(f"POST to {OPENAI_CHAT_COMPLETIONS_URL} with {payload=}")
    async with client_session.post(
        OPENAI_CHAT_COMPLETIONS_URL, json=payload
    ) as response:
        logger.debug(f"Response {response.status=} from {payload=} {response.headers=}")
        if response.status == 429:
            if num_ratelimit_retries >= MAX_NUM_RATELIMIT_RETRIES:
                raise ParallelParrotError(
                    f"Too many ratelimit retries: {num_ratelimit_retries=} for {payload=}"
                )
            retry_after = float(response.headers.get("retry-after", "0"))
            if retry_after > 0:
                sleep_seconds = retry_after
            else:
                sleep_seconds = RATELIMIT_RETRY_SLEEP_SECONDS
            logger.debug(
                f"Sleeping for {sleep_seconds=} due to ratelimit {response.status=} {response.headers=}"
            )
            await asyncio.sleep(sleep_seconds)
            return await _chat_completion_with_ratelimit(
                client_session=client_session,
                payload=payload,
                num_ratelimit_retries=(num_ratelimit_retries + 1),
            )
        response.raise_for_status()
        response_result = await response.json()
        logger.debug(f"Response {response_result=} from {payload=}")
        return response_result


async def do_chat_completion_simple(
    client_session: ClientSessionType,
    config: OpenAIChatCompletionConfig,
    prompt: str,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> tuple[dict, dict]:
    payload = create_chat_completion_request_payload(
        config=config,
        prompt=prompt,
        functions=functions,
        function_call=function_call,
    )
    logger.info(f"POST to {OPENAI_CHAT_COMPLETIONS_URL} with {payload=}")
    async with client_session.post(
        OPENAI_CHAT_COMPLETIONS_URL, json=payload
    ) as response:
        logger.info(f"Response {response.status=} from {payload=} {response.headers=}")
        response.raise_for_status()
        response_result = await response.json()
        logger.info(f"Response {response_result=} from {payload=}")
        return (response_result, dict(response.headers))


def parse_chat_completion_message_and_usage(
    response_result: dict,
) -> tuple[Union[None, str, list], dict]:
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
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> dict:
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """
    payload = {
        "model": config.model,
        "stream": False,
    }
    if config.temperature is not None:
        payload["temperature"] = config.temperature
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.n is not None:
        payload["n"] = config.n
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.presence_penalty is not None:
        payload["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        payload["frequency_penalty"] = config.frequency_penalty
    if config.logit_bias is not None:
        payload["logit_bias"] = config.logit_bias
    if config.user is not None:
        payload["user"] = config.user
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
