import asyncio
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
    system_message: str = None,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
):
    async with create_chat_completion_client_session(
        config, use_retries=False
    ) as client_session:
        result_tuple = await do_chat_completion_simple(
            client_session=client_session,
            config=config,
            prompt=prompt,
            system_message=system_message,
            functions=functions,
            function_call=function_call,
        )
    return result_tuple


async def parallel_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    prompts: list[str],
    system_message: str = None,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
    ratelimit_limit_requests: str = None,
) -> tuple[list[Optional[str]], list[dict]]:
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
                    system_message=system_message,
                    functions=functions,
                    function_call=function_call,
                )
            )
            for prompt in prompts
        ]
        result_tuples = await asyncio.gather(*tasks)
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
        statuses=[409, 500],
        random_interval_size=5.0,
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
    system_message: Optional[str] = None,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
) -> tuple[Optional[str], dict]:
    if not prompt:
        return (None, OPENAI_EMPTY_USAGE_STATS)
    payload = create_chat_completion_request_payload(
        config=config,
        prompt=prompt,
        system_message=system_message,
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
) -> tuple[Optional[str], dict]:
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
            retry_after = int(response.headers.get("retry-after", "0"))
            if retry_after > 0:
                sleep_seconds = retry_after + RATELIMIT_RETRY_SLEEP_SECONDS
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
        return parse_chat_completion_message_content(response_result)


async def do_chat_completion_simple(
    client_session: ClientSessionType,
    config: OpenAIChatCompletionConfig,
    prompt: str,
    system_message: Optional[str] = None,
    functions: Optional[list[dict]] = None,
    function_call: Union[None, dict, str] = None,
):
    payload = create_chat_completion_request_payload(
        config=config,
        prompt=prompt,
        system_message=system_message,
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
        (model_output, usage) = parse_chat_completion_message_content(response_result)
        return (model_output, usage, response.headers)


def parse_chat_completion_message_content(
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
        model_output = message.get("content")
        finish_reason = message.get("finish_reason")
        if finish_reason != "stop":
            logger.warning(f"Unexpected {finish_reason=} in {response_result=}")
        return (model_output, usage)
    else:
        messages = [choice.get("message", {}) for choice in choices]
        model_outputs = set()
        for message in messages:
            finish_reason = message.get("finish_reason")
            if finish_reason != "stop":
                logger.warning(f"Unexpected {finish_reason=} in {response_result=}")
            model_output = message.get("content")
            if model_output:
                model_outputs.add(model_output)
        return (list(model_outputs), usage)


def create_chat_completion_request_payload(
    config: OpenAIChatCompletionConfig,
    prompt: str,
    system_message: Optional[str] = None,
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
    if system_message:
        messages.append({"role": "system", "content": system_message})
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
