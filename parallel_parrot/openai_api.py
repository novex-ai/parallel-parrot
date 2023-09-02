import asyncio
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from aiohttp_retry import RetryClient, JitterRetry

from .types import ClientSessionType, OpenAIChatCompletionConfig


OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0
OPENAI_TOTAL_RETRIES = 10
OPENAI_TOTAL_TIMEOUT_SECONDS = 600.0
OPENAI_NUM_CONCURRENT_REQUESTS = 20
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMPTY_USAGE_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


async def parallel_openai_chat_completion(
    config: OpenAIChatCompletionConfig,
    prompts: list[str],
    system_message: str = None,
) -> tuple[list[Optional[str]], dict]:
    async with create_chat_completion_client_session(
        config, use_retries=False
    ) as client_session:
        semaphore = asyncio.Semaphore(OPENAI_NUM_CONCURRENT_REQUESTS)
        tasks = [
            asyncio.create_task(
                do_chat_completion(
                    client_session=client_session,
                    semaphore=semaphore,
                    config=config,
                    prompt=prompt,
                    system_message=system_message,
                )
            )
            for prompt in prompts
        ]
        result_tuples = await asyncio.gather(*tasks)
    unzipped_results = list(zip(*result_tuples))
    model_outputs = unzipped_results[0]
    usage_stats_list = unzipped_results[1]
    return (model_outputs, usage_stats_list)


def create_chat_completion_client_session(
    config: OpenAIChatCompletionConfig, use_retries: bool = True
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


async def do_chat_completion(
    client_session: ClientSessionType,
    semaphore: asyncio.Semaphore,
    config: OpenAIChatCompletionConfig,
    prompt: str,
    system_message: Optional[str] = None,
) -> tuple[Optional[str], dict]:
    if not prompt:
        return (None, OPENAI_EMPTY_USAGE_STATS)
    async with semaphore:
        payload = create_chat_completion_request_payload(config, prompt, system_message)
        async with client_session.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload) as resp:
            if resp.status == 429:
                retry_after = int(resp.headers.get("retry-after", "0"))
                if retry_after > 0:
                    await asyncio.sleep(retry_after)
                return await do_chat_completion(
                    client_session, config, prompt, system_message
                )
            resp.raise_for_status()
            result = await resp.json()
            return parse_chat_completion_result(result)


def parse_chat_completion_result(result: dict) -> tuple[Optional[str], dict]:
    """
    https://platform.openai.com/docs/api-reference/chat/object
    """
    if result.get("object") != "chat.completion":
        raise ValueError(f"Unexpected object type: {result.get('object')}")
    choices = result.get("choices", [])
    usage = result.get("usage", OPENAI_EMPTY_USAGE_STATS)
    if len(choices) == 0:
        return (None, usage)
    else:
        choice = choices[0]
        message = choice.get("message", {})
        model_output = message.get("content")
        return (model_output, usage)


def create_chat_completion_request_payload(
    config: OpenAIChatCompletionConfig,
    prompt: str,
    system_message: Optional[str] = None,
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
    return payload


def create_openai_http_headers(config: OpenAIChatCompletionConfig) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }
    if config.openai_org_id:
        headers["OpenAI-Organization"] = config.openai_org_id
    return headers
