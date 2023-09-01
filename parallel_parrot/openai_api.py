from typing import Optional, Union

from aiohttp import ClientSession
from aiohttp_retry import RetryClient, JitterRetry
from pydantic import BaseModel



OPENAI_REQUEST_TIMEOUT_SECONDS = 30.0
OPENAI_TOTAL_TIMEOUT_SECONDS = 300.0
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

ClientSessionType = Union[ClientSession, RetryClient]


class OpenAIChatCompletionConfig(BaseModel):
    openai_api_key: str
    openai_org_id: Optional[str] = None
    model: str
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None


def create_openai_http_headers(config: OpenAIChatCompletionConfig):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.openai_api_key}",
    }
    if config.openai_org_id:
        headers["OpenAI-Organization"] = config.openai_org_id
    return headers


def create_chat_completion_client_session(config: OpenAIChatCompletionConfig, use_retries: bool = True) -> ClientSessionType:
    headers = create_openai_http_headers(config)
    client_session = ClientSession(
        headers=headers,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS
    )
    if not use_retries:
        return client_session
    retry_options = JitterRetry(attempts=5, max_timeout=OPENAI_TOTAL_TIMEOUT_SECONDS)
    retry_client_session = RetryClient(client_session=client_session, retry_options=retry_options)
    return retry_client_session


async def do_chat_completion(client_session: ClientSessionType, config: OpenAIChatCompletionConfig, prompt: str):
    payload = create_chat_completion_request_payload(config, prompt)
    async with client_session.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json()


def create_chat_completion_request_payload(config: OpenAIChatCompletionConfig, prompt: str):
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
    if config.system_message is not None:
        messages.append({
            "role": "system",
            "content": config.system_message
        })
    messages.append({
        "role": "user",
        "content": prompt
    })
    payload["messages"] = messages
    return payload