from typing import Optional, Union

from aiohttp import ClientSession
from aiohttp_retry import RetryClient
from pydantic import BaseModel


class ParallelParrotError(Exception):
    pass


ClientSessionType = Union[ClientSession, RetryClient]


class OpenAIChatCompletionConfig(BaseModel):
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """

    openai_api_key: str
    openai_org_id: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
