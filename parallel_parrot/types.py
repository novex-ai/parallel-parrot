from collections import namedtuple
from typing import Optional, Union

from aiohttp import ClientSession
from aiohttp_retry import RetryClient
from pydantic import BaseModel


class ParallelParrotError(Exception):
    pass


ParallelParrotOutput = namedtuple("ParallelParrotOutput", ["output", "usage_stats"])

ClientSessionType = Union[ClientSession, RetryClient]


class LLMConfig(BaseModel):
    pass


class OpenAIChatCompletionConfig(LLMConfig):
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """

    openai_api_key: str
    openai_org_id: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None
