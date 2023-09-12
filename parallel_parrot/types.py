from abc import ABC
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Optional, Union

from aiohttp import ClientSession
from aiohttp_retry import RetryClient
from dataclass_utils import check_type


class ParallelParrotError(Exception):
    pass


ParallelParrotOutput = namedtuple("ParallelParrotOutput", ["output", "usage_stats"])

ClientSessionType = Union[ClientSession, RetryClient]


@dataclass()
class LLMConfig(ABC):
    def __post_init__(self):
        check_type(self)


@dataclass()
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
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
