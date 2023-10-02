from abc import ABC
from collections import namedtuple
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Optional, List, Union

from aiohttp import ClientSession
from aiohttp_retry import RetryClient
from dataclass_utils import check_type


class ParallelParrotError(Exception):
    pass


ParallelParrotOutput = namedtuple("ParallelParrotOutput", ["output", "usage_stats"])

ClientSessionType = Union[ClientSession, RetryClient]


class TokenLimitMode(Enum):
    RAISE_ERROR = "RAISE_ERROR"
    TRUNCATE = "TRUNCATE"
    IGNORE = "IGNORE"


@dataclass()
class LLMConfig(ABC):
    def __post_init__(self):
        if self.token_limit_mode is None:
            self.token_limit_mode = TokenLimitMode.RAISE_ERROR
        check_type(self)

    def get_nonpassthrough_names(self) -> List[str]:
        return []

    def to_payload_dict(self):
        raw_dict = asdict(self)
        nonpassthrough_names = self.get_nonpassthrough_names()
        payload_dict = {
            key: value
            for key, value in raw_dict.items()
            if key not in nonpassthrough_names and value is not None
        }
        return payload_dict


@dataclass()
class OpenAIChatCompletionConfig(LLMConfig):
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """

    openai_api_key: str
    model: str = "gpt-3.5-turbo"
    openai_org_id: Optional[str] = None
    system_message: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    token_limit_mode: TokenLimitMode = TokenLimitMode.RAISE_ERROR

    def get_nonpassthrough_names(self) -> List[str]:
        return [
            "openai_api_key",
            "openai_org_id",
            "system_message",
            "token_limit_mode",
        ] + super().get_nonpassthrough_names()
