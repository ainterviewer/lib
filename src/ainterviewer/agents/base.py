from abc import ABC
from typing import Any, Generic, TypeVar, overload

from jinja2 import BaseLoader
from pydantic import BaseModel

from ainterviewer.agents.prompts import BasePrompts, get_agent_prompts
from ainterviewer.loggers import get_logger
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message
from ainterviewer.types import LanguageCode

PromptT = TypeVar("PromptT", bound="BasePrompts")

T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC, Generic[PromptT]):
    messages: list[Message]

    def __init__(
        self,
        model: str,
        language: LanguageCode,
        chat_kwargs: dict[str, Any] | None = None,
        template_loader: BaseLoader | None = None,
        *args,
        **kwargs,
    ):
        self.prompts: PromptT = get_agent_prompts(
            self.__class__.__name__,
            lang=language,
            template_loader=template_loader,
            *args,
            **kwargs,
        )
        self.messages = []
        self.chat_kwargs = chat_kwargs if chat_kwargs else {}
        self.model = model
        self._language = language
        self.logger = get_logger(agent=self.__class__.__name__, language=language)
        self.logger.info("Agent initialized")

    @property
    def language(self):
        return self._language

    @overload
    async def chat_api(
        self,
        messages: list[Message],
        response_format: type[T],
        **kwargs,
    ) -> T: ...

    @overload
    async def chat_api(
        self,
        messages: list[Message],
        response_format: None = None,
        **kwargs,
    ) -> str: ...

    async def chat_api(
        self,
        messages: list[Message],
        response_format: type[T] | None = None,
        **kwargs,
    ) -> str | T:
        return await chat(
            messages,
            model=self.model,
            response_format=response_format,
            **self.chat_kwargs,
            **kwargs,
        )
