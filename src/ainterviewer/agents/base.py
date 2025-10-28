from abc import ABC
from typing import Awaitable, Callable, Generic, Optional, TypeVar

from jinja2 import BaseLoader

from ainterviewer.loggers import get_logger
from ainterviewer.lpm.types import Message
from ainterviewer.prompts import get_agent_prompts
from ainterviewer.prompts.models import BasePrompts
from ainterviewer.types import LanguageCode

PromptT = TypeVar("PromptT", bound="BasePrompts")


class BaseAgent(ABC, Generic[PromptT]):
    def __init__(
        self,
        model: str,
        chat_api: Callable[..., Awaitable[str]],
        lang: LanguageCode = "EN",
        template_loader: Optional[BaseLoader] = None,
        *args,
        **kwargs,
    ):
        self.prompts: PromptT = get_agent_prompts(
            self.__class__.__name__,
            lang=lang,
            template_loader=template_loader,
            *args,
            **kwargs,
        )
        self.messages: list[Message] = []
        self.chat_api = chat_api
        self.model = model
        self._lang = lang
        self.logger = get_logger(agent=self.__class__.__name__)
        self.logger.info("Agent initialized")
