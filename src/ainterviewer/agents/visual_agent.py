from functools import partial
from typing import Callable

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import VisualAgentPrompts
from ainterviewer.lpm.clients import visual_chat
from ainterviewer.types import MessageRole


class VisualAgent(BaseAgent[VisualAgentPrompts]):
    """Agent that handles visual input"""

    def __init__(
        self,
        model: str,
        *args,
        **kwargs,
    ):
        # TODO: Fix attributes and methods
        super().__init__(*args, **kwargs)
        self.messages += [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]
        self.chat_api: Callable[..., str] = partial(visual_chat, model)

    def describe_image(self, images: list[str]) -> str:
        if not isinstance(images, list):
            images = [images]

        messages = self.messages + [
            {
                "role": "user",
                "content": self.prompts.description_prompt,
                "images": images,
            },
        ]

        text = self.chat_api(messages)  # ty:ignore[invalid-argument-type]

        return text
