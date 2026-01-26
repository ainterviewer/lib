from enum import StrEnum
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from pydantic import Field

from ainterviewer.types import MessageRole


class CustomTokens(StrEnum):
    end_of_probe = "<|endofprobe|>"
    end_of_section = "<|endofsection|>"
    end_of_interview = "<|endofinterview|>"
    skip_question = "<|skipquestion|>"
    skip_section = "<|skipsection|>"
    no_answer = "<|noanswer|>"
    restart_interview = "<|restartinterview|>"

    @classmethod
    @property
    def all(cls) -> tuple[str, ...]:
        return tuple(token.value for token in cls)


class Message(TypedDict):
    role: MessageRole
    content: str
    images: NotRequired[list[Path]]


Temperature = Annotated[float, Field(ge=0, le=2)]
