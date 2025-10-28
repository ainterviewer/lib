from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from ainterviewer.types import MessageRole


class BinaryClassification(BaseModel):
    output: Literal["0", "1"]


class CustomTokens:
    end_of_probe = "<|endofprobe|>"
    end_of_section = "<|endofsection|>"
    end_of_interview = "<|endofinterview|>"
    skip_question = "<|skipquestion|>"
    skip_section = "<|skipsection|>"
    no_answer = "<|noanswer|>"
    restart_interview = "<|restartinterview|>"
    all = [
        end_of_probe,
        end_of_section,
        end_of_interview,
        skip_question,
        no_answer,
        restart_interview,
    ]


class Message(TypedDict):
    role: MessageRole
    content: str
    images: NotRequired[list[Path]]


Temperature = Annotated[float, Field(ge=0, le=2)]
