from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel
from pydantic.types import StringConstraints

type LanguageCode = Annotated[
    str, StringConstraints(to_upper=True, min_length=2, max_length=2)
]


type Device = Literal["cuda", "cpu", "auto"]


class LanguageDict(TypedDict):
    name: str
    code: LanguageCode


class TimeDelta(BaseModel):
    days: int = 0
    seconds: int = 0
    microseconds: int = 0
    milliseconds: int = 0
    minutes: int = 0
    hours: int = 0
    weeks: int = 0


class TranslationDirection(StrEnum):
    TO_FRONTEND = "to_frontend"
    TO_BACKEND = "to_backend"


class MessageType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    CUSTOM_TOKEN = "custom_token"
    SURVEY_ITEM = "survey_item"


class InterviewRole(StrEnum):
    INTERVIEWER = "interviewer"
    RESPONDENT = "respondent"


class MessageRole(StrEnum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class Interviewer(StrEnum):
    HUMAN = "human"
    AI = "ai"


class ClassificationResponse(StrEnum):
    ZERO = "0"
    ONE = "1"
    FALSE = "False"
    TRUE = "True"


class DatabaseType(StrEnum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class Feedback(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class TestType(StrEnum):
    SHUFFLED_AI = "shuffled_ai"
    FIXED_AI = "fixed_ai"
    FIXED_ANSWERS = "fixed_answers"
