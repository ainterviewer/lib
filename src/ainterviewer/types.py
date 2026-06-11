from __future__ import annotations

from datetime import timedelta
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict
from pydantic.types import StringConstraints

type LanguageCode = Annotated[
    str, StringConstraints(to_upper=True, min_length=2, max_length=2)
]


type Device = Literal["cuda", "cpu", "auto"]


class LanguageDict(TypedDict):
    name: str
    code: LanguageCode


class TimeDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    days: int = 0
    seconds: int = 0
    microseconds: int = 0
    milliseconds: int = 0
    minutes: int = 0
    hours: int = 0
    weeks: int = 0

    @classmethod
    def parse_timedelta(cls, value: timedelta) -> TimeDelta:
        return TimeDelta(seconds=int(value.total_seconds()))

    def to_timedelta(self) -> timedelta:
        return timedelta(**self.model_dump())


class TranslationDirection(StrEnum):
    TO_FRONTEND = "to_frontend"
    TO_BACKEND = "to_backend"


class MessageType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
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


class InterviewStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    COMPLETED = "completed"
