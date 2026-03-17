from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from ainterviewer.interview_guides.questions import QuestionBase

Q = TypeVar("Q", bound=QuestionBase)


class QuestionSectionBase(BaseModel, Generic[Q]):
    description: str = Field(
        description="A description of the section, used as context for the prober to limit its scope."
    )
    questions: list[Q]


class QuestionSection(QuestionSectionBase[Q], Generic[Q]):
    """A section of questions that all revolve around the same topic"""

    shuffle: bool = Field(
        False,
        description="Should the section be included in shuffling?",
    )
    ai_generated_questions: int = 0


class QuestionSectionTemplate(QuestionSectionBase[QuestionBase]):
    model_config = {"title": "QuestionSectionTemplate"}
