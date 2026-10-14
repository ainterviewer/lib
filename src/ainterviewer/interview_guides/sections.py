from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from ainterviewer.interview_guides.questions import QuestionBase, QuestionBaseExtended


class GeneratedQuestions(BaseModel):
    n: int = 0
    max_probes_n: int | None = None
    max_probes_time: int | None = None


class QuestionSectionBase[Q: QuestionBase](BaseModel):
    description: str = Field(
        description="A description of the section, used as context for the prober to limit its scope."
    )
    questions: list[Q]


class QuestionSection[Q: QuestionBase](QuestionSectionBase[Q]):
    """A section of questions that all revolve around the same topic"""

    id: UUID = Field(
        default_factory=uuid4,
        description="Stable identity of the section. Survives reordering and is shared by every translation of it.",
    )

    shuffle: bool = Field(
        False,
        description="Should the section be included in shuffling?",
    )
    ai_generated_questions: GeneratedQuestions = GeneratedQuestions()


class QuestionSectionTemplate(QuestionSectionBase[QuestionBase]):
    model_config = {"title": "QuestionSectionTemplate"}

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, values: list):
        return values[:5]


class QuestionSectionBaseExtendedTemplate(QuestionSectionBase[QuestionBaseExtended]):
    model_config = {"title": "QuestionSectionTemplate"}
