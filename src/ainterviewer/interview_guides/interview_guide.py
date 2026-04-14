# TODO:
# - Consider refactoring:
#   - Question into image, survey items and conditions as separate classes
# - Add SkipJsonSchema to fields that shouldn't be considered in the
# json schema --> This will help when giving the schema to the AI as a response
# model

from __future__ import annotations

import warnings
from typing import Generic

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from ainterviewer.interview_guides.questions import Question, QuestionBase
from ainterviewer.interview_guides.sections import Q, QuestionSection
from ainterviewer.interview_guides.utils import shuffle_items


class InterviewGuideBase(BaseModel, Generic[Q]):
    """A guide for the interviewer to follow during the interview."""

    framing: str | None = Field(
        None,
        description="A description of the interview and its purpose. Only used by the model.",
    )
    introduction: str | SkipJsonSchema[InterviewMessage] | None = Field(
        None,
        description="An introduction to the interview. Displayed to the interviewee as the first message. They whon't be able to respond to this message.",
    )
    question_sections: list[QuestionSection[Q]] = Field(
        default_factory=list,
        description="A list of sections containing questions to ask the interviewee",
    )
    outro: str | SkipJsonSchema[InterviewMessage] | None = Field(
        None,
        description="An outro message for the interview. Displayed to the interviewee as the last message they will see. They will not be able to answer this message.",
    )
    # TODO: Should this kind of message instead be specified in the condition?
    alt_outro: SkipJsonSchema[str | None] = Field(
        None,
        description="Used if a condition results in an EndInterview, eg. from missing consent. Displayed to the interviewee as the last message they will see. They will not be able to answer this message.",
    )


class TimedMessage(BaseModel):
    """A message that is displayed to the interviewee after a certain amount of time"""

    message: str = Field(description="The message to display")

    variables: SkipJsonSchema[list[str] | None] = Field(
        None,
        description="Variables that can be used in the message, ie. uuid. In case they are supplied, they will be filled in before the message is displayed. The message should be formatted with Jinja2 style templating.",
    )
    time: int = Field(
        description="The time in seconds to wait before displaying the message"
    )
    include_in_history: bool = Field(
        False,
        description="Should the message be included in the interview history?",
    )
    as_modal: bool = Field(
        False,
        description="Should the message be shown as a modal? Defaults to False meaning it will be shown as a normal message.",
    )


class InterviewMessage(BaseModel):
    message: str
    variables: SkipJsonSchema[list[str] | None] = Field(
        None,
        description="Variables that can be used in the question, ie. uuid",
        # FIXME: Define a list of viable variables from the AInterviewer
        # class that the user should be able to reference, i.e. UUID.
    )


class InterviewGuideTemplate(InterviewGuideBase[QuestionBase]):
    model_config = {"title": "InterviewGuideTemplate"}


class InterviewGuide(InterviewGuideBase[Question]):
    model_config = {"title": "InterviewGuide"}

    timed_messages: list[TimedMessage] | None = Field(
        None,
        description="Messages that are displayed to the interviewee after a certain amount of time",
    )
    ai_generated_sections: int = 0

    def __init__(self, **data):
        super().__init__(**data)

        self.index_questions()

    @property
    def n_total_questions(self) -> int:
        return (
            sum(
                len(section.questions) + section.ai_generated_questions
                for section in self.question_sections
            )
            + self.ai_generated_sections * 5
        )

    def index_questions(self):
        for n_section, section in enumerate(self.question_sections):
            for n_question, question in enumerate(section.questions):
                # Avoids settings when shuffling
                if question.index is None:
                    indexes = [
                        question.index
                        for section in self.question_sections
                        for question in section.questions
                        if question.index
                    ]
                    if (index := (n_section, n_question)) in indexes:
                        warnings.warn(f"Index {index} already exists, overwriting")

                    question.index = index

    def shuffle(self):
        """Shuffles sections and questions in the interview guide, based on
        their configuration."""
        self.question_sections = shuffle_items(self.question_sections)

        for section in self.question_sections:
            section.questions = shuffle_items(section.questions)

        self.index_questions()

    def reduce(self):
        """Reduce the interview guide to contain one probe per main question.
        Used as a utility when synthesizing fixed answers.
        """

        for section in self.question_sections:
            for question in section.questions:
                if question.can_answer:
                    question.max_probes_n = 1
                    question.max_probes_time = None


if __name__ == "__main__":
    from pathlib import Path

    interview_guides = Path("data/interview_guides").glob("*.json")

    for interview_guide in interview_guides:
        print(f"Validating {interview_guide}")
        with open(interview_guide) as f:
            interview_guide = InterviewGuide.model_validate_json(f.read())
