# TODO:
# - Should there be a different model for the interview and interview, so
# that the interview version can have another initialization step, i.e.
# shuffle? This could also mean that the interviews should have a fixed
# version of the interview_guide attached, which could make re-creating the
# interview easier.
# - Consider refactoring:
#   - Question into image, survey items and conditions as separate classes
# - Add SkipJsonSchema to fields that shouldn't be considered in the
# json schema --> This will help when giving the schema to the AI as a response
# model

from __future__ import annotations

import warnings
from typing import Generic, Optional, TypeAlias, TypeVar

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from ainterviewer.interview_guides.questions import Question, QuestionBase
from ainterviewer.interview_guides.utils import shuffle_items

Q = TypeVar("Q", bound=QuestionBase)


class InterviewGuideBase(BaseModel, Generic[Q]):
    """A guide for the interviewer to follow during the interview."""

    framing: Optional[str] = Field(
        None,
        description="A description of the interview and its purpose. Only used by the model.",
    )
    introduction: Optional[str | SkipJsonSchema[InterviewMessage]] = Field(
        None,
        description="An introduction to the interview. Displayed to the interviewee as the first message. They whon't be able to respond to this message.",
    )
    question_sections: list[QuestionSection[Q]] = Field(
        default_factory=list,
        description="A list of sections containing questions to ask the interviewee",
    )
    outro: Optional[str | SkipJsonSchema[InterviewMessage]] = Field(
        None,
        description="An outro to the interview. Displayed to the interviewee as the last message. They will not be able to answer this message.",
    )
    alt_outro: SkipJsonSchema[Optional[str]] = Field(
        None,
        description="Used as if the a condition results in an EndInterview, eg. from missing consent.",
    )
    timed_message: SkipJsonSchema[Optional[TimedMessage]] = (
        Field(  # TODO: This should be a list
            None,
            description="A message that is displayed to the interviewee after a certain amount of time",
        )
    )

    def __init__(self, **data):
        super().__init__(**data)

        self.index_questions()

    def index_questions(self):
        n_total = 0

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

                question.n_question = n_total
                n_total += 1

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


class QuestionSection(BaseModel, Generic[Q]):
    """A section of questions that all revolve around the same topic"""

    description: str = Field(
        description="A description of the section, used as context for the prober to limit its scope."
    )
    questions: list[Q]
    shuffle: bool = Field(
        False,
        description="Should the section be included in shuffling?",
    )


class TimedMessage(BaseModel):
    """A message that is displayed to the interviewee after a certain amount of time"""

    message: str = Field(description="The message to display")

    variables: SkipJsonSchema[Optional[list[str]]] = Field(
        None,
        description="Variables that can be used in the message, ie. uuid. In case they are supplied, they will be filled in before the question is asked. The question should be formatted with Jinja2 style templating.",
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
    variables: SkipJsonSchema[Optional[list[str]]] = Field(
        None,
        description="Variables that can be used in the question, ie. uuid",
        # FIXME: Define a list of viable variables from the AInterviewer
        # class that the user should be able to reference, i.e. UUID.
    )


class InterviewGuideTemplate(InterviewGuideBase[QuestionBase]):
    model_config = {"title": "InterviewGuideTemplate"}


class InterviewGuide(InterviewGuideBase[Question]):
    model_config = {"title": "InterviewGuide"}


DecimalString: TypeAlias = str
"""Decimal string, should be in reference to section and question number
where initial condition is added, ie. 1.3 for the 3rd question in the 1st section"""
# TODO: Should the index be a DecimalString or a Tuple[int, int]?


if __name__ == "__main__":
    # print(InterviewGuideContent.model_json_schema())
    #
    # exit()
    #
    # from dainty.documentation import generate_model_docs

    # docs = generate_model_docs(InterviewGuideContent, with_style=True)
    #
    # exit()

    from pathlib import Path

    interview_guides = Path("data/interview_guides").glob("*.json")

    for interview_guide in interview_guides:
        print(f"Validating {interview_guide}")
        with open(interview_guide) as f:
            interview_guide = InterviewGuide.model_validate_json(f.read())
