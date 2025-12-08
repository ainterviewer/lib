# TODO:
# - Should there be a different model for the interview and interview, so
# that the interview version can have another initialization step, i.e.
# shuffle? This could also mean that the interviews should have a fixed
# version of the interview_guide attached, which could make re-creating the
# interview easier.
# - Consider refactoring:
#   - Question into image, survey items and conditions as separate classes
#   - Should all messages have a language specification?
# - Add SkipJsonSchema to fields that shouldn't be considered in the
# json schema --> This will help when giving the schema to the AI as a response
# model

from __future__ import annotations

import random
import warnings
from typing import Any, Optional, Protocol, TypeAlias, TypeVar

from jinja2 import Template
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from ainterviewer.constants import FP_ASSETS_DIR
from ainterviewer.interview_guides.conditions import Condition
from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.references import Reference
from ainterviewer.interview_guides.survey_item import SurveyItem
from ainterviewer.interview_guides.types import ContextType
from ainterviewer.types import LanguageCode
from ainterviewer.utils import encode_image


class Shuffleable(Protocol):
    shuffle: bool


T = TypeVar("T", bound=Shuffleable)


def shuffle_items(items: list[T]) -> list[T]:
    """Shuffle a list of items, keeping the order of items that have shuffle=False"""

    # Get indices and items where shuffle is True
    shuffleable_indices = [i for i, item in enumerate(items) if item.shuffle]
    shuffleable_items = [items[i] for i in shuffleable_indices]

    # Shuffle the selected items
    random.shuffle(shuffleable_items)

    # Create new list, replacing shuffleable items with shuffled ones
    result = list(items)  # Make a copy
    for new_item, idx in zip(shuffleable_items, shuffleable_indices):
        result[idx] = new_item

    return result


def fill_variables_in_message(
    text: str,
    variables: list[str],
    referable_values: dict[str, Any],
) -> str:
    """Fill in the variables in the main question and probes"""
    values = {}
    for variable in variables:
        if value := referable_values.get(variable):
            values[variable] = value

    template = Template(text)
    return template.render(**values)


class InterviewGuideContent(BaseModel):
    """A guide for the interviewer to follow during the interview."""

    consent: SkipJsonSchema[Consent | None] = Field(
        None,
        description="Object containing text for the consent modal, that the user should accept before the interview can begin",
    )
    welcome: SkipJsonSchema[Welcome | None] = Field(
        None,
        description="Object containing text and relevant information, which will be displayed in a modal after the user has consented to their participation, and before the actual interview starts.",
    )
    framing: Optional[str] = Field(
        None,
        description="A description of the interview and its purpose. Only used by the model.",
    )
    introduction: Optional[str | SkipJsonSchema[InterviewMessage]] = Field(
        None,
        description="An introduction to the interview. Displayed to the interviewee as the first message. They whon't be able to respond to this message.",
    )
    question_sections: list[QuestionSection] = Field(
        default_factory=list,
        description="A list of sections containing questions to ask the interviewee",
    )
    outro: Optional[str | SkipJsonSchema[InterviewMessage]] = Field(
        None,
        description="An outro to the interview. Displayed to the interviewee as the last message.",
    )
    alt_outro: Optional[str] = Field(
        None,
        description="Used as if the a condition results in an EndInterview, eg. from missing consent.",
    )
    timed_message: Optional[TimedMessage] = Field(  # TODO: This should be a list
        None,
        description="A message that is displayed to the interviewee after a certain amount of time",
    )
    shuffle_sections: bool = Field(
        False, description="Should the sections be shuffled?"
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
        if self.shuffle_sections:
            self.question_sections = shuffle_items(self.question_sections)

        for section in self.question_sections:
            if section.shuffle_questions:
                section.questions = shuffle_items(section.questions)

        self.index_questions()

    def reduce(self):
        """Reduce the interview guide to contain one probe per main question"""
        for section in self.question_sections:
            for question in section.questions:
                if question.can_answer:
                    question.max_probes_n = 1
                    question.max_probes_time = None


class QuestionSection(BaseModel):
    """A section of questions that all revolve around the same topic"""

    description: str = Field(
        description="A description of the section, used as context for the prober to limit its scope."
    )
    questions: list[Question]
    shuffle: bool = Field(
        False,
        description="Should the section be included in shuffling? Requires that section shuffling is toggled.",
    )
    shuffle_questions: bool = Field(
        False, description="Should the order of the questions be shuffled?"
    )


class Question(BaseModel):
    """A question that can be asked to the interviewee"""

    description: str | None = Field(
        None,
        description="A description of the question, may be used to reformulate the question and improve the relevance of the probes.",
    )
    main_question: str = Field(description="The question to ask the interviewee")
    alternative_main_questions: list[str] | None = Field(
        None,
        description="List of alternative formulations of the main question, will be chosen at random.",
    )
    probes: Optional[list[str]] = Field(
        None,
        description="A list of possible follow-up questions to ask after the main question",
    )
    max_probes_n: Optional[int] = Field(
        default=4, ge=0, description="Max number of probes"
    )
    max_probes_time: Optional[float] = Field(
        None, gt=0, description="Max time to spend on probing, in seconds"
    )
    language: LanguageCode = Field(
        "EN", description="The language of the main question"
    )
    variables: SkipJsonSchema[Optional[list[str]]] = Field(
        None,
        description="Variables that can be used in the question, ie. uuid. In case they are supplied, they will be filled in before the question is asked. The question should be formatted with Jinja2 style templating.",
    )
    survey_item: SkipJsonSchema[Optional[SurveyItem]] = None
    references: SkipJsonSchema[Optional[list[Reference]]] = None
    image: SkipJsonSchema[Optional[Image]] = None
    user_image: SkipJsonSchema[bool] = Field(
        False,
        description="Whether the user should be able to upload an image as a response",
    )
    condition: SkipJsonSchema[Optional[Condition]] = None
    exclude_from_history: SkipJsonSchema[bool] = Field(
        False,
        description="Exclude from the interview history. This means that the model will not use this question or the response as a context when it asks further questions.",
    )
    probing_context: SkipJsonSchema[Optional[ContextType]] = None
    create_segue: SkipJsonSchema[bool] = Field(
        True,
        description="Create a segue from the previous question, to possibly improve the flow of the interview.",
    )
    check_if_answered: SkipJsonSchema[bool] = Field(
        True,
        description="Check if the question has already been answered under a previous question",
    )
    check_if_exhausted: SkipJsonSchema[bool] = Field(
        False,
        description="During the probing, check if the question has been exhausted and the model should continue to the next question.",
    )
    can_answer: bool = Field(
        True,
        description="Should the user be able to answer the question? Disable this to make the question into a message",
    )
    can_skip: SkipJsonSchema[bool] = Field(
        True, description="Should the user be able to skip the question?"
    )
    shuffle: SkipJsonSchema[bool] = Field(
        True,
        description="Should the question be shuffled? Requires the section to have toggled question shuffling",
    )
    n_question: SkipJsonSchema[int] = Field(
        default=0,
        description="Question number, from 0-n total questions in the interview.",
    )
    index: SkipJsonSchema[tuple[int, int] | None] = Field(
        None,
        description="The index of the question in the interview, ie (section, question) = (2, 2) (for 3rd section 3rd question)",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.__data = data

        if self.survey_item is not None:
            # If survey item is present, set different defaults
            self._set_default("max_probes_n", 0)
            self._set_default("create_segue", False)
            self._set_default("check_if_answered", False)

    def _set_default(self, key, value, strict: bool = False):
        if key not in self.__data:
            setattr(self, key, value)
        elif strict:
            raise OverwriteError(f"{self.__data[key]=:}, {key=:} {value=:}")

    @model_validator(mode="after")
    def check_optionals(self):
        # TODO:
        # Add more checks
        if self.max_probes_n == 0 and self.max_probes_time is not None:
            raise ValueError(
                "Probing time specified, but max_probes is 0. Set max_probes to None (null) to only have time limit."
            )

        if (
            self.probes
            and (self.max_probes_n == 0 or self.max_probes_n is None)
            and (self.max_probes_time is None or self.max_probes_time == 0)
        ):
            raise ValueError(
                f"Probe questions specified, but no proper constraint is set. {self.max_probes_n=:}, {self.max_probes_time=:}"
            )

        return self

    @model_validator(mode="after")
    def check_references(self):
        if self.references:
            n_references = len(self.references)
            if not (n_placeholders := self.main_question.count("{}")) == n_references:
                raise ValueError(
                    f"{n_references} references are specified, but the main question only contains {n_placeholders}. Provide an equal amount of `{{}}` in the main question."
                )
        return self


class OverwriteError(Exception):
    pass


class Image(BaseModel):
    """An image to show the interviewee"""

    primer: Optional[str] = Field(
        None, description="A primer to show the interviewee before showing the image"
    )
    description: str = Field(
        description="A description of the image used to guide the probing"
    )
    alt: str = Field(description="The alt text for the image for accessibility")
    name: str = Field(description="The filename")
    data: str | bytes | None = Field(None, repr=False)

    @property
    def path(self) -> Path:
        return FP_ASSETS_DIR / "images" / self.name

    def encode(self) -> None:
        """Reads the image file and encodes it to Base64, saving it to the data attribute"""
        self.data = encode_image(FP_ASSETS_DIR / "images" / self.name)


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
            interview_guide = InterviewGuideContent.model_validate_json(f.read())
