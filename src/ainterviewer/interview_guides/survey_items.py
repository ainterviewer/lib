from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator, create_model


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class SurveyItem(BaseModel):
    type: SurveyItemType
    options: list[str | SurveyOption]
    required: bool = True
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = 1
    with_other: bool = False


class SurveyItemType(StrEnum):
    RADIO = "radio"
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    NUMBER = "number"
    DATE = "date"
    # TODO: Implement
    DATETIME = "datetime"
    TIME = "time"


class SurveyOption(BaseModel):
    """
    An option for a survey item.
    """

    label: str = Field(
        description="The label for the option, will be displayed in the ui."
    )
    tip: str | None = Field(
        None,
        description="A tip to show the interviewee when they hover over the option.",
    )

    @model_validator(mode="after")
    def validate_value(self) -> Self:
        if self.value is None:
            self.value = self.label.lower().replace(" ", "_")

        return self


def create_survey_answer_model(survey_item: SurveyItem) -> type[BaseModel]:
    """Create a pydantic model which can be used to validate/generate answers to the survey item based on its configuration"""

    ...
