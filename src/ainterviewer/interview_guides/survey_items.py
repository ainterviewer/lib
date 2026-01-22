from enum import StrEnum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class SurveyItemType(StrEnum):
    RADIO = "radio"
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    NUMBER = "number"
    DATE = "date"


class SurveyOption(BaseModel):
    """
    An option for a survey item.
    """

    label: str = Field(
        description="The label for the option, will be displayed in the ui."
    )
    value: str | None = Field(
        None,
        description="The value for the option, can be used as a point of reference later in the interview.",
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


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class SurveyItem(BaseModel):
    type: SurveyItemType
    options: list[str | SurveyOption]
    required: bool = True
