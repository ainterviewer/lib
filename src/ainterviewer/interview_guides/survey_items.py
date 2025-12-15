from enum import StrEnum
from typing import Optional

from pydantic import BaseModel


class SurveyItemType(StrEnum):
    RADIO = "radio"
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    NUMBER = "number"
    DATE = "date"


class SurveyOption(BaseModel):
    """
    An option for a survey item.
    label [str]: The label for the option, will be displayed in the ui.
    value [Optional[str]]: The value for the option, can be used as a point of reference later in the interview.
    tip: [Optional[str]]: A tip to show the interviewee when they hover over the option.
    """

    label: str
    value: Optional[str] = None
    tip: Optional[str] = None


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class SurveyItem(BaseModel):
    type: SurveyItemType
    options: list[str | SurveyOption]
    required: bool = True
