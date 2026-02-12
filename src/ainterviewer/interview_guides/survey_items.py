from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from typing import Annotated, Literal, Any

from pydantic import BaseModel, Field, create_model


class SurveyItemType(StrEnum):
    RADIO = "radio"
    CHECKBOX = "checkbox"
    SLIDER = "slider"
    NUMBER = "number"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"


class SurveyItemBase(BaseModel):
    required: bool = True


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class RadioItem(SurveyItemBase):
    type: Literal["radio"] = "radio"
    options: list[str]

    with_other: bool = False

    def validate_answer(self, answer: str):
        return answer in self.options


class CheckboxItem(SurveyItemBase):
    type: Literal["checkbox"] = "checkbox"
    options: list[str]

    with_other: bool = False
    ui: Literal["slider", "radio"] = "radio"

    def validate_answer(self, answer: list[str]):
        return all(x in self.options for x in answer)


class LikertItem(SurveyItemBase):
    type: Literal["likert"] = "likert"
    options: list[str]

    def validate_answer(self, answer: str):
        return answer in self.options


class SliderItem(SurveyItemBase):
    type: Literal["slider"] = "slider"
    options: list[str]

    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = 1

    def validate_answer(self, answer: int | float):
        valid = True

        if self.min is not None and answer <= self.min:
            valid = False

        if self.max is not None and answer >= self.max:
            valid = False

        if self.step is not None:
            base = self.min if self.min is not None else 0
            diff = answer - base

            # handle float precision safely
            eps = 1e-9
            steps = diff / self.step
            if abs(round(steps) - steps) > eps:
                valid = False

        return valid


class NumberItem(SurveyItemBase):
    type: Literal["number"] = "number"

    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = 1

    def validate_answer(self, answer: int | float):
        valid = True

        if self.min is not None and answer <= self.min:
            valid = False

        if self.max is not None and answer >= self.max:
            valid = False

        if self.step is not None:
            base = self.min if self.min is not None else 0
            diff = answer - base

            # handle float precision safely
            eps = 1e-9
            steps = diff / self.step
            if abs(round(steps) - steps) > eps:
                valid = False

        return valid


class DateItem(SurveyItemBase):
    type: Literal["date"] = "date"

    min: str | None = None
    max: str | None = None

    def validate_answer(self, answer: str):
        valid = True

        if self.min is not None and date.fromisoformat(answer) <= date.fromisoformat(
            self.min
        ):
            valid = False

        if self.max is not None and date.fromisoformat(answer) >= date.fromisoformat(
            self.max
        ):
            valid = False

        return valid


class DatetimeItem(SurveyItemBase):
    type: Literal["datetime"] = "datetime"

    min: str | None = None
    max: str | None = None

    def validate_answer(self, answer: str):
        valid = True

        if self.min is not None and datetime.fromisoformat(
            answer
        ) <= datetime.fromisoformat(self.min):
            valid = False

        if self.max is not None and datetime.fromisoformat(
            answer
        ) >= datetime.fromisoformat(self.max):
            valid = False

        return valid


class TimeItem(SurveyItemBase):
    type: Literal["time"] = "time"

    min: str | None = None
    max: str | None = None

    def validate_answer(self, answer: str):
        valid = True

        if self.min is not None and time.fromisoformat(answer) <= time.fromisoformat(
            self.min
        ):
            valid = False

        if self.max is not None and time.fromisoformat(answer) >= time.fromisoformat(
            self.max
        ):
            valid = False

        return valid


SurveyItem = Annotated[
    RadioItem
    | CheckboxItem
    | LikertItem
    | SliderItem
    | NumberItem
    | DateItem
    | DatetimeItem
    | TimeItem,
    Field(discriminator="type"),
]


class SurveyAnswer(BaseModel):
    answer: Any


def create_survey_answer_model(survey_item: SurveyItem) -> SurveyAnswer:
    """Create a pydantic model which can be used to validate/generate answers to the survey item based on its configuration"""

    kwargs: dict = {}

    match survey_item:
        case RadioItem():
            field_type = Literal[tuple(survey_item.options)]
            if survey_item.with_other:
                field_type = field_type | str

        case CheckboxItem():
            inner_type = Literal[tuple(survey_item.options)]
            if survey_item.with_other:
                inner_type = inner_type | str
            field_type = list[inner_type]

        case LikertItem():
            field_type = Literal[tuple(survey_item.options)]

        case SliderItem() | NumberItem():
            field_type = int | float
            if survey_item.min is not None:
                kwargs["gt"] = survey_item.min
            if survey_item.max is not None:
                kwargs["lt"] = survey_item.max

        case DateItem():
            field_type = date
            if survey_item.min is not None:
                kwargs["gt"] = date.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["lt"] = date.fromisoformat(survey_item.max)

        case DatetimeItem():
            field_type = datetime
            if survey_item.min is not None:
                kwargs["gt"] = datetime.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["lt"] = datetime.fromisoformat(survey_item.max)

        case TimeItem():
            field_type = time
            if survey_item.min is not None:
                kwargs["gt"] = time.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["lt"] = time.fromisoformat(survey_item.max)

    if not survey_item.required:
        field_type = field_type | None
        kwargs["default"] = None

    return create_model(
        "SurveyAnswer",
        answer=(field_type, Field(**kwargs)),
    )
