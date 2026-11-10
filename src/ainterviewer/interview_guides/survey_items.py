from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, validate_call


class SurveyItemBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = True


validate_strict = validate_call(config=ConfigDict(strict=True))


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class RadioItem(SurveyItemBase):
    type: Literal["radio"] = "radio"
    options: list[str]

    with_other: bool = False

    @validate_strict
    def validate_answer(self, answer: str):
        if self.with_other:
            return True

        return answer in self.options


class CheckboxItem(SurveyItemBase):
    type: Literal["checkbox"] = "checkbox"
    options: list[str]

    with_other: bool = False

    @validate_strict
    def validate_answer(self, answer: list[str]):
        if self.with_other:
            return True

        return all(x in self.options for x in answer)


class LikertItem(SurveyItemBase):
    type: Literal["likert"] = "likert"
    options: list[str]

    ui: Literal["radio", "slider"] = "radio"

    @validate_strict
    def validate_answer(self, answer: str):
        return answer in self.options


def _within[T: (int, float, date, datetime, time)](
    value: T, low: T | None, high: T | None
) -> bool:
    """Whether `value` lies in the closed interval `[low, high]`, either end
    open. Inclusive, like the `min` and `max` of the respondent's input."""
    return (low is None or value >= low) and (high is None or value <= high)


def _on_step(value: float, step: float | None, base: float | None) -> bool:
    """Whether `value` is a whole number of `step`s from `base` (or 0)."""
    if step is None:
        return True
    steps = (value - (base if base is not None else 0)) / step
    # Allow for float imprecision, e.g. 0.3 / 0.1.
    return abs(round(steps) - steps) <= 1e-9


def _iso_bound[T: (date, datetime, time)](
    parse: Callable[[str], T], value: str | None
) -> T | None:
    return None if value is None else parse(value)


class SliderItem(SurveyItemBase):
    type: Literal["slider"] = "slider"

    min: int | float
    max: int | float
    min_label: str | None = None
    max_label: str | None = None
    step: int | float | None = 1

    @validate_strict
    def validate_answer(self, answer: float):
        return _within(answer, self.min, self.max) and _on_step(
            answer, self.step, self.min
        )


class NumberItem(SurveyItemBase):
    type: Literal["number"] = "number"

    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = 1

    @validate_strict
    def validate_answer(self, answer: float):
        return _within(answer, self.min, self.max) and _on_step(
            answer, self.step, self.min
        )


class DateItem(SurveyItemBase):
    type: Literal["date"] = "date"

    min: str | None = None
    max: str | None = None

    @validate_strict
    def validate_answer(self, answer: str):
        parse = date.fromisoformat
        return _within(
            parse(answer), _iso_bound(parse, self.min), _iso_bound(parse, self.max)
        )


class DatetimeItem(SurveyItemBase):
    type: Literal["datetime"] = "datetime"

    min: str | None = None
    max: str | None = None

    @validate_strict
    def validate_answer(self, answer: str):
        parse = datetime.fromisoformat
        return _within(
            parse(answer), _iso_bound(parse, self.min), _iso_bound(parse, self.max)
        )


class TimeItem(SurveyItemBase):
    type: Literal["time"] = "time"

    min: str | None = None
    max: str | None = None

    @validate_strict
    def validate_answer(self, answer: str):
        parse = time.fromisoformat
        return _within(
            parse(answer), _iso_bound(parse, self.min), _iso_bound(parse, self.max)
        )


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
            field_type = Literal[tuple(survey_item.options)]  # ty:ignore[invalid-type-form]
            if survey_item.with_other:
                field_type = field_type | str

        case CheckboxItem():
            inner_type = Literal[tuple(survey_item.options)]  # ty:ignore[invalid-type-form]
            if survey_item.with_other:
                inner_type = inner_type | str
            field_type = list[inner_type]

        case LikertItem():
            field_type = Literal[tuple(survey_item.options)]  # ty:ignore[invalid-type-form]

        case SliderItem() | NumberItem():
            field_type = int | float
            if survey_item.min is not None:
                kwargs["ge"] = survey_item.min
            if survey_item.max is not None:
                kwargs["le"] = survey_item.max

        case DateItem():
            field_type = date
            if survey_item.min is not None:
                kwargs["ge"] = date.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["le"] = date.fromisoformat(survey_item.max)

        case DatetimeItem():
            field_type = datetime
            if survey_item.min is not None:
                kwargs["ge"] = datetime.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["le"] = datetime.fromisoformat(survey_item.max)

        case TimeItem():
            field_type = time
            if survey_item.min is not None:
                kwargs["ge"] = time.fromisoformat(survey_item.min)
            if survey_item.max is not None:
                kwargs["le"] = time.fromisoformat(survey_item.max)

    if not survey_item.required:
        field_type = field_type | None
        kwargs["default"] = None

    return create_model(
        "SurveyAnswer",
        answer=(field_type, Field(**kwargs)),
    )  # ty:ignore[invalid-return-type]
