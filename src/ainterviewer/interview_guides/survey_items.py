from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

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


#: What a checkbox selection is joined with when it is submitted as one answer.
CHECKBOX_SEPARATOR = ", "


def _normalize_option(value: str) -> str:
    return " ".join(value.split()).casefold()


def split_checkbox_answer(answer: str, options: Sequence[str]) -> list[str]:
    """Undo the `CHECKBOX_SEPARATOR` join a checkbox selection is stored as.

    An option may itself contain a comma ("Yes, often"), so the answer is not
    split blindly: a run of comma-separated parts that together spell one of
    `options` is kept whole, the longest run first. The parts that spell no
    option are what the respondent typed into "Other", which may hold commas
    of its own, so they are joined back into a single value, placed last.

    Options are recognised on whitespace-collapsed, case-folded text; the
    values are returned as they appear in the answer.
    """
    parts = answer.split(",")
    known = {_normalize_option(option) for option in options}
    longest = max((option.count(",") for option in options), default=0) + 1

    values: list[str] = []
    other: list[str] = []
    start = 0
    while start < len(parts):
        for end in range(min(len(parts), start + longest), start, -1):
            candidate = ",".join(parts[start:end]).strip()
            if candidate and _normalize_option(candidate) in known:
                values.append(candidate)
                start = end
                break
        else:
            if parts[start].strip():
                other.append(parts[start].strip())
            start += 1

    if other:
        values.append(CHECKBOX_SEPARATOR.join(other))
    return values


class LikertItem(SurveyItemBase):
    type: Literal["likert"] = "likert"
    options: list[str]

    ui: Literal["radio", "slider"] = "radio"

    @validate_strict
    def validate_answer(self, answer: str):
        return answer in self.options


class GridRow(BaseModel):
    """One statement of a Likert grid, rated on the grid's scale."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(
        default_factory=uuid4,
        description="Stable identity of the row; the answer is keyed by it.",
    )
    text: str
    required: bool = Field(
        True, description="Whether the row must be rated to submit the grid."
    )


class LikertGridItem(SurveyItemBase):
    """Several statements rated on one shared Likert scale, answered together
    as one answer.

    The answer is a JSON object from row id to the chosen option's text
    (`encode_answer`); a row left blank is absent from it. It is stored and
    sent as that JSON, and read back with `parse_answer` -- or as text a
    person or model can read with `describe_answer`.
    """

    type: Literal["likert_grid"] = "likert_grid"
    options: list[str]
    rows: list[GridRow]

    @validate_strict
    def validate_answer(self, answer: dict[str, str]):
        rows = {str(row.id): row for row in self.rows}
        return all(
            key in rows and value in self.options for key, value in answer.items()
        ) and all(str(row.id) in answer for row in self.rows if row.required)

    def encode_answer(self, values: Mapping[UUID | str, str | None]) -> str:
        """The answer for `values` (row id -> option), in row order, blanks left out."""
        by_id = {str(key): value for key, value in values.items()}
        return json.dumps(
            {
                str(row.id): value
                for row in self.rows
                if (value := by_id.get(str(row.id)))
            },
            ensure_ascii=False,
        )

    def parse_answer(self, answer: str) -> dict[UUID, str]:
        """Row id -> chosen option of a stored answer, for the rows that still
        exist. Anything that is not such an answer reads as no ratings."""
        try:
            data = json.loads(answer)
        except ValueError:
            return {}
        if not isinstance(data, dict):
            return {}
        rows = {str(row.id): row.id for row in self.rows}
        return {
            rows[key]: value
            for key, value in data.items()
            if key in rows and isinstance(value, str)
        }

    def describe_answer(self, answer: str) -> str:
        """The answer as "row: option" lines, for transcripts and models.
        Text that is not a grid answer (a control token) is returned as is."""
        values = self.parse_answer(answer)
        if not values:
            return answer
        return "\n".join(
            f"{row.text}: {values[row.id]}" for row in self.rows if row.id in values
        )


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
    | LikertGridItem
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

        case LikertGridItem():
            # One field per row, named by position: row texts are not
            # identifiers. `grid_answer` maps them back to row ids.
            scale = Literal[tuple(survey_item.options)]  # ty:ignore[invalid-type-form]
            rows: dict[str, Any] = {
                f"row_{n}": (
                    scale if row.required else scale | None,
                    Field(... if row.required else None, description=row.text),
                )
                for n, row in enumerate(survey_item.rows)
            }
            field_type = create_model("GridAnswer", **rows)

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

    fields: dict[str, Any] = {"answer": (field_type, Field(**kwargs))}
    return create_model("SurveyAnswer", **fields)  # ty:ignore[invalid-return-type]


def grid_answer(item: LikertGridItem, answer: BaseModel) -> str:
    """The stored answer for a `create_survey_answer_model` grid answer."""
    return item.encode_answer(
        {row.id: getattr(answer, f"row_{n}", None) for n, row in enumerate(item.rows)}
    )
