from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, time
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
    validate_call,
)

#: What a checkbox selection is joined with when it is submitted as one answer.
CHECKBOX_SEPARATOR = ", "


class SurveyItemBase(BaseModel):
    """What every survey item shares.

    An answer is stored as one string, whatever its item. Most items store
    the answer as the respondent gave it; an item whose answer has structure
    (a grid, a list) stores an encoding of it, and overrides the methods here
    so that callers never need to know which items do.
    """

    model_config = ConfigDict(extra="forbid")

    required: bool = True

    def describe_answer(self, answer: str) -> str:
        """The stored answer as text a person or model can read.

        Text that is not an answer of this item (a control token) is returned
        as is."""
        return answer

    def generated_answer(self, answer: Any) -> str:
        """The stored answer for the `answer` field of a
        `create_survey_answer_model` response."""
        return str(answer)

    def draw_order(self) -> None:
        """Draw this interview's random order for whatever the item shuffles,
        or clear it when shuffling is off. Most items shuffle nothing."""


validate_strict = validate_call(config=ConfigDict(strict=True))


class ShuffleableOptions(SurveyItemBase):
    """A choice item whose options may be shown in a random order.

    The order is drawn once per interview (`InterviewGuide.shuffle`) and kept
    beside `options` rather than applied to it: `options` stays in authored
    order, because analysis places an answer by its position there -- that is
    how the same choice in two languages counts as one. "Other" is not among
    the options, and stays last.
    """

    options: list[str]

    shuffle_options: bool = Field(
        False,
        description="Show the options in a random order, drawn once per interview.",
    )
    option_order: list[int] | None = Field(
        None,
        description="The order the options are shown in, as positions in "
        "`options`. Set per interview when `shuffle_options` is on; unset, "
        "they are shown as authored.",
    )

    @model_validator(mode="after")
    def _check_option_order(self):
        if self.option_order is not None and sorted(self.option_order) != list(
            range(len(self.options))
        ):
            raise ValueError(
                "option_order must hold each position in options exactly once"
            )
        return self

    def draw_order(self) -> None:
        """Draw this interview's order, or clear it when shuffling is off."""
        self.option_order = (
            random.sample(range(len(self.options)), len(self.options))
            if self.shuffle_options
            else None
        )

    @property
    def shown_options(self) -> list[str]:
        """The options in the order the respondent sees them."""
        if self.option_order is None:
            return list(self.options)
        return [self.options[n] for n in self.option_order]


# TODO:
# Let options be a keyword of {label: value} pairs
# Make labels and values referable in the next question
class RadioItem(ShuffleableOptions):
    type: Literal["radio"] = "radio"

    with_other: bool = False

    @validate_strict
    def validate_answer(self, answer: str):
        if self.with_other:
            return True

        return answer in self.options


class CheckboxItem(ShuffleableOptions):
    type: Literal["checkbox"] = "checkbox"

    with_other: bool = False

    @validate_strict
    def validate_answer(self, answer: list[str]):
        if self.with_other:
            return True

        return all(x in self.options for x in answer)

    def generated_answer(self, answer: Any) -> str:
        if isinstance(answer, list):
            return CHECKBOX_SEPARATOR.join(answer)
        return super().generated_answer(answer)


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

    shuffle_rows: bool = Field(
        False,
        description="Show the rows in a random order, drawn once per interview.",
    )
    row_order: list[int] | None = Field(
        None,
        description="The order the rows are shown in, as positions in `rows`. "
        "Set per interview when `shuffle_rows` is on; unset, they are shown "
        "as authored.",
    )

    @model_validator(mode="after")
    def _check_row_order(self):
        if self.row_order is not None and sorted(self.row_order) != list(
            range(len(self.rows))
        ):
            raise ValueError("row_order must hold each position in rows exactly once")
        return self

    def draw_order(self) -> None:
        """Draw this interview's row order, or clear it when shuffling is off.

        Like `ShuffleableOptions.option_order`, the order is kept beside
        `rows` rather than applied to it: the answer, its description and
        analysis all read the rows in authored order."""
        self.row_order = (
            random.sample(range(len(self.rows)), len(self.rows))
            if self.shuffle_rows
            else None
        )

    @property
    def shown_rows(self) -> list[GridRow]:
        """The rows in the order the respondent sees them."""
        if self.row_order is None:
            return list(self.rows)
        return [self.rows[n] for n in self.row_order]

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

    def generated_answer(self, answer: Any) -> str:
        """The stored answer for a `create_survey_answer_model` grid answer,
        whose fields are the rows by position."""
        return self.encode_answer(
            {
                row.id: getattr(answer, f"row_{n}", None)
                for n, row in enumerate(self.rows)
            }
        )


def _json_list(answer: str) -> list | None:
    """The JSON array `answer` holds, or None when it holds none."""
    try:
        data = json.loads(answer)
    except ValueError:
        return None
    return data if isinstance(data, list) else None


class ListItem(SurveyItemBase):
    """Several answers of the respondent's own, given together as one answer:
    "name the apps you use daily".

    `min_entries` and `max_entries` bound how many; set them equal for a fixed
    number of entries. The answer is a JSON array of the entries, in the order
    given, blanks left out (`encode_answer`), read back with `parse_answer` --
    or as text a person or model can read with `describe_answer`.
    """

    type: Literal["list"] = "list"

    min_entries: int | None = Field(
        None,
        ge=1,
        description="The fewest entries the answer may hold; unset, one when "
        "the item is required.",
    )
    max_entries: int | None = Field(
        None,
        ge=1,
        description="The most entries the answer may hold; unset, no limit. "
        "Equal to `min_entries`, the respondent fills a fixed number.",
    )

    @model_validator(mode="after")
    def _check_bounds(self):
        if (
            self.min_entries is not None
            and self.max_entries is not None
            and self.min_entries > self.max_entries
        ):
            raise ValueError("min_entries must not exceed max_entries")
        return self

    @property
    def fewest(self) -> int:
        """The fewest entries a submitted answer holds."""
        return self.min_entries or (1 if self.required else 0)

    @validate_strict
    def validate_answer(self, answer: list[str]):
        entries = [entry for entry in answer if entry.strip()]
        return len(entries) == len(answer) and _within(
            len(entries), self.fewest, self.max_entries
        )

    def encode_answer(self, entries: Sequence[str]) -> str:
        """The answer for `entries`, trimmed, blanks left out."""
        return json.dumps(
            [entry.strip() for entry in entries if entry.strip()],
            ensure_ascii=False,
        )

    def parse_answer(self, answer: str) -> list[str]:
        """The entries of a stored answer. Anything that is not such an
        answer reads as no entries."""
        data = _json_list(answer) or []
        return [entry for entry in data if isinstance(entry, str)]

    def describe_answer(self, answer: str) -> str:
        """The answer as one "- entry" line per entry, for transcripts and
        models. Text that is not a list answer (a control token) is returned
        as is."""
        entries = self.parse_answer(answer)
        if not entries:
            return answer
        return "\n".join(f"- {entry}" for entry in entries)

    def generated_answer(self, answer: Any) -> str:
        if isinstance(answer, list):
            return self.encode_answer(answer)
        return super().generated_answer(answer)


class RankingItem(ShuffleableOptions):
    """The options put in order of preference, answered as one answer.

    With `top_n` set, only that many are ranked: the answer is the respondent's
    first `top_n` choices, and the rest are left unranked. The answer is a JSON
    array of the ranked options' texts, first choice first (`encode_answer`),
    read back with `parse_answer`, as positions in `options` with
    `ranked_positions` (which is how one choice counts as one across
    languages), or as text with `describe_answer`.
    """

    type: Literal["ranking"] = "ranking"

    top_n: int | None = Field(
        None,
        ge=1,
        description="Rank only the first this many options; unset, rank all.",
    )

    @property
    def ranked_count(self) -> int:
        """How many options an answer ranks. A `top_n` beyond the options
        ranks them all."""
        if self.top_n is None:
            return len(self.options)
        return min(self.top_n, len(self.options))

    @validate_strict
    def validate_answer(self, answer: list[str]):
        return (
            len(answer) == self.ranked_count
            and len(set(answer)) == len(answer)
            and all(option in self.options for option in answer)
        )

    def encode_answer(self, ranked: Sequence[str]) -> str:
        """The answer for `ranked`, first choice first."""
        return json.dumps(list(ranked), ensure_ascii=False)

    def parse_answer(self, answer: str) -> list[str]:
        """The ranked options of a stored answer, first choice first, leaving
        out any that is not (or no longer) an option. Anything that is not
        such an answer reads as no ranking."""
        data = _json_list(answer) or []
        seen: set[str] = set()
        ranked = []
        for option in data:
            if (
                isinstance(option, str)
                and option in self.options
                and option not in seen
            ):
                seen.add(option)
                ranked.append(option)
        return ranked

    def ranked_positions(self, answer: str) -> list[int]:
        """The ranked options of a stored answer as positions in `options`,
        first choice first."""
        position = {option: n for n, option in enumerate(self.options)}
        return [position[option] for option in self.parse_answer(answer)]

    def describe_answer(self, answer: str) -> str:
        """The answer as "1. option" lines, for transcripts and models. Text
        that is not a ranking answer (a control token) is returned as is."""
        ranked = self.parse_answer(answer)
        if not ranked:
            return answer
        return "\n".join(f"{n}. {option}" for n, option in enumerate(ranked, 1))

    def generated_answer(self, answer: Any) -> str:
        """A generated ranking, a repeated option kept only where it first
        appears."""
        if isinstance(answer, list):
            return self.encode_answer(list(dict.fromkeys(answer))[: self.ranked_count])
        return super().generated_answer(answer)


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
    | ListItem
    | RankingItem
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
            # identifiers. `generated_answer` maps them back to row ids.
            scale = Literal[tuple(survey_item.options)]  # ty:ignore[invalid-type-form]
            rows: dict[str, Any] = {
                f"row_{n}": (
                    scale if row.required else scale | None,
                    Field(... if row.required else None, description=row.text),
                )
                for n, row in enumerate(survey_item.rows)
            }
            field_type = create_model("GridAnswer", **rows)

        case ListItem():
            field_type = list[str]
            kwargs["min_length"] = survey_item.fewest
            if survey_item.max_entries is not None:
                kwargs["max_length"] = survey_item.max_entries

        case RankingItem():
            field_type = list[Literal[tuple(survey_item.options)]]  # ty:ignore[invalid-type-form]
            kwargs["min_length"] = kwargs["max_length"] = survey_item.ranked_count
            kwargs["description"] = (
                "The options in order of preference, first choice first, "
                "each at most once."
            )

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
