from datetime import date, datetime, time

import pytest
from pydantic import ValidationError

from ainterviewer.interview_guides.survey_items import (
    CheckboxItem,
    DateItem,
    DatetimeItem,
    LikertItem,
    NumberItem,
    RadioItem,
    SliderItem,
    TimeItem,
    create_survey_answer_model,
)


# ── RadioItem ────────────────────────────────────────────────────────────────


class TestRadioItem:
    def test_validate_answer_valid(self):
        item = RadioItem(options=["a", "b", "c"])
        assert item.validate_answer("a") is True

    def test_validate_answer_invalid(self):
        item = RadioItem(options=["a", "b", "c"])
        assert item.validate_answer("d") is False

    def test_default_type(self):
        item = RadioItem(options=["x"])
        assert item.type == "radio"

    def test_with_other_default_false(self):
        item = RadioItem(options=["x"])
        assert item.with_other is False


# ── CheckboxItem ─────────────────────────────────────────────────────────────


class TestCheckboxItem:
    def test_validate_answer_valid(self):
        item = CheckboxItem(options=["a", "b", "c"])
        assert item.validate_answer(["a", "b"]) is True

    def test_validate_answer_invalid(self):
        item = CheckboxItem(options=["a", "b", "c"])
        assert item.validate_answer(["a", "d"]) is False

    def test_validate_answer_empty_list(self):
        item = CheckboxItem(options=["a", "b"])
        assert item.validate_answer([]) is True

    def test_default_type(self):
        item = CheckboxItem(options=["x"])
        assert item.type == "checkbox"

    def test_ui_field_not_accepted(self):
        with pytest.raises(ValidationError):
            CheckboxItem(options=["a"], ui="radio")  # ty:ignore[unknown-argument]


# ── LikertItem ───────────────────────────────────────────────────────────────


class TestLikertItem:
    def test_validate_answer_valid(self):
        item = LikertItem(
            options=[
                "Strongly disagree",
                "Disagree",
                "Neutral",
                "Agree",
                "Strongly agree",
            ]
        )
        assert item.validate_answer("Neutral") is True

    def test_validate_answer_invalid(self):
        item = LikertItem(
            options=[
                "Strongly disagree",
                "Disagree",
                "Neutral",
                "Agree",
                "Strongly agree",
            ]
        )
        assert item.validate_answer("Somewhat agree") is False

    def test_ui_default_radio(self):
        item = LikertItem(options=["low", "high"])
        assert item.ui == "radio"

    def test_ui_slider(self):
        item = LikertItem(options=["low", "high"], ui="slider")
        assert item.ui == "slider"

    def test_ui_invalid_value(self):
        with pytest.raises(ValidationError):
            LikertItem(options=["low", "high"], ui="dropdown")  # ty:ignore[invalid-argument-type]


# ── SliderItem ───────────────────────────────────────────────────────────────


class TestSliderItem:
    def test_requires_min_and_max(self):
        with pytest.raises(ValidationError):
            SliderItem()  # ty:ignore[missing-argument]

    def test_min_label_default_none(self):
        item = SliderItem(min=0, max=10)
        assert item.min_label is None

    def test_max_label_default_none(self):
        item = SliderItem(min=0, max=10)
        assert item.max_label is None

    def test_min_label_set(self):
        item = SliderItem(min=0, max=10, min_label="Low")
        assert item.min_label == "Low"

    def test_max_label_set(self):
        item = SliderItem(min=0, max=10, max_label="High")
        assert item.max_label == "High"

    def test_validate_within_range(self):
        item = SliderItem(min=0, max=10, step=1)
        assert item.validate_answer(5) is True

    def test_validate_below_min(self):
        item = SliderItem(min=0, max=10, step=1)
        assert item.validate_answer(0) is False  # <= min

    def test_validate_above_max(self):
        item = SliderItem(min=0, max=10, step=1)
        assert item.validate_answer(10) is False  # >= max

    def test_validate_wrong_step(self):
        item = SliderItem(min=0, max=10, step=2)
        assert item.validate_answer(3) is False

    def test_validate_correct_step(self):
        item = SliderItem(min=0, max=10, step=2)
        assert item.validate_answer(4) is True

    def test_validate_float_step(self):
        item = SliderItem(min=0, max=1, step=0.1)
        assert item.validate_answer(0.3) is True

    def test_validate_no_step_constraint(self):
        item = SliderItem(min=0, max=1000, step=None)
        assert item.validate_answer(999) is True


# ── NumberItem ───────────────────────────────────────────────────────────────


class TestNumberItem:
    def test_validate_within_range(self):
        item = NumberItem(min=1, max=100, step=1)
        assert item.validate_answer(50) is True

    def test_validate_below_min(self):
        item = NumberItem(min=1, max=100, step=1)
        assert item.validate_answer(1) is False  # <= min

    def test_validate_above_max(self):
        item = NumberItem(min=1, max=100, step=1)
        assert item.validate_answer(100) is False  # >= max

    def test_validate_wrong_step(self):
        item = NumberItem(min=0, max=100, step=5)
        assert item.validate_answer(3) is False

    def test_validate_no_constraints(self):
        item = NumberItem(min=None, max=None, step=None)
        assert item.validate_answer(-42) is True


# ── DateItem ─────────────────────────────────────────────────────────────────


class TestDateItem:
    def test_validate_within_range(self):
        item = DateItem(min="2020-01-01", max="2025-12-31")
        assert item.validate_answer("2023-06-15") is True

    def test_validate_before_min(self):
        item = DateItem(min="2020-01-01", max="2025-12-31")
        assert item.validate_answer("2020-01-01") is False  # <= min

    def test_validate_after_max(self):
        item = DateItem(min="2020-01-01", max="2025-12-31")
        assert item.validate_answer("2025-12-31") is False  # >= max

    def test_validate_no_constraints(self):
        item = DateItem()
        assert item.validate_answer("1900-01-01") is True


# ── DatetimeItem ─────────────────────────────────────────────────────────────


class TestDatetimeItem:
    def test_validate_within_range(self):
        item = DatetimeItem(min="2020-01-01T00:00:00", max="2025-12-31T23:59:59")
        assert item.validate_answer("2023-06-15T12:00:00") is True

    def test_validate_before_min(self):
        item = DatetimeItem(min="2020-01-01T00:00:00", max="2025-12-31T23:59:59")
        assert item.validate_answer("2020-01-01T00:00:00") is False

    def test_validate_after_max(self):
        item = DatetimeItem(min="2020-01-01T00:00:00", max="2025-12-31T23:59:59")
        assert item.validate_answer("2025-12-31T23:59:59") is False

    def test_validate_no_constraints(self):
        item = DatetimeItem()
        assert item.validate_answer("1999-01-01T00:00:00") is True


# ── TimeItem ─────────────────────────────────────────────────────────────────


class TestTimeItem:
    def test_validate_within_range(self):
        item = TimeItem(min="08:00:00", max="17:00:00")
        assert item.validate_answer("12:00:00") is True

    def test_validate_before_min(self):
        item = TimeItem(min="08:00:00", max="17:00:00")
        assert item.validate_answer("08:00:00") is False

    def test_validate_after_max(self):
        item = TimeItem(min="08:00:00", max="17:00:00")
        assert item.validate_answer("17:00:00") is False

    def test_validate_no_constraints(self):
        item = TimeItem()
        assert item.validate_answer("23:59:59") is True


# ── create_survey_answer_model ───────────────────────────────────────────────


class TestCreateSurveyAnswerModel:
    def test_radio_valid(self):
        item = RadioItem(options=["yes", "no"])
        Model = create_survey_answer_model(item)
        obj = Model(answer="yes")  # ty:ignore[call-non-callable]
        assert obj.answer == "yes"

    def test_radio_invalid(self):
        item = RadioItem(options=["yes", "no"])
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer="maybe")  # ty:ignore[call-non-callable]

    def test_radio_with_other(self):
        item = RadioItem(options=["yes", "no"], with_other=True)
        Model = create_survey_answer_model(item)
        obj = Model(answer="something else")  # ty:ignore[call-non-callable]
        assert obj.answer == "something else"

    def test_checkbox_valid(self):
        item = CheckboxItem(options=["a", "b", "c"])
        Model = create_survey_answer_model(item)
        obj = Model(answer=["a", "b"])  # ty:ignore[call-non-callable]
        assert obj.answer == ["a", "b"]

    def test_checkbox_invalid(self):
        item = CheckboxItem(options=["a", "b", "c"])
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer=["a", "d"])  # ty:ignore[call-non-callable]

    def test_checkbox_with_other(self):
        item = CheckboxItem(options=["a", "b"], with_other=True)
        Model = create_survey_answer_model(item)
        obj = Model(answer=["a", "custom"])  # ty:ignore[call-non-callable]
        assert obj.answer == ["a", "custom"]

    def test_likert_valid(self):
        item = LikertItem(options=["low", "mid", "high"])
        Model = create_survey_answer_model(item)
        obj = Model(answer="mid")  # ty:ignore[call-non-callable]
        assert obj.answer == "mid"

    def test_likert_invalid(self):
        item = LikertItem(options=["low", "mid", "high"])
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer="very high")  # ty:ignore[call-non-callable]

    def test_number_within_range(self):
        item = NumberItem(min=0, max=100)
        Model = create_survey_answer_model(item)
        obj = Model(answer=50)  # ty:ignore[call-non-callable]
        assert obj.answer == 50

    def test_number_below_min(self):
        item = NumberItem(min=0, max=100)
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer=-1)  # ty:ignore[call-non-callable]

    def test_number_above_max(self):
        item = NumberItem(min=0, max=100)
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer=101)  # ty:ignore[call-non-callable]

    def test_slider_within_range(self):
        item = SliderItem(min=1, max=10)
        Model = create_survey_answer_model(item)
        obj = Model(answer=5)  # ty:ignore[call-non-callable]
        assert obj.answer == 5

    def test_date_valid(self):
        item = DateItem(min="2020-01-01", max="2025-12-31")
        Model = create_survey_answer_model(item)
        obj = Model(answer=date(2023, 6, 15))  # ty:ignore[call-non-callable]
        assert obj.answer == date(2023, 6, 15)

    def test_date_out_of_range(self):
        item = DateItem(min="2020-01-01", max="2025-12-31")
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer=date(2019, 1, 1))  # ty:ignore[call-non-callable]

    def test_datetime_valid(self):
        item = DatetimeItem(min="2020-01-01T00:00:00", max="2025-12-31T23:59:59")
        Model = create_survey_answer_model(item)
        obj = Model(answer=datetime(2023, 6, 15, 12, 0, 0))  # ty:ignore[call-non-callable]
        assert obj.answer == datetime(2023, 6, 15, 12, 0, 0)

    def test_time_valid(self):
        item = TimeItem(min="08:00:00", max="17:00:00")
        Model = create_survey_answer_model(item)
        obj = Model(answer=time(12, 0, 0))  # ty:ignore[call-non-callable]
        assert obj.answer == time(12, 0, 0)

    def test_optional_field_allows_none(self):
        item = RadioItem(options=["yes", "no"], required=False)
        Model = create_survey_answer_model(item)
        obj = Model(answer=None)  # ty:ignore[call-non-callable]
        assert obj.answer is None

    def test_optional_field_defaults_to_none(self):
        item = RadioItem(options=["yes", "no"], required=False)
        Model = create_survey_answer_model(item)
        obj = Model()  # ty:ignore[call-non-callable]
        assert obj.answer is None

    def test_optional_field_allows_value(self):
        item = RadioItem(options=["yes", "no"], required=False)
        Model = create_survey_answer_model(item)
        obj = Model(answer="yes")  # ty:ignore[call-non-callable]
        assert obj.answer == "yes"

    def test_required_field_rejects_none(self):
        item = RadioItem(options=["yes", "no"], required=True)
        Model = create_survey_answer_model(item)
        with pytest.raises(ValidationError):
            Model(answer=None)  # ty:ignore[call-non-callable]
