"""Likert grids: several statements rated on one scale, as one question.

The grid is a survey item like any other -- one question, one answer, probed
like any other question -- so what is tested here is what the item adds: the
answer it takes (JSON keyed by row id), how that answer reads back for people
and models, and its text across languages.
"""

import json
import uuid

import pytest

from ainterviewer.interview_guides import (
    GridRow,
    InterviewGuide,
    LikertGridItem,
    LocalizedInterviewGuide,
)
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    QuestionHistory,
    Turn,
)
from ainterviewer.interview_guides.localized import LocalizedLikertGridItem
from ainterviewer.interview_guides.survey_items import (
    create_survey_answer_model,
    grid_answer,
)

SCALE = ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied"]
ROWS = [
    GridRow(id=uuid.UUID(int=1), text="Cleanliness"),
    GridRow(id=uuid.UUID(int=2), text="Signage"),
    GridRow(id=uuid.UUID(int=3), text="Food", required=False),
]


def _item() -> LikertGridItem:
    return LikertGridItem(options=SCALE, rows=[row.model_copy() for row in ROWS])


# ── The answer ───────────────────────────────────────────────────────────────


def test_encode_answer_is_json_keyed_by_row_in_row_order_without_blanks():
    item = _item()
    answer = item.encode_answer(
        {ROWS[1].id: "Neutral", str(ROWS[0].id): "Satisfied", ROWS[2].id: None}
    )
    assert json.loads(answer) == {
        str(ROWS[0].id): "Satisfied",
        str(ROWS[1].id): "Neutral",
    }
    assert list(json.loads(answer)) == [str(ROWS[0].id), str(ROWS[1].id)]


def test_parse_answer_round_trips_and_drops_rows_that_no_longer_exist():
    item = _item()
    answer = json.dumps({str(ROWS[0].id): "Satisfied", str(uuid.uuid4()): "Neutral"})
    assert item.parse_answer(answer) == {ROWS[0].id: "Satisfied"}


@pytest.mark.parametrize("answer", ["<|skipquestion|>", "[1, 2]", "", "{"])
def test_parse_answer_reads_anything_else_as_no_ratings(answer):
    assert _item().parse_answer(answer) == {}


def test_describe_answer_reads_as_row_and_option_lines():
    item = _item()
    answer = item.encode_answer({ROWS[0].id: "Satisfied", ROWS[1].id: "Neutral"})
    assert item.describe_answer(answer) == "Cleanliness: Satisfied\nSignage: Neutral"
    assert item.describe_answer("<|skipquestion|>") == "<|skipquestion|>"


def test_validate_answer_needs_every_required_row_and_known_options():
    item = _item()
    one, two, three = (str(row.id) for row in ROWS)
    assert item.validate_answer({one: "Neutral", two: "Satisfied"})
    assert item.validate_answer({one: "Neutral", two: "Satisfied", three: "Neutral"})
    assert not item.validate_answer({one: "Neutral"})
    assert not item.validate_answer({one: "Neutral", two: "Great"})
    assert not item.validate_answer({one: "Neutral", two: "Neutral", "x": "Neutral"})


def test_a_generated_answer_becomes_the_stored_json():
    item = _item()
    model = create_survey_answer_model(item)
    schema = json.dumps(model.model_json_schema())
    assert "Cleanliness" in schema and "Signage" in schema

    response = model.model_validate(
        {"answer": {"row_0": "Satisfied", "row_1": "Neutral"}}
    )
    assert item.parse_answer(grid_answer(item, response.answer)) == {
        ROWS[0].id: "Satisfied",
        ROWS[1].id: "Neutral",
    }


def test_a_generated_answer_must_rate_required_rows():
    model = create_survey_answer_model(_item())
    with pytest.raises(ValueError):
        model.model_validate({"answer": {"row_0": "Satisfied"}})


# ── The history ──────────────────────────────────────────────────────────────


def test_the_transcript_shows_the_ratings_not_the_json():
    item = _item()
    answer = item.encode_answer({ROWS[0].id: "Dissatisfied", ROWS[1].id: "Neutral"})
    question = QuestionHistory(
        description=None,
        main_question=Turn(
            question=HistoryMessage(message="How satisfied were you?"),
            answer=HistoryMessage(message=answer),
            survey_item=item,
        ),
    )
    assert question.transcribe() == (
        "Q: How satisfied were you?\nA: Cleanliness: Dissatisfied\nSignage: Neutral"
    )


# ── The guide ────────────────────────────────────────────────────────────────


def _guide(item: LikertGridItem, **question) -> InterviewGuide:
    return InterviewGuide.model_validate(
        {
            "question_sections": [
                {
                    "description": "Airport",
                    "questions": [
                        {
                            "id": str(uuid.UUID(int=10)),
                            "main_question": "How satisfied were you with…",
                            "survey_item": item.model_dump(mode="json"),
                            **question,
                        },
                        {"main_question": "Anything else?", "max_probes_n": 0},
                    ],
                }
            ]
        }
    )


def test_a_grid_can_be_probed():
    question = _guide(_item(), max_probes_n=2).question_sections[0].questions[0]
    assert question.max_probes_n == 2


def test_a_grid_is_not_probed_by_default():
    assert _guide(_item()).question_sections[0].questions[0].max_probes_n == 0


def _localized() -> LocalizedInterviewGuide:
    english = _guide(_item())
    danish = english.model_copy(deep=True)
    grid = danish.question_sections[0].questions[0].survey_item
    assert isinstance(grid, LikertGridItem)
    grid.options = [f"{option} (DA)" for option in SCALE]
    for row in grid.rows:
        row.text += " (DA)"
    return LocalizedInterviewGuide.from_rendered(
        {"EN": english, "DA": danish}, structure="EN"
    )


def test_rows_keep_their_ids_and_take_text_per_language():
    guide = _localized()
    rendered = guide.render("DA").question_sections[0].questions[0].survey_item
    assert isinstance(rendered, LikertGridItem)
    assert [row.id for row in rendered.rows] == [row.id for row in ROWS]
    assert [row.text for row in rendered.rows] == [
        "Cleanliness (DA)",
        "Signage (DA)",
        "Food (DA)",
    ]
    assert [row.required for row in rendered.rows] == [True, True, False]
    assert rendered.options[0] == "Very dissatisfied (DA)"


def test_untranslated_rows_and_options_are_reported_missing():
    guide = _localized()
    grid = guide.question_sections[0].questions[0].survey_item
    assert isinstance(grid, LocalizedLikertGridItem)
    del grid.rows[1].text["DA"]
    del grid.options[0].text["DA"]
    assert guide.missing("DA") == [
        "question_sections[0].questions[0].survey_item.options[0]",
        "question_sections[0].questions[0].survey_item.rows[1].text",
    ]


def test_a_condition_cannot_read_a_grid():
    data = _localized().model_dump(mode="json")
    data["question_sections"][0]["questions"][1]["conditions"] = {
        "action": "skip_question",
        "conditions": [
            {
                "question_context": {
                    "question_index": [0, 0],
                    "question_id": str(uuid.UUID(int=10)),
                },
                "evaluation": [{"trigger_value": "x"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="Likert grid"):
        LocalizedInterviewGuide.model_validate(data)


def test_duplicate_row_ids_are_refused():
    data = _localized().model_dump(mode="json")
    rows = data["question_sections"][0]["questions"][0]["survey_item"]["rows"]
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        LocalizedInterviewGuide.model_validate(data)
