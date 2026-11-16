"""List and ranking items: answers with structure, stored as one JSON string.

A list item takes several answers of the respondent's own; a ranking item puts
the item's options in order. What is tested here is what each adds: the answer
it takes, how that answer reads back for people and models, its text across
languages, and (for a ranking) its option order.
"""

import json
import uuid

import pytest

from ainterviewer.interview_guides import (
    InterviewGuide,
    ListItem,
    LocalizedInterviewGuide,
    RankingItem,
)
from ainterviewer.interview_guides.history import HistoryMessage, QuestionHistory, Turn
from ainterviewer.interview_guides.localized import LocalizedRankingItem
from ainterviewer.interview_guides.survey_items import create_survey_answer_model

SKIP = "<|skipquestion|>"
OPTIONS = ["Price", "Quality", "Delivery time", "Brand"]


def _transcribe(item, answer: str) -> str:
    return QuestionHistory(
        description=None,
        main_question=Turn(
            question=HistoryMessage(message="Q?"),
            answer=HistoryMessage(message=answer),
            survey_item=item,
        ),
    ).transcribe()


# ── List: the item ───────────────────────────────────────────────────────────


def test_list_bounds_must_be_in_order():
    ListItem(min_entries=3, max_entries=3)
    with pytest.raises(ValueError, match="min_entries"):
        ListItem(min_entries=4, max_entries=3)
    with pytest.raises(ValueError):
        ListItem(min_entries=0)


@pytest.mark.parametrize(
    ("item", "fewest"),
    [
        (ListItem(), 1),
        (ListItem(required=False), 0),
        (ListItem(min_entries=2, required=False), 2),
    ],
)
def test_list_fewest_entries(item, fewest):
    assert item.fewest == fewest


def test_list_validate_answer_counts_entries_within_bounds():
    item = ListItem(min_entries=2, max_entries=3)
    assert item.validate_answer(["a", "b"])
    assert item.validate_answer(["a", "b", "c"])
    assert not item.validate_answer(["a"])
    assert not item.validate_answer(["a", "b", "c", "d"])
    assert not item.validate_answer(["a", " "])


# ── List: the answer ─────────────────────────────────────────────────────────


def test_list_encode_answer_trims_and_drops_blanks_in_order():
    answer = ListItem().encode_answer([" Slack ", "", "Figma", "  "])
    assert json.loads(answer) == ["Slack", "Figma"]


def test_list_parse_answer_round_trips():
    item = ListItem()
    assert item.parse_answer(item.encode_answer(["Æble", "Figma"])) == ["Æble", "Figma"]


@pytest.mark.parametrize("answer", [SKIP, "{}", "", "[", "[1, 2]"])
def test_list_parse_answer_reads_anything_else_as_no_entries(answer):
    assert ListItem().parse_answer(answer) == []


def test_list_describe_answer_reads_as_bullet_lines():
    item = ListItem()
    assert item.describe_answer(item.encode_answer(["Slack", "Figma"])) == (
        "- Slack\n- Figma"
    )
    assert item.describe_answer(SKIP) == SKIP


def test_list_transcript_shows_the_entries_not_the_json():
    item = ListItem()
    assert _transcribe(item, item.encode_answer(["Slack", "Figma"])) == (
        "Q: Q?\nA: - Slack\n- Figma"
    )


def test_list_generated_answer_is_bounded_and_stored_as_json():
    item = ListItem(min_entries=2, max_entries=3)
    model = create_survey_answer_model(item)
    response = model.model_validate({"answer": ["Slack", "Figma"]})
    assert item.parse_answer(item.generated_answer(response.answer)) == [
        "Slack",
        "Figma",
    ]
    with pytest.raises(ValueError):
        model.model_validate({"answer": ["Slack"]})
    with pytest.raises(ValueError):
        model.model_validate({"answer": ["a", "b", "c", "d"]})


def test_an_optional_list_may_be_generated_empty():
    model = create_survey_answer_model(ListItem(required=False, max_entries=2))
    assert model.model_validate({"answer": None}).answer is None
    assert model.model_validate({"answer": []}).answer == []


# ── Ranking: the item ────────────────────────────────────────────────────────


def _ranking(**kwargs) -> RankingItem:
    return RankingItem(options=list(OPTIONS), **kwargs)


@pytest.mark.parametrize(("top_n", "count"), [(None, 4), (2, 2), (9, 4)])
def test_ranking_ranks_all_or_the_top_n(top_n, count):
    assert _ranking(top_n=top_n).ranked_count == count


def test_ranking_validate_answer_needs_distinct_known_options_of_the_right_count():
    item = _ranking(top_n=2)
    assert item.validate_answer(["Brand", "Price"])
    assert not item.validate_answer(["Brand"])
    assert not item.validate_answer(["Brand", "Brand"])
    assert not item.validate_answer(["Brand", "Colour"])
    assert not item.validate_answer(["Brand", "Price", "Quality"])
    assert _ranking().validate_answer(list(reversed(OPTIONS)))


# ── Ranking: the answer ──────────────────────────────────────────────────────


def test_ranking_parse_answer_round_trips_and_drops_unknown_and_repeated_options():
    item = _ranking()
    assert item.parse_answer(item.encode_answer(["Brand", "Price"])) == [
        "Brand",
        "Price",
    ]
    assert item.parse_answer(json.dumps(["Brand", "Colour", "Brand", "Price"])) == [
        "Brand",
        "Price",
    ]


@pytest.mark.parametrize("answer", [SKIP, "{}", "", "[", "[1]"])
def test_ranking_parse_answer_reads_anything_else_as_no_ranking(answer):
    assert _ranking().parse_answer(answer) == []


def test_ranking_positions_are_places_in_the_authored_options():
    item = _ranking()
    assert item.ranked_positions(item.encode_answer(["Brand", "Price"])) == [3, 0]


def test_ranking_describe_answer_reads_as_numbered_lines():
    item = _ranking()
    assert item.describe_answer(item.encode_answer(["Brand", "Price"])) == (
        "1. Brand\n2. Price"
    )
    assert item.describe_answer(SKIP) == SKIP


def test_ranking_generated_answer_ranks_each_option_once():
    item = _ranking(top_n=2)
    model = create_survey_answer_model(item)
    assert "first choice first" in json.dumps(model.model_json_schema())
    response = model.model_validate({"answer": ["Quality", "Quality"]})
    assert item.parse_answer(item.generated_answer(response.answer)) == ["Quality"]
    with pytest.raises(ValueError):
        model.model_validate({"answer": ["Quality", "Colour"]})
    with pytest.raises(ValueError):
        model.model_validate({"answer": ["Quality"]})


# ── The guide ────────────────────────────────────────────────────────────────


def _guide(item) -> InterviewGuide:
    return InterviewGuide.model_validate(
        {
            "question_sections": [
                {
                    "description": "Shopping",
                    "questions": [
                        {
                            "id": str(uuid.UUID(int=10)),
                            "main_question": "Rank these",
                            "survey_item": item.model_dump(mode="json"),
                        },
                        {"main_question": "Anything else?", "max_probes_n": 0},
                    ],
                }
            ]
        }
    )


def _localized(item) -> LocalizedInterviewGuide:
    english = _guide(item)
    danish = english.model_copy(deep=True)
    rendered = danish.question_sections[0].questions[0].survey_item
    if isinstance(rendered, RankingItem):
        rendered.options = [f"{option} (DA)" for option in OPTIONS]
    return LocalizedInterviewGuide.from_rendered(
        {"EN": english, "DA": danish}, structure="EN"
    )


def test_ranking_options_take_text_per_language_and_keep_settings():
    guide = _localized(_ranking(top_n=2, shuffle_options=True))
    local = guide.question_sections[0].questions[0].survey_item
    assert isinstance(local, LocalizedRankingItem)
    assert local.top_n == 2
    rendered = guide.render("DA").question_sections[0].questions[0].survey_item
    assert isinstance(rendered, RankingItem)
    assert rendered.options[0] == "Price (DA)"
    assert rendered.top_n == 2 and rendered.shuffle_options


def test_untranslated_ranking_options_are_reported_missing():
    guide = _localized(_ranking())
    local = guide.question_sections[0].questions[0].survey_item
    assert isinstance(local, LocalizedRankingItem)
    del local.options[1].text["DA"]
    assert guide.missing("DA") == [
        "question_sections[0].questions[0].survey_item.options[1]"
    ]


def test_duplicate_ranking_option_ids_are_refused():
    data = _localized(_ranking()).model_dump(mode="json")
    options = data["question_sections"][0]["questions"][0]["survey_item"]["options"]
    options[1]["id"] = options[0]["id"]
    with pytest.raises(ValueError, match="Duplicate"):
        LocalizedInterviewGuide.model_validate(data)


def test_a_list_round_trips_through_the_localized_guide():
    item = ListItem(min_entries=2, max_entries=4, required=False)
    rendered = (
        _localized(item).render("DA").question_sections[0].questions[0].survey_item
    )
    assert rendered == item


def test_shuffled_ranking_options_get_an_order_and_keep_their_authored_list():
    guide = _guide(_ranking(shuffle_options=True))
    orders = set()
    for _ in range(50):
        guide.shuffle()
        item = guide.question_sections[0].questions[0].survey_item
        assert isinstance(item, RankingItem)
        assert item.options == OPTIONS
        assert item.option_order is not None
        orders.add(tuple(item.option_order))
    assert len(orders) > 1
