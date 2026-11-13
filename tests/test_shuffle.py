"""Shuffling that keeps conditions answerable.

A condition reads the answer to another question, so that question has to be
asked first. The shuffle draws again until it is, instead of repairing the
order, so the valid orders stay equally likely; these pin both halves of that,
and the save-time check that keeps the fallback order valid.
"""

import random
from collections import Counter
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    Conditions,
    QuestionContext,
)
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.shuffle import shuffle_items
from ainterviewer.interview_guides.survey_items import (
    CheckboxItem,
    LikertItem,
    RadioItem,
)
from ainterviewer.interview_guides.types import ConditionAction


@pytest.fixture(autouse=True)
def _seed():
    random.seed(1234)


@dataclass
class Item:
    name: str
    shuffle: bool = True


def _gated(name: str, read: Question | UUID, *, shuffle: bool = True) -> Question:
    """A question with a condition reading `read`; pass a bare id to give the
    gated question that id, making the condition read its own question."""
    own_id = read if isinstance(read, UUID) else uuid4()
    return Question(
        id=own_id,
        main_question=name,
        shuffle=shuffle,
        conditions=Conditions(
            action=ConditionAction.SKIP_QUESTION,
            conditions=[
                Condition(
                    question_context=QuestionContext(
                        question_index=(0, 0),
                        question_id=read if isinstance(read, UUID) else read.id,
                    ),
                    evaluation=[ConditionEvaluation(trigger_value="no")],
                )
            ],
        ),
    )


def _names(items) -> tuple[str, ...]:
    return tuple(item.name for item in items)


def _questions(section: QuestionSection) -> tuple[str, ...]:
    return tuple(question.main_question for question in section.questions)


# ── shuffle_items ───────────────────────────────────────────────────────────


def test_fixed_items_keep_their_slot():
    items = [Item("a"), Item("fixed", shuffle=False), Item("b"), Item("c")]
    for _ in range(50):
        assert shuffle_items(items)[1].name == "fixed"


def test_constraint_holds_and_valid_orders_are_equally_likely():
    a, b, c = Item("a"), Item("b"), Item("c")
    draws = Counter(
        _names(shuffle_items([a, b, c], must_precede={(0, 1)})) for _ in range(3000)
    )

    assert all(order.index("a") < order.index("b") for order in draws)
    # Three of the six orders put a before b; each should come up about a third
    # of the time. A repairing shuffle would skew this.
    assert len(draws) == 3
    assert all(800 < count < 1200 for count in draws.values())


def test_unsatisfiable_constraints_keep_the_given_order():
    items = [Item("a"), Item("b")]
    assert _names(shuffle_items(items, must_precede={(0, 1), (1, 0)}, attempts=20)) == (
        "a",
        "b",
    )


# ── InterviewGuide.shuffle ──────────────────────────────────────────────────


def test_question_read_by_a_condition_stays_before_it():
    read = Question(main_question="read", shuffle=True)
    others = [Question(main_question=f"q{i}", shuffle=True) for i in range(3)]
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(
                description="s", questions=[read, *others, _gated("gated", read)]
            )
        ]
    )

    seen = set()
    for _ in range(200):
        guide.shuffle()
        order = _questions(guide.question_sections[0])
        assert order.index("read") < order.index("gated")
        seen.add(order)
    assert len(seen) > 1


def test_section_holding_a_read_question_stays_before_the_gated_one():
    read = Question(main_question="read")
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="first", shuffle=True, questions=[read]),
            QuestionSection(
                description="middle",
                shuffle=True,
                questions=[Question(main_question="x")],
            ),
            QuestionSection(
                description="last",
                shuffle=True,
                questions=[_gated("gated", read, shuffle=False)],
            ),
        ]
    )

    seen = set()
    for _ in range(200):
        guide.shuffle()
        order = [section.description for section in guide.question_sections]
        assert order.index("first") < order.index("last")
        seen.add(tuple(order))
    assert len(seen) == 3


def test_condition_reading_its_own_question_does_not_constrain():
    first = Question(main_question="first", shuffle=True)
    own = _gated("own", uuid4(), shuffle=True)
    guide = InterviewGuide(
        question_sections=[QuestionSection(description="s", questions=[first, own])]
    )

    seen = set()
    for _ in range(100):
        guide.shuffle()
        seen.add(_questions(guide.question_sections[0]))
    assert seen == {("first", "own"), ("own", "first")}


# ── InterviewGuide.condition_order_errors ───────────────────────────────────


def test_condition_reading_a_later_question_is_reported():
    read = Question(main_question="read")
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="s", questions=[_gated("gated", read), read])
        ]
    )

    [error] = guide.condition_order_errors()
    assert error.message == (
        "Question 1.1 has a condition on question 1.2, which is asked after it."
    )
    assert error.fix == "Move question 1.2 above question 1.1, or change the condition."
    assert error.question_id == guide.question_sections[0].questions[0].id
    assert error.field == "conditions"


def test_conditions_reading_earlier_or_own_questions_are_valid():
    read = Question(main_question="read")
    own = _gated("own", uuid4())
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="s", questions=[read, _gated("gated", read)]),
            QuestionSection(description="t", questions=[own]),
        ]
    )

    assert guide.condition_order_errors() == []


# ── Option order ────────────────────────────────────────────────────────────


def _item(guide: InterviewGuide):
    item = guide.question_sections[0].questions[0].survey_item
    assert isinstance(item, (RadioItem, CheckboxItem, LikertItem))
    return item


def _one_question_guide(item) -> InterviewGuide:
    return InterviewGuide(
        question_sections=[
            QuestionSection(
                description="s",
                questions=[Question(main_question="q", survey_item=item)],
            )
        ]
    )


@pytest.mark.parametrize("kind", [RadioItem, CheckboxItem])
def test_shuffled_options_get_an_order_and_keep_their_authored_list(kind):
    options = ["a", "b", "c", "d"]
    guide = _one_question_guide(kind(options=options, shuffle_options=True))

    seen = set()
    for _ in range(100):
        guide.shuffle()
        item = _item(guide)
        assert not isinstance(item, LikertItem)
        assert item.option_order is not None
        # Analysis places answers by position in `options`.
        assert item.options == options
        assert sorted(item.option_order) == [0, 1, 2, 3]
        assert item.shown_options == [options[n] for n in item.option_order]
        seen.add(tuple(item.option_order))
    assert len(seen) > 1


def test_options_not_asked_to_shuffle_keep_no_order():
    guide = _one_question_guide(RadioItem(options=["a", "b"], option_order=[1, 0]))
    guide.shuffle()
    item = _item(guide)
    assert not isinstance(item, LikertItem)
    assert item.option_order is None
    assert item.shown_options == ["a", "b"]


def test_likert_scale_is_never_shuffled():
    guide = _one_question_guide(LikertItem(options=["1", "2", "3"]))
    guide.shuffle()
    assert _item(guide).options == ["1", "2", "3"]
