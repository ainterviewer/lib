"""Stable ids on sections and questions, and the refs that point at them.

Conditions and references name the question they read by id; the position the
interview loop reads answers back by is derived from it. These pin that the id
wins over a stale position, for conditions and references alike, and that a
shuffle moves the derived position along with the question.
"""

import random
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    Conditions,
    QuestionContext,
)
from ainterviewer.interview_guides.references import Reference
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.types import ConditionAction


def _gated(target: Question, *, section: int = 0, question: int = 0) -> Question:
    """A question whose condition reads `target`, at a deliberately given
    position so tests can make it stale."""
    return Question(
        main_question="Gated",
        conditions=Conditions(
            action=ConditionAction.SKIP_QUESTION,
            conditions=[
                Condition(
                    question_context=QuestionContext(
                        question_index=(section, question), question_id=target.id
                    ),
                    evaluation=[ConditionEvaluation(trigger_value="no")],
                )
            ],
        ),
    )


def _context(guide: InterviewGuide, section: int, question: int) -> QuestionContext:
    conditions = guide.question_sections[section].questions[question].conditions
    assert conditions is not None
    return conditions.conditions[0].question_context


def test_ids_are_assigned_and_survive_a_round_trip():
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="s", questions=[Question(main_question="q")])
        ]
    )
    reloaded = InterviewGuide.model_validate(guide.model_dump(mode="json"))

    assert reloaded.question_sections[0].id == guide.question_sections[0].id
    assert (
        reloaded.question_sections[0].questions[0].id
        == guide.question_sections[0].questions[0].id
    )


def test_id_wins_over_a_stale_position():
    target = Question(main_question="Target")
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(
                description="first", questions=[Question(main_question="x")]
            ),
            QuestionSection(
                description="second",
                # Claims (0, 0), but the target actually sits at (1, 0).
                questions=[target, _gated(target, section=0, question=0)],
            ),
        ]
    )

    ctx = _context(guide, 1, 1)
    assert ctx.question_index == (1, 0)
    assert ctx.question_id == target.id


def test_stale_reference_position_is_corrected():
    target = Question(main_question="Target")
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(
                description="s",
                questions=[
                    Question(main_question="x"),
                    target,
                    Question(
                        main_question="You said {}",
                        references=[
                            Reference(question_index=(0, 0), question_id=target.id)
                        ],
                    ),
                ],
            )
        ]
    )

    references = guide.question_sections[0].questions[2].references
    assert references is not None
    assert references[0].question_index == (0, 1)


def test_ref_to_a_missing_question_is_rejected():
    with pytest.raises(ValidationError, match="does not exist"):
        InterviewGuide(
            question_sections=[
                QuestionSection(
                    description="s",
                    questions=[_gated(Question(id=uuid4(), main_question="gone"))],
                )
            ]
        )


def test_duplicate_question_ids_are_rejected():
    question = Question(main_question="q")
    with pytest.raises(ValidationError, match="Duplicate question id"):
        InterviewGuide(
            question_sections=[
                QuestionSection(
                    description="s", questions=[question, question.model_copy()]
                )
            ]
        )


def test_shuffle_moves_the_derived_position_with_the_question():
    random.seed(0)
    questions = [Question(main_question=f"q{i}", shuffle=True) for i in range(6)]
    target = questions[0]
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="shuffled", questions=questions),
            QuestionSection(description="fixed", questions=[_gated(target)]),
        ]
    )

    for _ in range(10):
        guide.shuffle()
        position = next(
            n
            for n, question in enumerate(guide.question_sections[0].questions)
            if question.id == target.id
        )
        ctx = _context(guide, 1, 0)
        assert ctx.question_index == (0, position)
