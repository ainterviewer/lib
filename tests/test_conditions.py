from typing import Literal
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionBlock,
    ConditionEvaluation,
    ConditionEvaluator,
    QuestionContext,
    _evaluate_single,
    evaluate_match_condition,
)
from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_condition(
    evaluations: list[ConditionEvaluation],
    *,
    section: int = 0,
    question: int = 0,
    negated: bool = False,
    trigger_type: ConditionTrigger = ConditionTrigger.MATCH,
    combine_next: Literal["AND", "OR"] | None = None,
) -> ConditionBlock:
    return ConditionBlock(
        question_context=QuestionContext(
            question_index=(section, question), question_id=uuid4()
        ),
        evaluation=evaluations,
        negated=negated,
        trigger_type=trigger_type,
        combine_next=combine_next,
    )


def _make_evaluator(classifier=None) -> ConditionEvaluator:
    return ConditionEvaluator(classifier=classifier)


# ── _evaluate_single ───────────────────────────────────────────────────────


class TestEvaluateSingle:
    def test_exact_match(self):
        ev = ConditionEvaluation(trigger_value="yes")
        assert _evaluate_single("yes", ev) is True

    def test_no_match(self):
        ev = ConditionEvaluation(trigger_value="yes")
        assert _evaluate_single("no", ev) is False

    def test_case_insensitive(self):
        ev = ConditionEvaluation(trigger_value="Yes")
        assert _evaluate_single("YES", ev) is True

    def test_checkbox_includes(self):
        ev = ConditionEvaluation(trigger_value="b")
        assert _evaluate_single(["a", "b", "c"], ev) is True

    def test_checkbox_does_not_include(self):
        ev = ConditionEvaluation(trigger_value="d")
        assert _evaluate_single(["a", "b", "c"], ev) is False

    def test_checkbox_option_with_comma(self):
        ev = ConditionEvaluation(trigger_value="yes, often")
        assert _evaluate_single(["Yes, often", "No"], ev) is True

    def test_checkbox_no_partial_match(self):
        ev = ConditionEvaluation(trigger_value="dog")
        assert _evaluate_single(["Hotdog"], ev) is False

    def test_text_matches_whole_answer(self):
        """Text is matched whole: a '|' or ',' in it splits nothing."""
        ev = ConditionEvaluation(trigger_value="b")
        assert _evaluate_single("a | b", ev) is False
        assert _evaluate_single("a, b", ev) is False

    def test_checkbox_has_no_order(self):
        ev = ConditionEvaluation(trigger_value="1", comparison_operator=">=")
        assert _evaluate_single(["2"], ev) is False

    def test_numeric_greater_than(self):
        ev = ConditionEvaluation(trigger_value="18", comparison_operator=">=")
        assert _evaluate_single("20", ev) is True

    def test_numeric_less_than(self):
        ev = ConditionEvaluation(trigger_value="18", comparison_operator="<")
        assert _evaluate_single("10", ev) is True

    def test_numeric_equal(self):
        ev = ConditionEvaluation(trigger_value="5", comparison_operator="<=")
        assert _evaluate_single("5", ev) is True

    def test_numeric_greater_than_fail(self):
        ev = ConditionEvaluation(trigger_value="18", comparison_operator=">")
        assert _evaluate_single("18", ev) is False

    def test_numeric_non_numeric_context_returns_false(self):
        ev = ConditionEvaluation(trigger_value="18", comparison_operator=">=")
        assert _evaluate_single("not a number", ev) is False

    @pytest.mark.parametrize(
        ("operator", "answer", "expected"),
        [
            (">=", "Agree", True),
            (">=", "Strongly agree", True),
            (">=", "Neutral", False),
            (">", "Agree", False),
            ("<", "Disagree", True),
            ("<=", "agree ", True),
        ],
    )
    def test_scale_compares_places(self, operator, answer, expected):
        ev = ConditionEvaluation(
            trigger_value="Agree",
            comparison_operator=operator,
            scale=["Disagree", "Neutral", "Agree", "Strongly agree"],
        )
        assert _evaluate_single(answer, ev) is expected

    def test_scale_answer_off_the_scale_is_no_match(self):
        ev = ConditionEvaluation(
            trigger_value="Agree", comparison_operator="<", scale=["Disagree", "Agree"]
        )
        assert _evaluate_single("Maybe", ev) is False

    def test_scale_orders_options_that_look_like_numbers_by_place(self):
        # "10" comes first on this scale, so it is below "2" despite the number.
        ev = ConditionEvaluation(
            trigger_value="2", comparison_operator="<", scale=["10", "2"]
        )
        assert _evaluate_single("10", ev) is True


# ── evaluate_match_condition ────────────────────────────────────────────────


class TestEvaluateMatchCondition:
    def test_single_evaluation(self):
        evals = [ConditionEvaluation(trigger_value="yes")]
        assert evaluate_match_condition("yes", evals) is True

    def test_and_both_true(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="AND"),
            ConditionEvaluation(trigger_value="b"),
        ]
        assert evaluate_match_condition(["a", "b"], evals) is True

    def test_and_one_false(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="AND"),
            ConditionEvaluation(trigger_value="c"),
        ]
        assert evaluate_match_condition(["a", "b"], evals) is False

    def test_or_one_true(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="OR"),
            ConditionEvaluation(trigger_value="c"),
        ]
        assert evaluate_match_condition(["a", "b"], evals) is True

    def test_or_both_false(self):
        evals = [
            ConditionEvaluation(trigger_value="x", combine_next="OR"),
            ConditionEvaluation(trigger_value="y"),
        ]
        assert evaluate_match_condition(["a", "b"], evals) is False

    def test_missing_combine_next_raises(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next=None),
            ConditionEvaluation(trigger_value="b"),
        ]
        with pytest.raises(ValueError, match="Must specify a combine_next"):
            evaluate_match_condition(["a", "b"], evals)

    def test_three_evaluations_and_or(self):
        """A AND B OR C — left-to-right: (A AND B) OR C"""
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="AND"),
            ConditionEvaluation(trigger_value="x", combine_next="OR"),
            ConditionEvaluation(trigger_value="c"),
        ]
        # a matches, x does not -> False after AND; c matches -> True after OR
        assert evaluate_match_condition(["a", "c"], evals) is True


# ── ConditionEvaluator.evaluate_condition ──────────────────────────────────


class TestEvaluateCondition:
    @pytest.fixture
    def evaluator(self):
        return _make_evaluator()

    @pytest.mark.anyio
    async def test_match_true(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        assert await evaluator.evaluate_block("yes", cond) is True

    @pytest.mark.anyio
    async def test_match_false(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        assert await evaluator.evaluate_block("no", cond) is False

    @pytest.mark.anyio
    async def test_negated_true_becomes_false(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")], negated=True)
        assert await evaluator.evaluate_block("yes", cond) is False

    @pytest.mark.anyio
    async def test_negated_false_becomes_true(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")], negated=True)
        assert await evaluator.evaluate_block("no", cond) is True

    @pytest.mark.anyio
    async def test_checkbox_includes(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="Dog")])
        assert await evaluator.evaluate_block(["Dog", "Cat"], cond) is True

    @pytest.mark.anyio
    async def test_classification_reads_checkbox_as_text(self):
        classifier = AsyncMock()
        classifier.classify.return_value = True
        evaluator = _make_evaluator(classifier=classifier)
        cond = _make_condition(
            [ConditionEvaluation(trigger_value="has a pet")],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        await evaluator.evaluate_block(["Dog", "Cat"], cond)
        classifier.classify.assert_called_once_with("Dog, Cat", "has a pet")

    @pytest.mark.anyio
    async def test_classification_without_classifier_raises(self):
        evaluator = _make_evaluator(classifier=None)
        cond = _make_condition(
            [ConditionEvaluation(trigger_value="is employed")],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        with pytest.raises(ValueError, match="classifier is required"):
            await evaluator.evaluate_block("I work at a bank", cond)

    @pytest.mark.anyio
    async def test_classification_with_classifier(self):
        classifier = AsyncMock()
        classifier.classify.return_value = True
        evaluator = _make_evaluator(classifier=classifier)

        cond = _make_condition(
            [ConditionEvaluation(trigger_value="is employed")],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        result = await evaluator.evaluate_block("I work at a bank", cond)

        assert result is True
        classifier.classify.assert_called_once_with("I work at a bank", "is employed")

    @pytest.mark.anyio
    async def test_classification_negated(self):
        classifier = AsyncMock()
        classifier.classify.return_value = True
        evaluator = _make_evaluator(classifier=classifier)

        cond = _make_condition(
            [ConditionEvaluation(trigger_value="is employed")],
            trigger_type=ConditionTrigger.CLASSIFICATION,
            negated=True,
        )
        result = await evaluator.evaluate_block("I work at a bank", cond)

        assert result is False

    @pytest.mark.anyio
    async def test_classification_multiple_evaluations_and(self):
        classifier = AsyncMock()
        classifier.classify.side_effect = [True, False]
        evaluator = _make_evaluator(classifier=classifier)

        cond = _make_condition(
            [
                ConditionEvaluation(trigger_value="is employed", combine_next="AND"),
                ConditionEvaluation(trigger_value="is married"),
            ],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        result = await evaluator.evaluate_block("I work at a bank", cond)

        assert result is False

    @pytest.mark.anyio
    async def test_classification_multiple_evaluations_or(self):
        classifier = AsyncMock()
        classifier.classify.side_effect = [False, True]
        evaluator = _make_evaluator(classifier=classifier)

        cond = _make_condition(
            [
                ConditionEvaluation(trigger_value="is employed", combine_next="OR"),
                ConditionEvaluation(trigger_value="is a student"),
            ],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        result = await evaluator.evaluate_block("I am a student", cond)

        assert result is True


# ── ConditionEvaluator.evaluate_conditions ─────────────────────────────────


class TestEvaluateConditions:
    @pytest.fixture
    def evaluator(self):
        return _make_evaluator()

    @pytest.mark.anyio
    async def test_empty_conditions_returns_true(self, evaluator):
        conds = Condition(blocks=[], action=ConditionAction.SKIP_QUESTION)
        assert await evaluator.evaluate_condition([], conds) is True

    @pytest.mark.anyio
    async def test_single_condition_true(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Condition(blocks=[cond], action=ConditionAction.SKIP_QUESTION)
        assert await evaluator.evaluate_condition(["yes"], conds) is True

    @pytest.mark.anyio
    async def test_single_condition_false(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Condition(blocks=[cond], action=ConditionAction.SKIP_QUESTION)
        assert await evaluator.evaluate_condition(["no"], conds) is False

    @pytest.mark.anyio
    async def test_two_conditions_and(self, evaluator):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="AND"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Condition(blocks=[cond1, cond2], action=ConditionAction.SKIP_QUESTION)
        assert await evaluator.evaluate_condition(["yes", "no"], conds) is True

    @pytest.mark.anyio
    async def test_two_conditions_and_one_fails(self, evaluator):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="AND"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Condition(blocks=[cond1, cond2], action=ConditionAction.SKIP_QUESTION)
        assert await evaluator.evaluate_condition(["yes", "yes"], conds) is False

    @pytest.mark.anyio
    async def test_two_conditions_or(self, evaluator):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="OR"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Condition(blocks=[cond1, cond2], action=ConditionAction.SKIP_SECTION)
        assert await evaluator.evaluate_condition(["nope", "no"], conds) is True

    @pytest.mark.anyio
    async def test_mismatched_contexts_raises(self, evaluator):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Condition(blocks=[cond], action=ConditionAction.SKIP_QUESTION)
        with pytest.raises(ValueError, match="Number of contexts"):
            await evaluator.evaluate_condition(["a", "b"], conds)

    @pytest.mark.anyio
    async def test_missing_combine_next_raises(self, evaluator):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next=None
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Condition(blocks=[cond1, cond2], action=ConditionAction.END_INTERVIEW)
        with pytest.raises(ValueError, match="Must specify a combine_next"):
            await evaluator.evaluate_condition(["yes", "no"], conds)


# ── QuestionContext ─────────────────────────────────────────────────────────


class TestQuestionContext:
    def test_defaults(self):
        ctx = QuestionContext(question_index=(0, 1), question_id=uuid4())
        assert ctx.part == "main"

    def test_custom_part(self):
        ctx = QuestionContext(question_index=(0, 1), question_id=uuid4(), part="probes")
        assert ctx.part == "probes"
