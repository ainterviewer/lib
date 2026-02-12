import pytest

from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    Conditions,
    QuestionContext,
    _evaluate_single,
    evaluate_condition,
    evaluate_conditions,
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
    combine_next: str | None = None,
) -> Condition:
    return Condition(
        question_context=QuestionContext(section=section, question=question),
        evaluation=evaluations,
        negated=negated,
        trigger_type=trigger_type,
        combine_next=combine_next,
    )


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

    def test_multi_select_pipe_match(self):
        ev = ConditionEvaluation(trigger_value="b")
        assert _evaluate_single("a | b | c", ev) is True

    def test_multi_select_pipe_no_match(self):
        ev = ConditionEvaluation(trigger_value="d")
        assert _evaluate_single("a | b | c", ev) is False

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
        assert evaluate_match_condition("a | b", evals) is True

    def test_and_one_false(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="AND"),
            ConditionEvaluation(trigger_value="c"),
        ]
        assert evaluate_match_condition("a | b", evals) is False

    def test_or_one_true(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="OR"),
            ConditionEvaluation(trigger_value="c"),
        ]
        assert evaluate_match_condition("a | b", evals) is True

    def test_or_both_false(self):
        evals = [
            ConditionEvaluation(trigger_value="x", combine_next="OR"),
            ConditionEvaluation(trigger_value="y"),
        ]
        assert evaluate_match_condition("a | b", evals) is False

    def test_missing_combine_next_raises(self):
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next=None),
            ConditionEvaluation(trigger_value="b"),
        ]
        with pytest.raises(ValueError, match="Must specify a combine_next"):
            evaluate_match_condition("a | b", evals)

    def test_three_evaluations_and_or(self):
        """A AND B OR C — left-to-right: (A AND B) OR C"""
        evals = [
            ConditionEvaluation(trigger_value="a", combine_next="AND"),
            ConditionEvaluation(trigger_value="x", combine_next="OR"),
            ConditionEvaluation(trigger_value="c"),
        ]
        # a matches, x does not -> False after AND; c matches -> True after OR
        assert evaluate_match_condition("a | c", evals) is True


# ── evaluate_condition ──────────────────────────────────────────────────────


class TestEvaluateCondition:
    def test_match_true(self):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        assert evaluate_condition("yes", cond) is True

    def test_match_false(self):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        assert evaluate_condition("no", cond) is False

    def test_negated_true_becomes_false(self):
        cond = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], negated=True
        )
        assert evaluate_condition("yes", cond) is False

    def test_negated_false_becomes_true(self):
        cond = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], negated=True
        )
        assert evaluate_condition("no", cond) is True

    def test_classification_raises(self):
        cond = _make_condition(
            [ConditionEvaluation(trigger_value="yes")],
            trigger_type=ConditionTrigger.CLASSIFICATION,
        )
        with pytest.raises(NotImplementedError):
            evaluate_condition("yes", cond)


# ── evaluate_conditions ─────────────────────────────────────────────────────


class TestEvaluateConditions:
    def test_empty_conditions_returns_true(self):
        conds = Conditions(conditions=[], action=ConditionAction.SKIP_QUESTION)
        assert evaluate_conditions([], conds) is True

    def test_single_condition_true(self):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Conditions(conditions=[cond], action=ConditionAction.SKIP_QUESTION)
        assert evaluate_conditions(["yes"], conds) is True

    def test_single_condition_false(self):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Conditions(conditions=[cond], action=ConditionAction.SKIP_QUESTION)
        assert evaluate_conditions(["no"], conds) is False

    def test_two_conditions_and(self):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="AND"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Conditions(
            conditions=[cond1, cond2], action=ConditionAction.ASK_QUESTION
        )
        assert evaluate_conditions(["yes", "no"], conds) is True

    def test_two_conditions_and_one_fails(self):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="AND"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Conditions(
            conditions=[cond1, cond2], action=ConditionAction.ASK_QUESTION
        )
        assert evaluate_conditions(["yes", "yes"], conds) is False

    def test_two_conditions_or(self):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next="OR"
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Conditions(
            conditions=[cond1, cond2], action=ConditionAction.SKIP_SECTION
        )
        assert evaluate_conditions(["nope", "no"], conds) is True

    def test_mismatched_contexts_raises(self):
        cond = _make_condition([ConditionEvaluation(trigger_value="yes")])
        conds = Conditions(conditions=[cond], action=ConditionAction.SKIP_QUESTION)
        with pytest.raises(ValueError, match="Number of contexts"):
            evaluate_conditions(["a", "b"], conds)

    def test_missing_combine_next_raises(self):
        cond1 = _make_condition(
            [ConditionEvaluation(trigger_value="yes")], combine_next=None
        )
        cond2 = _make_condition([ConditionEvaluation(trigger_value="no")])
        conds = Conditions(
            conditions=[cond1, cond2], action=ConditionAction.END_INTERVIEW
        )
        with pytest.raises(ValueError, match="Must specify a combine_next"):
            evaluate_conditions(["yes", "no"], conds)


# ── QuestionContext ─────────────────────────────────────────────────────────


class TestQuestionContext:
    def test_defaults(self):
        ctx = QuestionContext(section=0, question=1)
        assert ctx.part == "main"

    def test_custom_part(self):
        ctx = QuestionContext(section=0, question=1, part="probes")
        assert ctx.part == "probes"
