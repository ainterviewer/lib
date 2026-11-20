from __future__ import annotations

import asyncio
import operator
from collections.abc import Iterable
from typing import Literal, NoReturn, Protocol, assert_never
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from ainterviewer.exceptions import (
    EndInterviewCondition,
    SkipProbesCondition,
    SkipQuestionCondition,
    SkipSectionCondition,
)
from ainterviewer.interview_guides.references import QuestionIndex
from ainterviewer.interview_guides.survey_items import CHECKBOX_SEPARATOR
from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger

# The answer a condition block reads: text, or the values of a checkbox
# answer (see `split_checkbox_answer`).
Answer = str | list[str]

comparison_operators = {
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


# Strongest first. When several of a question's conditions fire at the same
# moment, the strongest action is taken: it subsumes the rest (a skipped
# section has no probes left to skip), so the order the conditions are listed
# in does not matter.
ACTION_PRECEDENCE: tuple[ConditionAction, ...] = (
    ConditionAction.END_INTERVIEW,
    ConditionAction.SKIP_SECTION,
    ConditionAction.SKIP_QUESTION,
    ConditionAction.SKIP_PROBES,
)


def strongest_action(actions: Iterable[ConditionAction]) -> ConditionAction | None:
    """The strongest of `actions` by `ACTION_PRECEDENCE`, or None if empty."""
    return min(actions, key=ACTION_PRECEDENCE.index, default=None)


def raise_condition(action: ConditionAction) -> NoReturn:
    """Raises the appropriate exception based on the condition action"""
    match action:
        case ConditionAction.SKIP_PROBES:
            raise SkipProbesCondition
        case ConditionAction.SKIP_QUESTION:
            raise SkipQuestionCondition
        case ConditionAction.SKIP_SECTION:
            raise SkipSectionCondition
        case ConditionAction.END_INTERVIEW:
            raise EndInterviewCondition
        case _:
            assert_never(action)


class Classifier(Protocol):
    async def classify(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples: str | dict | None = None,
    ) -> bool: ...


class Condition(BaseModel):
    """Blocks combined into one verdict, and the action taken when it is True.
    A question may carry several conditions, each with its own action (see
    `Question.conditions`)."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Stable identity of the condition, shared by every translation of it. Recorded with each evaluation, so its firings can be counted.",
    )
    blocks: list[ConditionBlock]
    action: ConditionAction


class ConditionBlock(BaseModel):
    """One part of a condition: the answer to one question, checked by its
    evaluations.

    A checkbox answer is read as the options ticked, and an option matches
    when it is one of them, whatever else was ticked. `exclusive` narrows
    that to "and nothing else"."""

    question_context: QuestionContext
    evaluation: list[ConditionEvaluation]
    negated: bool = False
    exclusive: bool = Field(
        default=False,
        description="For a checkbox answer: also require that nothing was ticked besides the options this block matches with '=='. Ignored for any other answer.",
    )
    trigger_type: ConditionTrigger = ConditionTrigger.MATCH
    combine_next: Literal["AND", "OR"] | None = Field(
        default=None,
        description="Operator to combine this block's result with the next. None for the last block.",
    )


class QuestionContext(BaseModel):
    """The question a condition block reads its answer from.

    `question_id` is the source of truth. `question_index` is its position,
    which the interview loop reads; the guide derives it from the id whenever
    it is validated or shuffled, so a client may send a stale one.
    """

    question_index: QuestionIndex = Field(
        description="Position (section, question) of the question to read."
    )
    question_id: UUID = Field(description="The question to read.")
    part: Literal["main", "probes", "both"] = Field(
        "main",
        description="The part of the question to include.",
    )


class ConditionEvaluation(BaseModel):
    """A single evaluation check with an optional operator to combine with the next evaluation.

    For "A AND B OR C", create:
        [
            ConditionEvaluation(trigger_value="A", combine_next="AND"),
            ConditionEvaluation(trigger_value="B", combine_next="OR"),
            ConditionEvaluation(trigger_value="C"),
        ]

    For numeric comparisons (e.g., age >= 18):
        ConditionEvaluation(trigger_value="18", comparison_operator=">=")

    For a Likert answer compared by its place on the scale (e.g., at least
    "Agree"), give the scale's options in order:
        ConditionEvaluation(
            trigger_value="Agree",
            comparison_operator=">=",
            scale=["Disagree", "Neutral", "Agree", "Strongly agree"],
        )
    """

    trigger_value: str | int | float = Field(
        description="The pattern to match or value to compare against"
    )
    comparison_operator: Literal["==", "<", "<=", ">", ">="] = Field(
        default="==",
        description="Comparison operator. '==' for pattern matching, others for numeric/date comparison, or for comparing places on `scale`.",
    )
    scale: list[str] | None = Field(
        default=None,
        description="The Likert item's options in order, first to last. Set, the answer and `trigger_value` are compared by their place on it rather than as numbers.",
    )
    combine_next: Literal["AND", "OR"] | None = Field(
        default=None,
        description="Operator to combine this evaluation's result with the next. None for the last evaluation.",
    )


def combine(left: bool | None, operator: Literal["AND", "OR"], right: bool | None):
    """Combine two verdicts in Kleene's three-valued logic.

    None is "unknown": the condition read a question the respondent was never
    asked, or skipped. It is decided by the other side where that side settles
    the result on its own -- `True OR unknown` is True, `False AND unknown` is
    False -- and stays unknown otherwise, so a rule never fires on an answer
    that does not exist.
    """
    match operator:
        case "AND":
            if left is False or right is False:
                return False
            if left is None or right is None:
                return None
            return True
        case "OR":
            if left is True or right is True:
                return True
            if left is None or right is None:
                return None
            return False


class ConditionEvaluator:
    """Evaluates conditions, optionally using a classifier for non-deterministic evaluation."""

    def __init__(self, classifier: Classifier | None = None):
        self.classifier = classifier

    async def evaluate_condition(
        self, contexts: list[Answer | None], condition: Condition
    ) -> bool | None:
        """Evaluate a condition's blocks against their corresponding contexts.

        A context of None is a question without an answer, which makes its
        block unknown (see `combine`). The result is None when the unknowns
        leave it undecided; the action only fires on True.
        """
        blocks = condition.blocks

        if not blocks:
            return True

        if len(contexts) != len(blocks):
            raise ValueError(
                f"Number of contexts ({len(contexts)}) must match number of blocks ({len(blocks)})"
            )

        result = await self.evaluate_block(contexts[0], blocks[0])

        for i in range(len(blocks) - 1):
            next_result = await self.evaluate_block(contexts[i + 1], blocks[i + 1])

            if (combine_next := blocks[i].combine_next) is None:
                raise ValueError(
                    "Must specify a combine_next value when there are more blocks"
                )
            result = combine(result, combine_next, next_result)

        return result

    async def evaluate_block(
        self, context: Answer | None, block: ConditionBlock
    ) -> bool | None:
        """Evaluate a block against the given context, or None when there is
        no answer to evaluate it against. Negating unknown leaves it
        unknown."""
        if context is None:
            return None

        match block.trigger_type:
            case ConditionTrigger.MATCH:
                result = evaluate_match_condition(context, block.evaluation)
                if block.exclusive and isinstance(context, list):
                    result = result and _only_matched(context, block.evaluation)
            case ConditionTrigger.CLASSIFICATION:
                text = (
                    context
                    if isinstance(context, str)
                    else CHECKBOX_SEPARATOR.join(context)
                )
                result = await self._evaluate_classification_condition(
                    text, block.evaluation
                )

        return not result if block.negated else result

    async def _evaluate_classification_condition(
        self, context: str, evaluations: list[ConditionEvaluation]
    ) -> bool:
        """Evaluate conditions using classification (non-deterministic).

        Each evaluation's trigger_value is used as the classification instruction.
        All classifications run concurrently, then results are combined left-to-right.
        """
        if self.classifier is None:
            raise ValueError(
                "A classifier is required to evaluate classification conditions"
            )

        results = await asyncio.gather(
            *(
                self.classifier.classify(context, str(ev.trigger_value))
                for ev in evaluations
            )
        )

        result = results[0]

        for i in range(len(evaluations) - 1):
            match evaluations[i].combine_next:
                case "AND":
                    result = result and results[i + 1]
                case "OR":
                    result = result or results[i + 1]
                case None:
                    raise ValueError(
                        "Must specify a combine_next value when there are more evaluations"
                    )

        return result


def evaluate_match_condition(
    context: Answer, evaluations: list[ConditionEvaluation]
) -> bool:
    """Evaluate a list of conditions and combine them with their operators.

    For '==': the answer is the trigger, or a checkbox answer includes it.
    For an ordering operator (<, <=, >, >=): compares numbers, or places on
    a `scale`.

    Evaluations are combined left-to-right using each evaluation's combine_next operator.
    """
    result = _evaluate_single(context, evaluations[0])

    for i in range(len(evaluations) - 1):
        current_eval = evaluations[i]
        next_result = _evaluate_single(context, evaluations[i + 1])

        match current_eval.combine_next:
            case "AND":
                result = result and next_result
            case "OR":
                result = result or next_result
            case None:
                raise ValueError(
                    "Must specify a combine_next value when there are more conditions"
                )

    return result


def _normalize(value: str | float) -> str:
    return str(value).strip().casefold()


def _only_matched(selected: list[str], evaluations: list[ConditionEvaluation]) -> bool:
    """Whether every ticked value is one the evaluations match with '==':
    what `ConditionBlock.exclusive` adds. A value typed into "Other" is
    none of them."""
    matched = {
        _normalize(evaluation.trigger_value)
        for evaluation in evaluations
        if evaluation.comparison_operator == "=="
    }
    return all(_normalize(value) in matched for value in selected)


def _evaluate_single(context: Answer, evaluation: ConditionEvaluation) -> bool:
    """Evaluate a single condition against the context.

    For '==': the answer equals trigger_value, or, for a checkbox answer,
    trigger_value is one of the values ticked. Case and surrounding
    whitespace are ignored.

    For an ordinal comparison (a `scale` is given): compares the places of
    the answer and trigger_value on the scale. An answer not on it is no match.

    For numeric comparison: compares float values.

    A checkbox answer has no order, so only '==' can match it.
    """
    if evaluation.comparison_operator == "==":
        trigger = _normalize(evaluation.trigger_value)
        if isinstance(context, list):
            return trigger in {_normalize(value) for value in context}
        return _normalize(context) == trigger

    if isinstance(context, list):
        return False

    op = comparison_operators[evaluation.comparison_operator]

    if evaluation.scale is not None:
        scale = [_normalize(option) for option in evaluation.scale]
        try:
            answer = scale.index(_normalize(context))
            trigger = scale.index(_normalize(evaluation.trigger_value))
        except ValueError:
            return False
        return op(answer, trigger)

    # Numeric comparison
    try:
        context_value = float(context)
        trigger_value = float(evaluation.trigger_value)
    except ValueError:
        # If conversion fails, try date comparison or return False
        return False

    return op(context_value, trigger_value)
