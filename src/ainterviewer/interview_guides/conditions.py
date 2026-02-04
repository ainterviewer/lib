# TODO:
# - Should conditions refactored to be separate from questions?

from __future__ import annotations

import operator
from typing import Literal

from pydantic import BaseModel, Field

from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger

comparison_operators = {
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class Conditions(BaseModel):
    conditions: list[Condition]
    action: ConditionAction


class Condition(BaseModel):
    """Create a condition that must be met to ask a question"""

    question_context: QuestionContext
    evaluation: list[ConditionEvaluation]
    negated: bool = False
    trigger_type: ConditionTrigger = ConditionTrigger.MATCH
    combine_next: Literal["AND", "OR"] | None = Field(
        default=None,
        description="Operator to combine this condition's result with the next. None for the last condition.",
    )


class QuestionContext(BaseModel):
    """The index of a question in a section
    section [int]: The index of the section
    question [int]: The index of the question
    part
    """

    section: int = Field(description="The index of a secition")
    question: int = Field(description="The index of a question")
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
    """

    trigger_value: str = Field(
        description="The pattern to match or value to compare against"
    )
    comparison_operator: Literal["==", "<", "<=", ">", ">="] = Field(
        default="==",
        description="Comparison operator. '==' for pattern matching, others for numeric/date comparison.",
    )
    combine_next: Literal["AND", "OR"] | None = Field(
        default=None,
        description="Operator to combine this evaluation's result with the next. None for the last evaluation.",
    )


def evaluate_conditions(contexts: list[str], conditions: Conditions) -> bool:
    """Evaluate multiple conditions against their corresponding contexts.

    Args:
        contexts: List of response/values to evaluate, one per condition.
        conditions: The Conditions object containing the list of conditions.

    Returns:
        True if the combined conditions are met, False otherwise.
    """
    condition_list = conditions.conditions

    if not condition_list:
        return True

    if len(contexts) != len(condition_list):
        raise ValueError(
            f"Number of contexts ({len(contexts)}) must match number of conditions ({len(condition_list)})"
        )

    result = evaluate_condition(contexts[0], condition_list[0])

    for i in range(len(condition_list) - 1):
        current_condition = condition_list[i]
        next_result = evaluate_condition(contexts[i + 1], condition_list[i + 1])

        match current_condition.combine_next:
            case "AND":
                result = result and next_result
            case "OR":
                result = result or next_result
            case None:
                raise ValueError(
                    "Must specify a combine_next value when there are more conditions"
                )

    return result


def evaluate_condition(context: str, condition: Condition) -> bool:
    """Evaluate a condition against the given context.

    Args:
        context: The response/value to evaluate against.
        condition: The condition with its evaluations to check.

    Returns:
        True if the condition is met, False otherwise.
        Result is negated if condition.negated is True.
    """
    match condition.trigger_type:
        case ConditionTrigger.MATCH:
            result = evaluate_re_condition(context, condition.evaluation)
        case ConditionTrigger.CLASSIFICATION:
            result = evaluate_classification_condition(context, condition.evaluation)

    return not result if condition.negated else result


def evaluate_re_condition(context: str, evaluations: list[ConditionEvaluation]) -> bool:
    """Evaluate a list of conditions using pattern matching and combine with specified operators.

    For pattern matching (==): checks if trigger_value pattern is found in context.
    For numeric comparison (<, <=, >, >=): compares numeric values.

    Evaluations are combined left-to-right using each evaluation's combine_next operator.
    """
    if not evaluations:
        return True

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


def _evaluate_single(context: str, evaluation: ConditionEvaluation) -> bool:
    """Evaluate a single condition against the context.

    For pattern matching (==): context is split by '|' (for multi-select values)
    and checks if trigger_value matches any of the values (case-insensitive).

    For numeric comparison: compares float values.
    """
    if evaluation.comparison_operator == "==":
        # Split by '|' for multi-select values, normalize case
        context_values = [v.strip().lower() for v in context.split("|")]
        trigger = evaluation.trigger_value.strip().lower()
        return trigger in context_values

    # Numeric comparison
    try:
        context_value = float(context)
        trigger_value = float(evaluation.trigger_value)
    except ValueError:
        # If conversion fails, try date comparison or return False
        return False

    op = comparison_operators[evaluation.comparison_operator]
    return op(context_value, trigger_value)


def evaluate_classification_condition(
    context: str, evaluations: list[ConditionEvaluation]
) -> bool:
    """Evaluate conditions using classification (non-deterministic).

    TODO: Implement classification-based evaluation.
    """
    raise NotImplementedError("Classification-based evaluation not yet implemented")
