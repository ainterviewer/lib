# TODO:
# - Should conditions refactored to be separate from questions?

from __future__ import annotations

import asyncio
import operator
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger

comparison_operators = {
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class Classifier(Protocol):
    async def classify(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples: str | dict | None = None,
    ) -> bool: ...


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


class ConditionEvaluator:
    """Evaluates conditions, optionally using a classifier for non-deterministic evaluation."""

    def __init__(self, classifier: Classifier | None = None):
        self.classifier = classifier

    async def evaluate_conditions(
        self, contexts: list[str], conditions: Conditions
    ) -> bool:
        """Evaluate multiple conditions against their corresponding contexts."""
        condition_list = conditions.conditions

        if not condition_list:
            return True

        if len(contexts) != len(condition_list):
            raise ValueError(
                f"Number of contexts ({len(contexts)}) must match number of conditions ({len(condition_list)})"
            )

        result = await self.evaluate_condition(contexts[0], condition_list[0])

        for i in range(len(condition_list) - 1):
            current_condition = condition_list[i]
            next_result = await self.evaluate_condition(
                contexts[i + 1], condition_list[i + 1]
            )

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

    async def evaluate_condition(self, context: str, condition: Condition) -> bool:
        """Evaluate a condition against the given context."""
        match condition.trigger_type:
            case ConditionTrigger.MATCH:
                result = evaluate_match_condition(context, condition.evaluation)
            case ConditionTrigger.CLASSIFICATION:
                result = await self._evaluate_classification_condition(
                    context, condition.evaluation
                )

        return not result if condition.negated else result

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
            *(self.classifier.classify(context, ev.trigger_value) for ev in evaluations)
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
    context: str, evaluations: list[ConditionEvaluation]
) -> bool:
    """Evaluate a list of conditions using pattern matching and combine with specified operators.

    For pattern matching (==): checks if trigger_value pattern is found in context.
    For numeric comparison (<, <=, >, >=): compares numeric values.

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
