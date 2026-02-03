# TODO:
# - Should conditions refactored to be separate from questions?

from __future__ import annotations

import operator
import re
from typing import Literal

from pydantic import BaseModel, Field

from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger

logic_operators = {
    "AND": operator.and_,
    "OR": operator.or_,
}

comparison_operators = {
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


class Condition(BaseModel):
    """Create a condition that must be met to ask a question"""

    question_context: QuestionContext
    evaluation: list[ConditionEvaluation]
    action: ConditionAction
    negated: bool = False
    trigger_type: ConditionTrigger = ConditionTrigger.RE


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
    trigger_value: str
    logic_operator: Literal["AND", "OR"]
    comparison_operator: Literal["==", "!=", "<", "<=", ">", ">="]


def evaluate_condition(context: str, condition: Condition) -> bool:
    match condition.trigger_type:
        case ConditionTrigger.RE:
            result = evaluate_re_condition(context, condition.evaluation)
        case ConditionTrigger.CLASSIFICATION:
            result = evaluate_classification_condition(context, condition.evaluation)

    if result:
        ...

    return not bool() if condition.negated else bool()


def evaluate_re_condition(context, evaluation: ConditionEvaluation) -> bool:
    match evaluation.comparison_operator:
        case "==":
            result = evaluation.trigger_value == context
        case "in":
            result = bool(re.search(evaluation.trigger_value, context))
        case _:
            raise ValueError("invalid condition rule")

    return result


def evaluate_classification_condition(context, evaluation: ConditionEvaluation) -> bool:
    return bool()


# class ConditionTrigger(ABC):
#     def __init__(self, trigger_value: str):
#         self.trigger_value = trigger_value
#
#     async def check_condition(
#         self,
#         context: str,
#         clean_context: bool,
#         *args,
#         **kwargs,
#     ) -> bool:
#         raise NotImplementedError
#
#     def clean_value(self, value: str) -> str:
#         return value.strip().lower()
#
#
# class ConditionTriggerRE(ConditionTrigger):
#     def __init__(self, trigger_value: str):
#         super().__init__(trigger_value=trigger_value)
#
#     async def check_condition(
#         self,
#         context: str,
#         clean_context: bool,
#         *args,
#         **kwargs,
#     ) -> bool:
#         if clean_context:
#             context = self.clean_value(context)
#
#         return bool(re.match(f"^{self.trigger_value}$", context))
#
#
# class ConditionTriggerClassification(ConditionTrigger):
#     def __init__(
#         self,
#         trigger_value: str,
#         classification_agent: agents.ClassificationAgent,
#     ):
#         super().__init__(trigger_value=trigger_value)
#         self.classification_agent = classification_agent
#
#     async def check_condition(self, context: str, *args, **kwargs) -> bool:
#         raise NotImplementedError
#         return await self.classification_agent.classify(
#             self.trigger_value, self.value, *args, **kwargs
#         )
#
#
# async def evaluate_condition(
#     context: str,
#     trigger_value: str,
#     trigger_type: ConditionTriggerEvaluator,
#     *args,
#     **kwargs,
# ) -> bool:
#     match trigger_type:
#         case ConditionTriggerEvaluator.RE:
#             clean_value = kwargs.pop("clean_value", True)
#             trigger = ConditionTriggerRE(trigger_value=trigger_value, *args, **kwargs)
#             return await trigger.check_condition(context, clean_value)
#
#         case ConditionTriggerEvaluator.CLASSIFICATION:
#             raise NotImplementedError
#             trigger = ConditionTriggerClassification(
#                 trigger_value=trigger_value, *args, **kwargs
#             )
#             return await trigger.check_condition(context)
#
#         case _:
#             raise ValueError(f"Invalid trigger type {trigger_type}")
