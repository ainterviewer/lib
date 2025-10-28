# TODO:
# - Should conditions refactored to be separate from questions?

from __future__ import annotations

import re
from abc import ABC
from typing import Any, Literal

from pydantic import BaseModel

from ainterviewer import agents
from ainterviewer.interview_guides.types import (
    ConditionAction,
    ConditionTriggerType,
    ConditionTriggerValue,
    ConditionType,
)


class Condition(BaseModel):
    """Create a condition that must be met to ask a question"""

    type: ConditionType
    trigger_type: ConditionTriggerType
    question_context: QuestionContext
    trigger_value: Any
    action: ConditionAction


class QuestionContext(BaseModel):
    """The index of a question in a section
    section [int]: The index of the section
    question [int]: The index of the question
    """

    section: int
    question: int
    part: Literal["main", "probes", "all"] = "main"


class ConditionTrigger(ABC):
    def __init__(self, trigger_value: str):
        self.trigger_value = trigger_value

    async def check_condition(
        self,
        context: str,
        clean_context: bool,
        *args,
        **kwargs,
    ) -> bool:
        raise NotImplementedError

    def clean_value(self, value: str) -> str:
        return value.strip().lower()


class ConditionTriggerRE(ConditionTrigger):
    def __init__(self, trigger_value: ConditionTriggerValue):
        super().__init__(trigger_value=trigger_value)

    async def check_condition(
        self,
        context: str,
        clean_context: bool,
        *args,
        **kwargs,
    ) -> bool:
        if clean_context:
            context = self.clean_value(context)
        return bool(re.match(f"^{self.trigger_value}$", context))


class ConditionTriggerClassification(ConditionTrigger):
    def __init__(
        self,
        trigger_value: str,
        classification_agent: agents.ClassificationAgent,
    ):
        super().__init__(trigger_value=trigger_value)
        self.classification_agent = classification_agent

    async def check_condition(self, context: str, *args, **kwargs) -> bool:
        raise NotImplementedError
        return await self.classification_agent.classify(
            self.trigger_value, self.value, *args, **kwargs
        )


async def evaluate_condition(
    context,
    trigger_value,
    trigger_type: ConditionTriggerType,
    *args,
    **kwargs,
) -> bool:
    match trigger_type:
        case ConditionTriggerType.RE:
            clean_value = kwargs.pop("clean_value", True)
            trigger = ConditionTriggerRE(trigger_value=trigger_value, *args, **kwargs)
            return await trigger.check_condition(context, clean_value)

        case ConditionTriggerType.CLASSIFICATION:
            raise NotImplementedError
            trigger = ConditionTriggerClassification(
                trigger_value=trigger_value, *args, **kwargs
            )
            return await trigger.check_condition(context)
        case _:
            raise ValueError(f"Invalid trigger type {trigger_type}")
