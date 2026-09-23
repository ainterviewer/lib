from dataclasses import dataclass
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field, create_model, field_validator

from ainterviewer.interview_guides.types import ConditionAction

ACTION_TEXT_DEFAULT = "Our automated safety system has triggered an intervention."


class SecurityEvaluation(BaseModel):
    reasoning: str = Field(description="A very short reason behind the decision")
    probability: float = Field(
        ge=0,
        le=1,
        description="Probability that the answer is yes.",
    )


class SecurityDecision(BaseModel):
    name: str
    description: str
    threshold: float = Field(
        ge=0, le=1, description="The threshold which triggers the action"
    )
    action: ConditionAction
    action_text: str = Field(
        default=ACTION_TEXT_DEFAULT,
        description="Text displayed to the user when the condition has triggered",
    )
    respondent_override: bool = Field(
        default=False,
        description="Should the user be able to override the action decision?",
    )

    @field_validator("name")
    @classmethod
    def check_name(cls, name: str) -> str:
        # The name becomes a field on the generated assessment model.
        if not name.isidentifier():
            raise ValueError(f"'{name}' is not a valid identifier")
        return name


class SecurityPolicy(BaseModel):
    decisions: list[SecurityDecision] = Field(min_length=1)

    @field_validator("decisions")
    @classmethod
    def check_unique_names(
        cls, decisions: list[SecurityDecision]
    ) -> list[SecurityDecision]:
        names = [decision.name for decision in decisions]
        if duplicates := {name for name in names if names.count(name) > 1}:
            raise ValueError(f"Duplicate security decision names: {duplicates}")
        return decisions

    @cached_property
    def assessment_model(self) -> type[BaseModel]:
        """The response model the security agent fills in, one field per decision"""
        return generate_assessment_model(self.decisions)


@dataclass(frozen=True)
class TriggeredDecision:
    decision: SecurityDecision
    evaluation: SecurityEvaluation


@dataclass(frozen=True)
class SecurityAssessmentResult:
    assessment: BaseModel
    """The full response, including decisions below their threshold"""
    triggered: list[TriggeredDecision]


def generate_assessment_model(
    security_decisions: list[SecurityDecision],
) -> type[BaseModel]:
    security_assessment_fields: dict[str, Any] = {
        security_decision.name: (
            SecurityEvaluation,
            Field(description=security_decision.description),
        )
        for security_decision in security_decisions
    }

    return create_model("SecurityAssessment", **security_assessment_fields)


def default_security_policy() -> SecurityPolicy:
    return SecurityPolicy(
        decisions=[
            SecurityDecision(
                name="self_harm",
                description="Does the transcript in any way indicate that the respondent has any intentions of self harm?",
                threshold=0.3,
                action=ConditionAction.END_INTERVIEW,
            ),
            SecurityDecision(
                name="others_harm",
                description="Does the transcript in any way indicate that the respondent has any intentions of harming others?",
                threshold=0.3,
                action=ConditionAction.END_INTERVIEW,
            ),
            SecurityDecision(
                name="termination",
                description="Does the transcript in any way indicate that the respondent wants to terminate the interview?",
                threshold=0.5,
                action=ConditionAction.END_INTERVIEW,
                respondent_override=True,
            ),
            SecurityDecision(
                name="uncomfortable",
                description="Does the transcript in any way indicate that the respondent is uncomfortable answering the question?",
                threshold=0.5,
                action=ConditionAction.SKIP_QUESTION,
                respondent_override=True,
            ),
        ]
    )
