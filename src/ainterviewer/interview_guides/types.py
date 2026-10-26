from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class ConditionTrigger(StrEnum):
    MATCH = "match"
    CLASSIFICATION = "classification"


class ConditionAction(StrEnum):
    END_INTERVIEW = "end_interview"
    SKIP_SECTION = "skip_section"
    SKIP_QUESTION = "skip_question"
    SKIP_PROBES = "skip_probes"


# A security assessment only happens once an answer has been given, so the
# question can no longer be skipped -- skipping its remaining probes is the same
# thing.
type SecurityAction = Literal[
    ConditionAction.SKIP_PROBES,
    ConditionAction.SKIP_SECTION,
    ConditionAction.END_INTERVIEW,
]


class ProbingContext(StrEnum):
    SECTION = "section"
    QUESTION = "question"


class SecurityIntervention(BaseModel):
    """Marks a message as sent by the security check, to be shown in a modal
    rather than in the chat."""

    # Which decision triggered is deliberately left out, as it is sent to the
    # respondent's client.
    action: SecurityAction
    respondent_override: bool


class SecurityOverride(StrEnum):
    """The respondent's answer to a security intervention with
    `respondent_override`, sent back as the content of an ordinary message."""

    ACCEPT = "accept"
    """Let the intervention's action go ahead."""
    OVERRIDE = "override"
    """Carry on with the interview as if nothing had triggered."""
