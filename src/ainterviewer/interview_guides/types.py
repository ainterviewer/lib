from enum import StrEnum
from typing import Literal


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
