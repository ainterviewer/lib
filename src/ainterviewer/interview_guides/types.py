from enum import StrEnum


class ConditionTrigger(StrEnum):
    MATCH = "match"
    CLASSIFICATION = "classification"


class ConditionAction(StrEnum):
    END_INTERVIEW = "end_interview"
    SKIP_SECTION = "skip_section"
    SKIP_QUESTION = "skip_question"
    SKIP_PROBES = "skip_probes"


class ContextType(StrEnum):
    """Context to be upsed for probing"""

    SECTION = "section"
    QUESTION = "question"
