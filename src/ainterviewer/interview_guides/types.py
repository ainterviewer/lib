from enum import StrEnum


class ConditionType(StrEnum):
    YES_NO = "yes/no"


class ConditionTriggerType(StrEnum):
    RE = "re"
    CLASSIFICATION = "classification"


class ConditionTriggerValue(StrEnum):
    """A value to trigger a condition"""

    # TODO: Should this be refarctored based on and tied to the condition type?
    YES = "yes"
    NO = "no"


class ContextType(StrEnum):
    """Context to be upsed for probing"""

    SECTION = "section"
    QUESTION = "question"


class ConditionAction(StrEnum):
    SKIP_SECTION = "skip_section"
    SKIP_QUESTION = "skip_question"
    END_INTERVIEW = "end_interview"
