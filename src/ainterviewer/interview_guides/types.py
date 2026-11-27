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


class ProbeSteering(StrEnum):
    """How closely the probes follow a question's suggested probes.

    - loose: the model may use a suggestion, paraphrased, or ask its own.
    - reword: every probe is one of the suggestions, reworded to fit the
      conversation -- it may build on the respondent's last answer.
    - strict: every probe is one of the suggestions, asked as written.

    Under reword and strict, probing ends once no suggestion is left to ask.
    """

    LOOSE = "loose"
    REWORD = "reword"
    STRICT = "strict"

    @property
    def must_use_suggestion(self) -> bool:
        return self in (ProbeSteering.REWORD, ProbeSteering.STRICT)
