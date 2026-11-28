from enum import StrEnum


class ConditionalException(Exception):
    pass


class SkipProbesCondition(ConditionalException):
    pass


class SkipQuestionCondition(ConditionalException):
    pass


class SkipSectionCondition(ConditionalException):
    pass


class EndInterviewCondition(ConditionalException):
    pass


type InterviewControlCondition = type[
    SkipProbesCondition
    | SkipQuestionCondition
    | SkipSectionCondition
    | EndInterviewCondition
]


class SkipQuestionException(Exception):
    pass


class LanguageNotSupportedError(Exception):
    pass


class ConfigError(Exception):
    pass


class MissingEnvironmentVariable(Exception):
    pass


class ClassificationError(Exception):
    pass


class StructuredOutputError(Exception):
    """The model was asked for structured output and returned none: it refused,
    or the provider handed back nothing that parsed."""


class UnavailableReason(StrEnum):
    """Why a model could not take a request."""

    #: On its way up -- an instance booting. Worth waiting for.
    STARTING = "starting"
    #: Up, but turning requests away for now: at capacity, or rate limited.
    BUSY = "busy"
    #: Not coming without someone acting on it: nothing can be started, or
    #: the deployment runs no such model. Waiting does not help.
    DOWN = "down"


class ModelUnavailable(Exception):
    """The model did not take the request, and the provider said why.

    Raised by `chat` in place of the provider's error, so that the code which
    falls back when a call fails can tell a reply that went wrong -- where a
    fallback is the right answer -- from a model that is not there, where it
    only hides the outage and spends another call's timeout finding out the
    same thing. Every fallback re-raises it.

    `retry_after` is how many seconds until it is worth asking again, None
    when it is not (`DOWN`). The caller decides whether to wait.
    """

    def __init__(
        self,
        model: str,
        reason: UnavailableReason,
        retry_after: int | None,
        detail: str = "",
    ) -> None:
        super().__init__(f"{model} is unavailable ({reason}): {detail}")
        self.model = model
        self.reason = reason
        self.retry_after = retry_after
        self.detail = detail

    @property
    def worth_waiting(self) -> bool:
        return self.retry_after is not None
