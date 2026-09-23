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
