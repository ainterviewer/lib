class ConditionalException(Exception):
    pass


class SkipSectionCondition(ConditionalException):
    pass


class SkipQuestionCondition(ConditionalException):
    pass


class EndInterviewCondition(ConditionalException):
    pass


class SkipProbesCondition(ConditionalException):
    pass


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
