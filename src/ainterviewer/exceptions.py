class ConditionalException(Exception):
    pass


class SkipSectionException(ConditionalException):
    pass


class SkipQuestionException(ConditionalException):
    pass


class EndInterviewException(ConditionalException):
    pass


class SkipProbesException(ConditionalException):
    pass


class LanguageNotSupportedError(Exception):
    pass


class ConfigError(Exception):
    pass


class MissingEnvironmentVariable(Exception):
    pass


class ClassificationError(Exception):
    pass
