from pydantic import BaseModel, Field, model_validator

from ainterviewer.interview_guides.conditions import Conditions
from ainterviewer.interview_guides.exceptions import OverwriteError
from ainterviewer.interview_guides.media import Image
from ainterviewer.interview_guides.references import QuestionIndex, Reference
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.interview_guides.types import ContextType


class QuestionBase(BaseModel):
    """A question that can be asked to the interviewee"""

    description: str | None = Field(
        None,
        description="A description of the question, may be used to reformulate the question and improve the relevance of the probes.",
    )
    main_question: str = Field(description="The question to ask the interviewee.")
    probes: list[str] | None = Field(
        None,
        description="A list of possible follow-up questions to ask after the main question. If provided set max_probes_n or max_probes_time to greater than 0.",
    )
    max_probes_n: int | None = Field(
        default=4, ge=0, description="Max number of probes."
    )
    max_probes_time: float | None = Field(
        None, gt=0, description="Max time to spend on probing, in seconds."
    )


class QuestionBaseExtended(QuestionBase):
    survey_item: SurveyItem | None = None

    # NOTE: We currently avoid doing this during the interview so the
    # interviewee doesn't get stuck because the last message has
    # can_answer=False
    # TODO: Consider flipping to cant_answer instead
    can_answer: bool = Field(
        True,
        description="Should the user be able to answer the question? Disable this to make the question into a message",
    )


class Question(QuestionBaseExtended):
    """A question that can be asked to the interviewee"""

    alternative_main_questions: list[str] | None = Field(
        None,
        description="List of alternative formulations of the main question, will be chosen at random.",
    )

    check_if_answered: bool = Field(
        True,
        description="Check if the question has already been answered under a previous question",
    )
    check_if_exhausted: bool = Field(
        False,
        description="During the probing, check if the question has been exhausted and the model should continue to the next question.",
    )
    can_skip: bool = Field(
        True, description="Should the user be able to skip the question?"
    )
    shuffle: bool = Field(
        True,
        description="Should the question be shuffled?",
    )
    create_segue: bool = Field(
        True,
        description="Create a segue from the previous question, to possibly improve the flow of the interview.",
    )
    exclude_from_history: bool = Field(
        False,
        description="Exclude from the interview history. This means that the model will not use this question or the response as a context when it asks further questions.",
    )

    variables: list[str] | None = Field(
        None,
        description="Variables that can be used in the question, ie. uuid. In case they are supplied, they will be filled in before the question is asked. The question should be formatted with Jinja2 style templating.",
    )
    references: list[Reference] | None = None
    image: Image | None = None
    user_image: bool = Field(
        False,
        description="Whether the user should be able to upload an image as a response",
    )
    conditions: Conditions | None = None
    probing_context: ContextType | None = None

    # Automatically generated in interview guide generation
    index: QuestionIndex | None = Field(
        None,
        description="The index of the question in the interview, ie (section, question) = (2, 2) (for 3rd section 3rd question). Used to keep track of questions initial position after shuffling.",
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.__data = data

        if self.survey_item is not None:
            # If survey item is present, set different defaults
            self._set_default("max_probes_n", 0)
            self._set_default("create_segue", False)
            self._set_default("check_if_answered", False)

    def _set_default(self, key, value, strict: bool = False):
        if key not in self.__data:
            setattr(self, key, value)
        elif strict:
            raise OverwriteError(f"{self.__data[key]=:}, {key=:} {value=:}")

    @model_validator(mode="after")
    def check_optionals(self):
        # TODO:
        # Add more checks
        if self.max_probes_n == 0 and self.max_probes_time is not None:
            raise ValueError(
                "Probing time specified, but max_probes is 0. Set max_probes to None (null) to only have time limit."
            )

        # TODO: Handle in frontend
        # if (
        #     self.probes
        #     and (self.max_probes_n == 0 or self.max_probes_n is None)
        #     and (self.max_probes_time is None or self.max_probes_time == 0)
        # ):
        #     raise ValueError(
        #         f"Probe questions specified, but no proper constraint is set. {self.max_probes_n=:}, {self.max_probes_time=:}"
        #     )

        return self

    @model_validator(mode="after")
    def check_references(self):
        if self.references:
            n_references = len(self.references)
            if not (n_placeholders := self.main_question.count("{}")) == n_references:
                raise ValueError(
                    f"{n_references} references are specified, but the main question only contains {n_placeholders}. Provide an equal amount of `{{}}` in the main question."
                )
        return self
