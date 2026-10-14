from uuid import UUID

from pydantic import BaseModel, Field

type QuestionIndex = tuple[int, int]


class Reference(BaseModel):
    """
    Create a reference to a previous answer, useful when refering back to a
    previous answer, eg. an answer to a survey item. If used, the main question
    must contains an empty `{}` as a placeholder for the referred answer.
    """

    question_index: QuestionIndex = Field(
        description="Position (section, question) of the referred question. Derived from `question_id` whenever the guide is validated or shuffled."
    )
    question_id: UUID | None = Field(
        None,
        description="The referred question. The source of truth; None only for references stored before questions had ids, which the guide fills in from `question_index`.",
    )
