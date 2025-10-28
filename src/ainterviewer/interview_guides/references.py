from pydantic import BaseModel

type QuestionIndex = tuple[int, int]


class Reference(BaseModel):
    """
    Create a reference to a previous answer, useful when refering back to a
    previous answer, eg. an answer to a survey item. If used, the main question
    must contains an empty `{}` as a placeholder for the referred answer.
    """

    question_index: QuestionIndex
