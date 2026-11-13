import pytest
from pydantic import ValidationError

from ainterviewer.interview_guides import (
    InterviewGuide,
    LocalizedInterviewGuide,
    Question,
    QuestionSection,
    validation_errors,
)


def _data():
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(description="s", questions=[Question(main_question="a")]),
            QuestionSection(
                description="t",
                questions=[Question(main_question="b"), Question(main_question="c")],
            ),
        ]
    )
    rendered = LocalizedInterviewGuide.from_rendered({"EN": guide}, structure="EN")
    return rendered.model_dump(mode="json")


def _errors(data):
    with pytest.raises(ValidationError) as error:
        LocalizedInterviewGuide.model_validate(data)
    return validation_errors(error.value, data)


def test_an_error_in_a_question_names_and_points_at_it():
    data = _data()
    question = data["question_sections"][1]["questions"][1]
    question["can_skip"] = "maybe"

    [error] = _errors(data)

    assert error.message.startswith("Question 2.2, can_skip: ")
    assert str(error.question_id) == question["id"]
    assert error.section_id is None
    assert error.field == "can_skip"


def test_an_error_in_a_section_names_and_points_at_it():
    data = _data()
    data["question_sections"][0]["shuffle"] = "maybe"

    [error] = _errors(data)

    assert error.message.startswith("Section 1, shuffle: ")
    assert str(error.section_id) == data["question_sections"][0]["id"]
    assert error.question_id is None
