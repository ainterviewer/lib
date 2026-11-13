"""Problems in a guide, told to whoever edits it.

Each names where it is, so an editor can be taken there: the question or
section by id, and the field under it the problem is in.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError


class GuideError(BaseModel):
    """A problem that keeps a guide from being valid."""

    message: str = Field(description="What is wrong, in a sentence.")
    fix: str | None = Field(None, description="How to fix it, if not obvious.")
    question_id: UUID | None = Field(
        None, description="The question it is in, if it is in one."
    )
    section_id: UUID | None = Field(
        None, description="The section it is in, if it is in one."
    )
    field: str | None = Field(
        None,
        description="The field of the question (or section) it is in, e.g. "
        "`conditions`, if it is in one.",
    )


def _at(data: Any, *path: str | int) -> Any:
    for key in path:
        try:
            data = data[key]
        except (KeyError, IndexError, TypeError):
            return None
    return data


def _id(value: Any) -> UUID | None:
    try:
        return UUID(str(value))
    except ValueError:
        return None


def validation_errors(error: ValidationError, data: Any) -> list[GuideError]:
    """`error`, raised validating the guide `data`, as one `GuideError` per
    problem, named by the question or section it is in ("Question 5.2"), as
    the editor numbers them."""
    errors = []
    for e in error.errors():
        loc, message = e["loc"], e["msg"].removeprefix("Value error, ")
        match loc:
            case ("question_sections", int(s), "questions", int(q), *rest):
                question = _at(data, "question_sections", s, "questions", q)
                where, section_id = f"Question {s + 1}.{q + 1}", None
                question_id = _id(_at(question, "id"))
            case ("question_sections", int(s), *rest):
                where, question_id = f"Section {s + 1}", None
                section_id = _id(_at(data, "question_sections", s, "id"))
            case _:
                where, question_id, section_id, rest = None, None, None, list(loc)
        field = str(rest[0]) if rest else None
        path = ".".join(str(part) for part in rest)
        name = ", ".join(filter(None, [where, path]))
        errors.append(
            GuideError(
                message=f"{name}: {message}" if name else message,
                question_id=question_id,
                section_id=section_id,
                field=field,
            )
        )
    return errors
