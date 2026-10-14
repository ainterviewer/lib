# TODO:
# - Consider refactoring:
#   - Question into image, survey items and conditions as separate classes
# - Add SkipJsonSchema to fields that shouldn't be considered in the
# json schema --> This will help when giving the schema to the AI as a response
# model

from __future__ import annotations

import warnings
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from ainterviewer.interview_guides.questions import Question, QuestionBase
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.shuffle import shuffle_items
from ainterviewer.interview_guides.variables import (
    BUILTIN_VARIABLES,
    extract_placeholders,
)


class InterviewGuideBase[Q: QuestionBase](BaseModel):
    """A guide for the interviewer to follow during the interview."""

    framing: str = Field(
        "",
        description="A description of the interview and its purpose. Only used by the model.",
    )
    introduction: str | SkipJsonSchema[InterviewMessage] = Field(
        "",
        description="An introduction to the interview. Displayed to the interviewee as the first message. They whon't be able to respond to this message.",
    )
    question_sections: list[QuestionSection[Q]] = Field(
        default_factory=list,
        description="A list of sections containing questions to ask the interviewee",
    )
    outro: str | SkipJsonSchema[InterviewMessage] | None = Field(
        None,
        description="An outro message for the interview. Displayed to the interviewee as the last message they will see. They will not be able to answer this message.",
    )
    # TODO: Should this kind of message instead be specified in the condition?
    alt_outro: SkipJsonSchema[str | None] = Field(
        None,
        description="Used if a condition results in an EndInterview, eg. from missing consent. Displayed to the interviewee as the last message they will see. They will not be able to answer this message.",
    )
    extra_variables: list[str] = Field(
        default_factory=list,
        description=(
            "Names of variables (beyond the built-in project_id and interview_id) "
            "that this guide expects to be supplied via referable_values at runtime. "
            "Every {{ placeholder }} in user-facing templates must appear here or "
            "be a built-in."
        ),
    )

    @model_validator(mode="after")
    def _validate_template_placeholders(self):
        allowed = BUILTIN_VARIABLES | set(self.extra_variables)
        errors: list[str] = []

        def check(label: str, text: str | None) -> None:
            if not text:
                return
            unknown = extract_placeholders(text) - allowed
            if unknown:
                errors.append(f"{label}: unknown placeholder(s) {sorted(unknown)}")

        intro = self.introduction
        check(
            "introduction",
            intro.message if isinstance(intro, InterviewMessage) else intro,
        )

        outro = self.outro
        check(
            "outro",
            outro.message if isinstance(outro, InterviewMessage) else outro,
        )

        check("alt_outro", self.alt_outro)

        for tm in getattr(self, "timed_messages", None) or []:
            check(f"timed_messages[time={tm.time}]", tm.message)

        for s_idx, section in enumerate(self.question_sections):
            for q_idx, question in enumerate(section.questions):
                check(
                    f"question_sections[{s_idx}].questions[{q_idx}].main_question",
                    question.main_question,
                )

        if errors:
            raise ValueError(
                "Interview guide references undeclared variables. "
                "Add them to `extra_variables` or fix the placeholder.\n  - "
                + "\n  - ".join(errors)
            )
        return self


class TimedMessage(BaseModel):
    """A message that is displayed to the interviewee after a certain amount of time"""

    message: str = Field(description="The message to display")

    time: int = Field(
        description="The time in seconds to wait before displaying the message"
    )
    include_in_history: bool = Field(
        False,
        description="Should the message be included in the interview history?",
    )
    as_modal: bool = Field(
        False,
        description="Should the message be shown as a modal? Defaults to False meaning it will be shown as a normal message.",
    )


class InterviewMessage(BaseModel):
    message: str


class InterviewGuideTemplate(InterviewGuideBase[QuestionBase]):
    model_config = {"title": "InterviewGuideTemplate"}


class InterviewGuide(InterviewGuideBase[Question]):
    model_config = {"title": "InterviewGuide"}

    timed_messages: list[TimedMessage] | None = Field(
        None,
        description="Messages that are displayed to the interviewee after a certain amount of time",
    )
    ai_generated_sections: int = 0

    def __init__(self, **data):
        super().__init__(**data)

        self.index_questions()

    @model_validator(mode="after")
    def _validate_ids(self):
        section_ids = [section.id for section in self.question_sections]
        question_ids = [
            question.id
            for section in self.question_sections
            for question in section.questions
        ]
        for kind, ids in (("section", section_ids), ("question", question_ids)):
            if duplicates := sorted({str(i) for i in ids if ids.count(i) > 1}):
                raise ValueError(f"Duplicate {kind} id(s): {duplicates}")
        return self

    @model_validator(mode="after")
    def _validate_question_refs(self):
        self.sync_question_refs()
        return self

    def sync_question_refs(self):
        """Point every condition and reference at its question's current
        position.

        The id is the source of truth and the position is derived from it, so
        this must run whenever the guide is reordered (see `shuffle`): the
        interview loop reads answers back by position. A condition or
        reference stored before questions had ids carries only a position, and
        gets the id of whatever sits there.
        """
        positions = {
            question.id: (n_section, n_question)
            for n_section, section in enumerate(self.question_sections)
            for n_question, question in enumerate(section.questions)
        }
        ids = {position: question_id for question_id, position in positions.items()}

        def resolve(question_id, position, label):
            if question_id is None:
                return ids.get(position), position
            if question_id not in positions:
                raise ValueError(f"{label} refers to a question that does not exist")
            return question_id, positions[question_id]

        for n_section, section in enumerate(self.question_sections):
            for n_question, question in enumerate(section.questions):
                label = f"question_sections[{n_section}].questions[{n_question}]"
                conditions = (
                    question.conditions.conditions if question.conditions else []
                )

                for condition in conditions:
                    ctx = condition.question_context
                    ctx.question_id, ctx.question_index = resolve(
                        ctx.question_id,
                        tuple(ctx.question_index),
                        f"{label} condition",
                    )

                for reference in question.references or []:
                    reference.question_id, reference.question_index = resolve(
                        reference.question_id,
                        tuple(reference.question_index),
                        f"{label} reference",
                    )

    @property
    def n_total_questions(self) -> int:
        return (
            sum(
                len(section.questions) + section.ai_generated_questions.n
                for section in self.question_sections
            )
            + self.ai_generated_sections * 5
        )

    def index_questions(self):
        for n_section, section in enumerate(self.question_sections):
            for n_question, question in enumerate(section.questions):
                # Avoids settings when shuffling
                if question.index is None:
                    indexes = [
                        question.index
                        for section in self.question_sections
                        for question in section.questions
                        if question.index
                    ]
                    if (index := (n_section, n_question)) in indexes:
                        warnings.warn(f"Index {index} already exists, overwriting")

                    question.index = index

    def condition_dependencies(self) -> list[tuple[UUID, UUID]]:
        """Pairs `(read, gated)` of question ids: the question a condition
        reads its answer from, and the question the condition hangs off.

        `read` has to be asked before `gated`, or there is no answer to read. A
        condition reading the question it hangs off is checked after that
        question is answered, so it constrains nothing and is left out.
        """
        return [
            (read, question.id)
            for section in self.question_sections
            for question in section.questions
            for condition in (
                question.conditions.conditions if question.conditions else []
            )
            if (read := condition.question_context.question_id) is not None
            and read != question.id
        ]

    def condition_order_errors(self) -> list[str]:
        """Conditions that read a question placed after the one they gate.

        Such a condition can never be evaluated -- the answer it reads does not
        exist yet -- and its absence is what lets `shuffle` fall back to the
        authored order. Checked when a guide is saved rather than on
        validation, so interview copies shuffled before the constraint existed
        still load.
        """
        positions = {
            question.id: (n_section, n_question)
            for n_section, section in enumerate(self.question_sections)
            for n_question, question in enumerate(section.questions)
        }
        errors = []
        for read, gated in self.condition_dependencies():
            (read_s, read_q), (gated_s, gated_q) = positions[read], positions[gated]
            if (read_s, read_q) > (gated_s, gated_q):
                errors.append(
                    f"question_sections[{gated_s}].questions[{gated_q}] has a "
                    f"condition reading question_sections[{read_s}]"
                    f".questions[{read_q}], which comes after it"
                )
        return errors

    def shuffle(self):
        """Shuffles sections and questions in the interview guide, based on
        their configuration, keeping every question a condition reads before
        the question the condition gates (see `shuffle_items`)."""
        section_of = {
            question.id: n_section
            for n_section, section in enumerate(self.question_sections)
            for question in section.questions
        }
        dependencies = self.condition_dependencies()

        self.question_sections = shuffle_items(
            self.question_sections,
            must_precede={
                (section_of[read], section_of[gated])
                for read, gated in dependencies
                if section_of[read] != section_of[gated]
            },
        )

        for section in self.question_sections:
            position = {question.id: n for n, question in enumerate(section.questions)}
            section.questions = shuffle_items(
                section.questions,
                must_precede={
                    (position[read], position[gated])
                    for read, gated in dependencies
                    if read in position and gated in position
                },
            )

        self.index_questions()
        self.sync_question_refs()

    def reduce(self):
        """Reduce the interview guide to contain one probe per main question.
        Used as a utility when synthesizing fixed answers.
        """

        for section in self.question_sections:
            for question in section.questions:
                if question.can_answer:
                    question.max_probes_n = 1
                    question.max_probes_time = None
