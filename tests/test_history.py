"""What the interview loop reads off the history when it asks for an answer.

`AInterviewer.receive_data` declares the kind of answer it is inviting to the
host, which cannot work it out for itself -- a respondent's client submits
a closed answer in the same frame as free text. The declaration comes from the
turn the answer will land on, so these pin the turn being the right one, on the
live path and on the resume path alike.
"""

from typing import Any

import pytest

from ainterviewer.interview_guides import Image, InterviewGuide, Question
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    HistoryRecord,
    InterviewHistory,
    Turn,
)
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import LikertItem, RadioItem
from ainterviewer.types import MessageRole


@pytest.fixture
def likert():
    return LikertItem(options=["Disagree", "Neutral", "Agree"])


def record(
    content: str,
    role: MessageRole,
    section: int | None = 0,
    main_question: int | None = 0,
    sub_question: int | None = 0,
) -> HistoryRecord:
    """A stored message; the id is irrelevant to reconstruction."""
    return HistoryRecord(
        message_id=0,
        content=content,
        role=role,
        section=section,
        main_question=main_question,
        sub_question=sub_question,
    )


def guide(*survey_items: Any) -> InterviewGuide:
    return InterviewGuide(
        question_sections=[
            QuestionSection[Question](
                description="s",
                questions=[
                    Question(
                        description="d",
                        main_question=f"q{i}",
                        survey_item=survey_item,
                    )
                    for i, survey_item in enumerate(survey_items)
                ],
            )
        ]
    )


# ── current_turn ─────────────────────────────────────────────────────────────


class TestCurrentTurn:
    def test_none_before_the_first_question(self):
        assert InterviewHistory().current_turn is None

    def test_none_when_a_section_has_no_questions_yet(self):
        history = InterviewHistory()
        history.add_section("s")

        assert history.current_turn is None

    def test_main_question_when_there_are_no_probes(self, likert):
        history = InterviewHistory()
        history.add_section("s")
        history.add_question(
            question_description="d",
            main_question=Turn(
                question=HistoryMessage(message="q"), survey_item=likert
            ),
        )

        turn = history.current_turn

        assert turn is not None
        assert turn.survey_item is likert

    def test_last_probe_once_probing_starts(self, likert):
        """A probe on a survey question draws free text, not another value.

        The guide's question still carries the survey item here, which is why
        the type is read off the turn rather than off the question.
        """
        history = InterviewHistory()
        history.add_section("s")
        history.add_question(
            question_description="d",
            main_question=Turn(
                question=HistoryMessage(message="q"), survey_item=likert
            ),
        )
        history.add_probe(probe=Turn(question=HistoryMessage(message="p")))

        turn = history.current_turn

        assert turn is not None
        assert turn.question.message == "p"
        assert turn.survey_item is None


# ── the resume path ──────────────────────────────────────────────────────────


class TestResumedHistory:
    """An interview reconnecting mid-question replays its stored messages and
    then waits for the answer the respondent never sent. The turn it waits on
    has to carry whatever the question carried, or the answer is stored as free
    text -- which is what made resumed survey answers indistinguishable from
    open ones in the message table.

    A stored message carries no guide data, so all of it is read from the
    guide by the message's coordinates."""

    def test_pending_survey_item_is_carried(self, likert):
        history = InterviewHistory()

        history.process_history(
            [record("q0", role=MessageRole.ASSISTANT)],
            guide(likert),
        )

        turn = history.current_turn

        assert turn is not None
        assert turn.survey_item == likert
        assert turn.answer is None

    def test_pending_probe_carries_no_survey_item(self, likert):
        history = InterviewHistory()

        history.process_history(
            [
                record("q0", role=MessageRole.ASSISTANT),
                record("Agree", role=MessageRole.USER),
                record("p", role=MessageRole.ASSISTANT, sub_question=1),
            ],
            guide(likert),
        )

        turn = history.current_turn

        assert turn is not None
        assert turn.question.message == "p"
        assert turn.survey_item is None

    def test_second_question_is_the_pending_one(self, likert):
        radio = RadioItem(options=["Mand", "Kvinde"])
        history = InterviewHistory()

        history.process_history(
            [
                record("q0", role=MessageRole.ASSISTANT),
                record("Agree", role=MessageRole.USER),
                record("q1", role=MessageRole.ASSISTANT, main_question=1),
            ],
            guide(likert, radio),
        )

        turn = history.current_turn

        assert turn is not None
        assert turn.survey_item == radio
        assert turn.answer is None

    def test_image_is_read_from_the_guide(self):
        image_guide = guide(None)
        question = image_guide.question_sections[0].questions[0]
        question.image = Image(
            name="cat.png",
            data=None,
            description="A cat",
            alt="A cat",
            primer="Look at this",
        )
        history = InterviewHistory()

        history.process_history([record("q0", role=MessageRole.ASSISTANT)], image_guide)

        image = history.current_question.image
        assert image is not None
        assert image.description.message == "A cat"
        assert image.primer is not None
        assert image.primer.message == "Look at this"
