"""What the interview loop reads off the history when it asks for an answer.

`AInterviewer.receive_data` declares the kind of answer it is inviting to the
IO layer, which cannot work it out for itself -- a respondent's client submits
a closed answer in the same frame as free text. The declaration comes from the
turn the answer will land on, so these pin the turn being the right one, on the
live path and on the resume path alike.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    InterviewHistory,
    Turn,
)
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import LikertItem, RadioItem
from ainterviewer.types import MessageRole


@pytest.fixture
def likert():
    return LikertItem(options=["Disagree", "Neutral", "Agree"])


@dataclass
class StoredMessage:
    """The shape `InterviewHistory.process_history` reads a stored row as."""

    content: str
    role: MessageRole
    section: int | None = 0
    main_question: int | None = 0
    sub_question: int | None = 0
    survey_item: Any = None
    is_introduction: bool = False
    outro: bool = False
    timed: bool = False
    skipped_by_condition: bool = False
    include_in_history: bool = True
    image: Any = None
    can_answer: bool = True


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
    open ones in the message table."""

    def test_pending_survey_item_is_carried(self, likert):
        history = InterviewHistory()

        history.process_history(
            [
                StoredMessage(
                    content="q0", role=MessageRole.ASSISTANT, survey_item=likert
                )
            ],
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
                StoredMessage(
                    content="q0", role=MessageRole.ASSISTANT, survey_item=likert
                ),
                StoredMessage(content="Agree", role=MessageRole.USER),
                StoredMessage(content="p", role=MessageRole.ASSISTANT, sub_question=1),
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
                StoredMessage(
                    content="q0", role=MessageRole.ASSISTANT, survey_item=likert
                ),
                StoredMessage(content="Agree", role=MessageRole.USER),
                StoredMessage(
                    content="q1",
                    role=MessageRole.ASSISTANT,
                    main_question=1,
                    survey_item=radio,
                ),
            ],
            guide(likert, radio),
        )

        turn = history.current_turn

        assert turn is not None
        assert turn.survey_item == radio
        assert turn.answer is None
