"""The interview loop's own bookkeeping: what a question is asked with, and
what a resumed interview does with the question it stopped at."""

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from ainterviewer.exceptions import EndInterviewCondition
from ainterviewer.interview_guides import InterviewGuide, TimedMessage
from ainterviewer.interview_guides.history import HistoryMessage, Turn
from ainterviewer.types import MessageRole
from tests.test_condition_branches import SKIP, Respondent, _interviewer, _reads
from tests.test_history import StoredMessage


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("ainterviewer.interview.asyncio.sleep", AsyncMock())


# ── References ──────────────────────────────────────────────────────────────

NAME, PLACE, GREETING = (uuid.uuid4() for _ in range(3))


def _referencing_guide(main_question: str, references: list[dict]) -> InterviewGuide:
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {"id": NAME, "main_question": "Name?", "max_probes_n": 0},
                        {"id": PLACE, "main_question": "Place?", "max_probes_n": 0},
                        {
                            "id": GREETING,
                            "main_question": main_question,
                            "max_probes_n": 0,
                            "references": references,
                        },
                    ],
                }
            ],
        }
    )


@pytest.mark.anyio
async def test_reference_fills_in_the_answer():
    guide = _referencing_guide(
        "Hi {}!", [{"question_id": NAME, "question_index": (0, 0)}]
    )
    respondent = Respondent({"Name?": ["Ann"], "Place?": ["Aarhus"], "Hi Ann!": ["Hi"]})

    await _interviewer(guide, respondent).handle_sections()

    assert respondent.asked[-1] == "Hi Ann!"


@pytest.mark.anyio
async def test_references_fill_in_their_answers_in_order():
    guide = _referencing_guide(
        "{} from {}?",
        [
            {"question_id": NAME, "question_index": (0, 0)},
            {"question_id": PLACE, "question_index": (0, 1)},
        ],
    )
    respondent = Respondent(
        {"Name?": ["Ann"], "Place?": ["Aarhus"], "Ann from Aarhus?": ["Yes"]}
    )

    await _interviewer(guide, respondent).handle_sections()

    assert respondent.asked[-1] == "Ann from Aarhus?"


# ── Resuming at a question with a rule on its own answer ─────────────────────

CONSENT, NEXT = uuid.uuid4(), uuid.uuid4()


def _consent_guide(action: str) -> InterviewGuide:
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Consent",
                    "questions": [
                        {
                            "id": CONSENT,
                            "main_question": "Consent?",
                            "survey_item": {"type": "radio", "options": ["Ja", "Nej"]},
                            "conditions": {
                                "action": action,
                                "conditions": [_reads(CONSENT, (0, 0), "Nej")],
                            },
                        },
                        {"id": NEXT, "main_question": "Next?", "max_probes_n": 0},
                    ],
                }
            ],
        }
    )


def _resumed(guide: InterviewGuide, answers: dict[str, list[str]]):
    respondent = Respondent(answers)
    # The question was shown in the earlier session.
    respondent.asked.append("Consent?")
    return _interviewer(guide, respondent), respondent


ASKED = StoredMessage("Consent?", MessageRole.ASSISTANT)
ANSWERED = [ASKED, StoredMessage("Nej", MessageRole.USER)]


@pytest.mark.anyio
async def test_rule_on_the_pending_answer_is_checked():
    interviewer, _ = _resumed(_consent_guide("end_interview"), {"Consent?": ["Nej"]})

    with pytest.raises(EndInterviewCondition):
        await interviewer.process_history([ASKED])


@pytest.mark.anyio
async def test_rule_on_an_answer_given_before_the_break_is_checked():
    interviewer, _ = _resumed(_consent_guide("end_interview"), {})

    with pytest.raises(EndInterviewCondition):
        await interviewer.process_history(ANSWERED)


@pytest.mark.anyio
async def test_rule_that_does_not_fire_carries_on():
    interviewer, respondent = _resumed(
        _consent_guide("end_interview"), {"Consent?": ["Ja"], "Next?": ["Sure"]}
    )

    await interviewer.process_history([ASKED])
    await interviewer.handle_sections()

    assert respondent.asked == ["Consent?", "Next?"]


@pytest.mark.anyio
async def test_skip_on_the_pending_answer_does_not_duplicate_the_question():
    interviewer, respondent = _resumed(
        _consent_guide("skip_question"), {"Next?": ["Sure"]}
    )

    await interviewer.process_history(ANSWERED)
    await interviewer.handle_sections()

    history = interviewer.interview_history
    assert len(history.sections) == 1
    assert [q.main_question.question.message for q in history[0].questions] == [
        "Consent?",
        "Next?",
    ]
    assert respondent.asked == ["Consent?", "Next?"]


# ── A rule that skips probes before its question is asked ────────────────────

FIRST, SECOND, THIRD = (uuid.uuid4() for _ in range(3))


@pytest.mark.anyio
async def test_skip_probes_before_the_question_still_asks_it():
    radio = {"type": "radio", "options": ["Ja", "Nej"]}
    guide = InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {"id": FIRST, "main_question": "First?", "survey_item": radio},
                        {
                            "id": SECOND,
                            "main_question": "Second?",
                            "max_probes_n": 2,
                            "conditions": {
                                "action": "skip_probes",
                                "conditions": [_reads(FIRST, (0, 0), "Ja")],
                            },
                        },
                        {"id": THIRD, "main_question": "Third?", "survey_item": radio},
                    ],
                }
            ],
        }
    )
    respondent = Respondent(
        {"First?": ["Ja"], "Second?": ["Because"], "Third?": ["Nej"]}
    )
    interviewer = _interviewer(guide, respondent)
    probed = []

    async def probe(question, section_description):
        probed.append(question.main_question)

    interviewer.probe = probe

    await interviewer.handle_sections()

    assert respondent.asked == ["First?", "Second?", "Third?"]
    assert probed == []
    # Every question keeps the index it has in the guide.
    assert [
        q.main_question.question.message
        for q in interviewer.interview_history[0].questions
    ] == ["First?", "Second?", "Third?"]


def _skip_probes_guide() -> InterviewGuide:
    radio = {"type": "radio", "options": ["Ja", "Nej"]}
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {"id": FIRST, "main_question": "First?", "survey_item": radio},
                        {
                            "id": SECOND,
                            "main_question": "Second?",
                            "max_probes_n": 2,
                            "conditions": {
                                "action": "skip_probes",
                                "conditions": [_reads(FIRST, (0, 0), "Ja")],
                            },
                        },
                    ],
                }
            ],
        }
    )


async def _resumed_at_second(first_answer: str) -> list[str]:
    respondent = Respondent({"Second?": ["Because"]})
    # Shown in the earlier session.
    respondent.asked.extend(["First?", "Second?"])
    interviewer = _interviewer(_skip_probes_guide(), respondent)
    probed = []

    async def probe(question, section_description):
        probed.append(question.main_question)

    interviewer.probe = probe

    await interviewer.process_history(
        [
            StoredMessage("First?", MessageRole.ASSISTANT),
            StoredMessage(first_answer, MessageRole.USER),
            StoredMessage("Second?", MessageRole.ASSISTANT, main_question=1),
        ]
    )

    return probed


@pytest.mark.anyio
async def test_skip_probes_before_the_question_holds_on_resume():
    assert await _resumed_at_second("Ja") == []


@pytest.mark.anyio
async def test_skip_probes_that_did_not_fire_still_probes_on_resume():
    assert await _resumed_at_second("Nej") == ["Second?"]


# ── check_if_answered ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_already_answered_is_checked_from_the_second_question_on():
    guide = InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {
                            "main_question": f"Q{i}?",
                            "max_probes_n": 0,
                            "check_if_answered": True,
                        }
                        for i in range(3)
                    ],
                }
            ],
        }
    )
    respondent = Respondent({f"Q{i}?": ["a"] for i in range(3)})
    interviewer = _interviewer(guide, respondent)
    checked = []

    async def has_question_been_answered(question):
        checked.append(question)
        return False

    interviewer.has_question_been_answered = has_question_been_answered

    await interviewer.handle_sections()

    assert checked == ["Q1?", "Q2?"]


# ── Resuming where nothing is owed ──────────────────────────────────────────


def _plain_guide(**first) -> InterviewGuide:
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {"main_question": "First?", **first},
                        {"main_question": "Second?", "max_probes_n": 0},
                    ],
                }
            ],
        }
    )


async def _resume(guide: InterviewGuide, stored: list[StoredMessage]):
    respondent = Respondent({"Second?": ["a"]})
    interviewer = _interviewer(guide, respondent)
    probed = []

    async def probe(question, section_description):
        probed.append(question.main_question)

    interviewer.probe = probe

    await interviewer.process_history(stored)
    await interviewer.handle_sections()

    return interviewer, respondent, probed


FIRST_ASKED = StoredMessage("First?", MessageRole.ASSISTANT)


@pytest.mark.anyio
async def test_resume_after_a_question_skipped_by_a_condition():
    _, respondent, probed = await _resume(
        _plain_guide(),
        [StoredMessage("First?", MessageRole.ASSISTANT, skipped_by_condition=True)],
    )

    # Not waited on: the respondent never saw it.
    assert respondent.asked == ["Second?"]
    assert probed == []


@pytest.mark.anyio
async def test_resume_after_a_message_that_takes_no_answer():
    _, respondent, probed = await _resume(
        _plain_guide(can_answer=False),
        [StoredMessage("First?", MessageRole.ASSISTANT, can_answer=False)],
    )

    assert respondent.asked == ["Second?"]
    assert probed == []


@pytest.mark.anyio
async def test_resume_after_the_respondent_skipped():
    _, respondent, probed = await _resume(
        _plain_guide(), [FIRST_ASKED, StoredMessage(SKIP, MessageRole.USER)]
    )

    assert respondent.asked == ["Second?"]
    assert probed == []


@pytest.mark.anyio
async def test_resume_after_a_timed_message():
    timed = StoredMessage(
        "Halfway", MessageRole.ASSISTANT, section=None, main_question=None, timed=True
    )
    interviewer, respondent, _ = await _resume(
        _plain_guide(), [FIRST_ASKED, StoredMessage("a", MessageRole.USER), timed]
    )

    assert respondent.asked == ["Second?"]
    assert len(interviewer.interview_history.timed_messages) == 1


@pytest.mark.anyio
async def test_resume_after_a_timed_message_before_any_question():
    timed = StoredMessage(
        "Hello", MessageRole.ASSISTANT, section=None, main_question=None, timed=True
    )
    respondent = Respondent({"First?": ["a"], "Second?": ["a"]})
    interviewer = _interviewer(_plain_guide(max_probes_n=0), respondent)

    await interviewer.process_history([timed])
    await interviewer.handle_sections()

    assert respondent.asked == ["First?", "Second?"]


@pytest.mark.anyio
async def test_resume_does_not_probe_a_question_without_probing():
    _, respondent, probed = await _resume(
        _plain_guide(max_probes_n=None),
        [FIRST_ASKED, StoredMessage("a", MessageRole.USER)],
    )

    assert respondent.asked == ["Second?"]
    assert probed == []


@pytest.mark.anyio
async def test_resume_probes_a_question_with_probing():
    _, _, probed = await _resume(
        _plain_guide(max_probes_n=2),
        [FIRST_ASKED, StoredMessage("a", MessageRole.USER)],
    )

    assert probed == ["First?"]


# ── Timed messages ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_timed_message_counts_time_from_earlier_sessions():
    guide = _plain_guide(max_probes_n=0)
    guide.timed_messages = [TimedMessage(message="Halfway", time=150)]
    respondent = Respondent({"First?": ["a"], "Second?": ["a"]})
    interviewer = _interviewer(guide, respondent, previous_time_spent=100)
    # 60 seconds into this session, 160 into the interview.
    interviewer.interview_started = datetime.now(UTC) - timedelta(seconds=60)

    await interviewer.handle_sections()

    assert [m.message for m in interviewer.interview_history.timed_messages] == [
        "Halfway"
    ]


def _stored_rows(interviewer) -> list[StoredMessage]:
    """What the interviewer wrote to the database, as a resume reads it."""
    fields = (
        "section",
        "main_question",
        "sub_question",
        "timed",
        "can_answer",
        "skipped_by_condition",
    )
    rows = []
    for call in interviewer.db.insert_message.call_args_list:
        kwargs = call.kwargs
        rows.append(
            StoredMessage(
                kwargs["content"],
                kwargs["role"],
                **{field: kwargs[field] for field in fields if field in kwargs},
            )
        )
    return rows


@pytest.mark.anyio
async def test_session_cancelled_after_a_timed_message_resumes_at_the_probe(
    monkeypatch,
):
    async def sleep(seconds):
        # The pause that lets a timed message be read before what follows: a
        # reconnect cancels the old session while it waits.
        if seconds == 2.5:
            raise asyncio.CancelledError

    monkeypatch.setattr("ainterviewer.interview.asyncio.sleep", sleep)

    guide = _plain_guide(max_probes_n=2)
    respondent = Respondent({"First?": ["a"]})
    interviewer = _interviewer(guide, respondent)
    interviewer.interview_started = datetime.now(UTC) - timedelta(seconds=60)

    async def generate_probe(section_description, question, transcript):
        # Falls due between the main question and its first probe.
        guide.timed_messages = [TimedMessage(message="Halfway", time=30)]
        return "Why?"

    interviewer.generate_probe = generate_probe

    with pytest.raises(asyncio.CancelledError):
        await interviewer.handle_sections()

    # The probe was never sent, but it was stored with the timed message.
    assert respondent.asked == ["First?", "Halfway"]
    stored = _stored_rows(interviewer)
    assert [(row.content, row.timed) for row in stored[-2:]] == [
        ("Halfway", True),
        ("Why?", False),
    ]

    resumed_respondent = Respondent({"Why?": ["Because"]})
    # Replayed from the stored history on reconnect.
    resumed_respondent.asked.extend(["First?", "Why?"])
    resumed = _interviewer(_plain_guide(max_probes_n=2), resumed_respondent)
    probed = []

    async def probe(question, section_description):
        probed.append(question.main_question)

    resumed.probe = probe

    await resumed.process_history(stored)

    # The probe is answered, and probing goes on with the budget left.
    assert resumed_respondent.answers["Why?"] == []
    assert probed == ["First?"]


# ── Exhaustion ──────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_exhaustion_is_judged_with_the_section_description():
    guide = _plain_guide()
    interviewer = _interviewer(guide, Respondent({}))
    question = guide.question_sections[0].questions[0]
    history = interviewer.interview_history
    history.add_section("Section")
    history.add_question(
        question_description="About the first",
        main_question=Turn(question=HistoryMessage(message="First?")),
    )
    contexts = []

    async def classify(
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples=None,
    ) -> bool:
        contexts.append(json.loads(text))
        return False

    interviewer.classification_agent.classify = classify

    await interviewer.has_main_question_been_exhausted(question, "The section")

    assert contexts[0]["section_description"] == "The section"
