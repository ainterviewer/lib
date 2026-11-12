"""The interview loop's own bookkeeping: what a question is asked with, and
what a resumed interview does with the question it stopped at."""

import uuid
from unittest.mock import AsyncMock

import pytest

from ainterviewer.exceptions import EndInterviewCondition
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.types import MessageRole
from tests.test_condition_branches import Respondent, _interviewer, _reads
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
