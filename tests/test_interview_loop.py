"""The interview loop's own bookkeeping: what a question is asked with, and
what a resumed interview does with the question it stopped at."""

import uuid
from unittest.mock import AsyncMock

import pytest

from ainterviewer.interview_guides import InterviewGuide
from tests.test_condition_branches import Respondent, _interviewer


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
