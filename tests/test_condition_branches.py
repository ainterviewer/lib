"""Conditions that read a question the respondent was never asked.

A guide that branches -- ask Q2 only if Q1 was "Ja", Q3 only if it was "Nej" --
leaves every later rule reading Q2 or Q3 with one side unanswered. Such a rule
is three-valued: the missing answer is unknown, decided by the rest of the rule
where it can be (`Q1 == "Ja" OR Q3 == "Nej"` fires on Q1 alone) and never
firing on its own. The guide here is the shape that used to stop the interview
with "Answer not found".
"""

import uuid
from unittest.mock import AsyncMock

import pytest

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.config import InterviewConfig
from ainterviewer.events import InputRequested, MessageEmitted, UserMessage
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    ConditionEvaluator,
    Conditions,
    QuestionContext,
    combine,
)
from ainterviewer.interview_guides.types import ConditionAction
from ainterviewer.lpm.types import CustomToken
from tests.fakes import FakeHost

SKIP = str(CustomToken.skip_question)

# ── combine ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("left", "operator", "right", "expected"),
    [
        (True, "OR", None, True),
        (None, "OR", True, True),
        (False, "OR", None, None),
        (None, "OR", None, None),
        (False, "AND", None, False),
        (None, "AND", False, False),
        (True, "AND", None, None),
        (None, "AND", None, None),
        (True, "AND", True, True),
        (False, "OR", False, False),
    ],
)
def test_combine_is_kleene(left, operator, right, expected):
    assert combine(left, operator, right) is expected


def _condition(value: str, negated: bool = False, combine_next=None) -> Condition:
    return Condition(
        question_context=QuestionContext(
            question_index=(0, 0), question_id=uuid.uuid4()
        ),
        evaluation=[ConditionEvaluation(trigger_value=value)],
        negated=negated,
        combine_next=combine_next,
    )


class TestUnknownContext:
    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    @pytest.mark.anyio
    async def test_unanswered_is_unknown(self, evaluator):
        assert await evaluator.evaluate_condition(None, _condition("Nej")) is None

    @pytest.mark.anyio
    async def test_negating_unknown_stays_unknown(self, evaluator):
        condition = _condition("Nej", negated=True)
        assert await evaluator.evaluate_condition(None, condition) is None

    @pytest.mark.anyio
    async def test_or_decided_by_the_answered_side(self, evaluator):
        conditions = Conditions(
            action=ConditionAction.SKIP_QUESTION,
            conditions=[_condition("Ja", combine_next="OR"), _condition("Nej")],
        )
        assert await evaluator.evaluate_conditions(["Ja", None], conditions) is True
        assert await evaluator.evaluate_conditions(["Nej", None], conditions) is None

    @pytest.mark.anyio
    async def test_and_does_not_fire_on_unknown(self, evaluator):
        conditions = Conditions(
            action=ConditionAction.SKIP_QUESTION,
            conditions=[_condition("Ja", combine_next="AND"), _condition("Nej")],
        )
        assert await evaluator.evaluate_conditions(["Ja", None], conditions) is None
        assert await evaluator.evaluate_conditions(["Nej", None], conditions) is False


# ── The interview loop ──────────────────────────────────────────────────────

# Simon's guide, cut down to what the conditions need: no probing, no LLM.
Q1, Q2, Q3, Q4, Q5 = (uuid.uuid4() for _ in range(5))
CHOICES = ["Ja", "Nej", "Ved ikke / måske"]


def _reads(question_id: uuid.UUID, index, value: str, combine_next=None) -> dict:
    return {
        "question_context": {"question_id": question_id, "question_index": index},
        "evaluation": [{"trigger_value": value}],
        "combine_next": combine_next,
    }


def _skip_if(*conditions: dict) -> dict:
    return {"action": "skip_question", "conditions": list(conditions)}


def _guide(**q1_overrides) -> InterviewGuide:
    radio = {"type": "radio", "options": CHOICES}
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Children",
                    "questions": [
                        {
                            "id": Q1,
                            "main_question": "Har du børn?",
                            "survey_item": {"type": "radio", "options": ["Ja", "Nej"]},
                            **q1_overrides,
                        },
                        {
                            "id": Q2,
                            "main_question": "Vil du gerne have børn?",
                            "survey_item": radio,
                            "conditions": _skip_if(_reads(Q1, (0, 0), "Ja")),
                        },
                        {
                            "id": Q3,
                            "main_question": "Vil du gerne have flere børn?",
                            "survey_item": radio,
                            "conditions": _skip_if(_reads(Q1, (0, 0), "Nej")),
                        },
                    ],
                },
                {
                    "description": "Considerations",
                    "questions": [
                        {
                            "id": Q4,
                            "main_question": "Hvad betyder mest for at få børn?",
                            "max_probes_n": 0,
                            "conditions": _skip_if(
                                _reads(Q1, (0, 0), "Ja", combine_next="OR"),
                                _reads(Q2, (0, 1), "Nej"),
                            ),
                        },
                        {
                            "id": Q5,
                            "main_question": "Hvad betyder mest for at få flere børn?",
                            "max_probes_n": 0,
                            "conditions": _skip_if(
                                _reads(Q1, (0, 0), "Nej", combine_next="OR"),
                                _reads(Q3, (0, 2), "Nej"),
                            ),
                        },
                    ],
                },
            ],
        }
    )


class Respondent:
    """Answers each question from a script, keyed by its text, and records
    what the interview sent."""

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = {question: list(given) for question, given in answers.items()}
        self.host = FakeHost(self.answer)

    @property
    def frames(self) -> list[MessageEmitted]:
        # A question skipped by a condition is recorded but never shown.
        return [
            frame
            for frame in self.host.of(MessageEmitted)
            if not frame.skipped_by_condition
        ]

    @property
    def asked(self) -> list[str]:
        return [frame.content for frame in self.frames]

    async def answer(self, request: InputRequested) -> UserMessage:
        given = self.answers[self.asked[-1]].pop(0)
        if given == SKIP:
            return UserMessage(message_id=request.message_id, skip=True)
        return UserMessage(message_id=request.message_id, content=given)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("ainterviewer.interview.asyncio.sleep", AsyncMock())


def _interviewer(guide: InterviewGuide, respondent: Respondent) -> AInterviewer:
    return AInterviewer(
        host=respondent.host,
        interview_guide=guide,
        config=InterviewConfig.model_validate({}),
        agent_configs=AgentConfigs.model_validate({}),
        project_id=uuid.uuid4(),
        interview_id=uuid.uuid4(),
    )


async def _run(answers: dict[str, list[str]], **q1_overrides) -> Respondent:
    respondent = Respondent(answers)
    await _interviewer(_guide(**q1_overrides), respondent).handle_sections()
    return respondent


FREE = {
    "Hvad betyder mest for at få børn?": ["Tid"],
    "Hvad betyder mest for at få flere børn?": ["Plads"],
}


@pytest.mark.anyio
async def test_or_fires_on_the_answered_branch():
    # Q2 is never asked, so Q4's `Q1 == Ja OR Q2 == Nej` is decided by Q1.
    respondent = await _run(
        {"Har du børn?": ["Ja"], "Vil du gerne have flere børn?": ["Ja"], **FREE}
    )
    assert respondent.asked == [
        "Har du børn?",
        "Vil du gerne have flere børn?",
        "Hvad betyder mest for at få flere børn?",
    ]


@pytest.mark.anyio
async def test_branch_answer_still_decides_when_it_was_asked():
    respondent = await _run(
        {"Har du børn?": ["Nej"], "Vil du gerne have børn?": ["Nej"], **FREE}
    )
    assert respondent.asked == ["Har du børn?", "Vil du gerne have børn?"]


@pytest.mark.anyio
async def test_respondent_skip_reads_as_unknown():
    # Nothing decides Q2's or Q3's rule, so neither fires and both are asked;
    # Q4 and Q5 likewise stay undecided and are asked.
    respondent = await _run(
        {
            "Har du børn?": [SKIP],
            "Vil du gerne have børn?": ["Ja"],
            "Vil du gerne have flere børn?": ["Ja"],
            **FREE,
        }
    )
    assert respondent.asked == [
        "Har du børn?",
        "Vil du gerne have børn?",
        "Vil du gerne have flere børn?",
        "Hvad betyder mest for at få børn?",
        "Hvad betyder mest for at få flere børn?",
    ]


# ── can_skip ────────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_can_skip_rides_on_the_frame():
    respondent = await _run(
        {"Har du børn?": ["Ja"], "Vil du gerne have flere børn?": ["Ja"], **FREE},
        can_skip=False,
    )
    assert [frame.can_skip for frame in respondent.frames] == [False, True, True]


@pytest.mark.anyio
async def test_skip_of_an_unskippable_question_is_ignored():
    # The stale skip is dropped and the answer after it is the one that counts.
    respondent = await _run(
        {
            "Har du børn?": [SKIP, "Nej"],
            "Vil du gerne have børn?": ["Nej"],
            **FREE,
        },
        can_skip=False,
    )
    assert respondent.asked == ["Har du børn?", "Vil du gerne have børn?"]
    assert respondent.answers["Har du børn?"] == []
