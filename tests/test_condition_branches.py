"""Conditions that read a question the respondent was never asked.

A guide that branches -- ask Q2 only if Q1 was "Ja", Q3 only if it was "Nej" --
leaves every later rule reading Q2 or Q3 with one side unanswered. Such a rule
is three-valued: the missing answer is unknown, decided by the rest of the rule
where it can be (`Q1 == "Ja" OR Q3 == "Nej"` fires on Q1 alone) and never
firing on its own. The guide here is the shape that used to stop the interview
with "Answer not found".
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.config import InterviewConfig
from ainterviewer.interfaces import OutgoingMessage
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionBlock,
    ConditionEvaluation,
    ConditionEvaluator,
    QuestionContext,
    combine,
)
from ainterviewer.interview_guides.types import ConditionAction
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import MessageType

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


def _block(value: str, negated: bool = False, combine_next=None) -> ConditionBlock:
    return ConditionBlock(
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
        assert await evaluator.evaluate_block(None, _block("Nej")) is None

    @pytest.mark.anyio
    async def test_negating_unknown_stays_unknown(self, evaluator):
        condition = _block("Nej", negated=True)
        assert await evaluator.evaluate_block(None, condition) is None

    @pytest.mark.anyio
    async def test_or_decided_by_the_answered_side(self, evaluator):
        condition = Condition(
            action=ConditionAction.SKIP_QUESTION,
            blocks=[_block("Ja", combine_next="OR"), _block("Nej")],
        )
        assert await evaluator.evaluate_condition(["Ja", None], condition) is True
        assert await evaluator.evaluate_condition(["Nej", None], condition) is None

    @pytest.mark.anyio
    async def test_and_does_not_fire_on_unknown(self, evaluator):
        condition = Condition(
            action=ConditionAction.SKIP_QUESTION,
            blocks=[_block("Ja", combine_next="AND"), _block("Nej")],
        )
        assert await evaluator.evaluate_condition(["Ja", None], condition) is None
        assert await evaluator.evaluate_condition(["Nej", None], condition) is False


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
    return {"action": "skip_question", "blocks": list(conditions)}


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
                            "conditions": [_skip_if(_reads(Q1, (0, 0), "Ja"))],
                        },
                        {
                            "id": Q3,
                            "main_question": "Vil du gerne have flere børn?",
                            "survey_item": radio,
                            "conditions": [_skip_if(_reads(Q1, (0, 0), "Nej"))],
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
                            "conditions": [
                                _skip_if(
                                    _reads(Q1, (0, 0), "Ja", combine_next="OR"),
                                    _reads(Q2, (0, 1), "Nej"),
                                )
                            ],
                        },
                        {
                            "id": Q5,
                            "main_question": "Hvad betyder mest for at få flere børn?",
                            "max_probes_n": 0,
                            "conditions": [
                                _skip_if(
                                    _reads(Q1, (0, 0), "Nej", combine_next="OR"),
                                    _reads(Q3, (0, 2), "Nej"),
                                )
                            ],
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
        self.asked: list[str] = []
        self.frames: list[OutgoingMessage] = []

    async def send_data(self, data) -> None:
        if isinstance(data, OutgoingMessage):
            self.frames.append(data)
            self.asked.append(data.content)

    async def receive_message(self, message_id: int, message_type=None):
        given = self.answers[self.asked[-1]]
        return given.pop(0), message_type or MessageType.TEXT, None


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("ainterviewer.interview.asyncio.sleep", AsyncMock())


def _interviewer(
    guide: InterviewGuide,
    respondent: Respondent,
    db: MagicMock | None = None,
    **kwargs,
) -> AInterviewer:
    io = MagicMock(
        send_data=AsyncMock(side_effect=respondent.send_data),
        receive_message=AsyncMock(side_effect=respondent.receive_message),
    )
    return AInterviewer(
        io=io,
        db=db or MagicMock(),
        interview_guide=guide,
        config=InterviewConfig.model_validate({}),
        agent_configs=AgentConfigs.model_validate({}),
        project_id=uuid.uuid4(),
        interview_id=uuid.uuid4(),
        **kwargs,
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


# ── Checkbox answers ────────────────────────────────────────────────────────

PETS = uuid.uuid4()
PET_OPTIONS = ["Dog", "Cat", "Yes, a fish"]


def _pets_guide(*evaluations: dict, exclusive: bool = False) -> InterviewGuide:
    """Asks which pets the respondent has, then about dogs, skipping that
    question when the rule reading the pets fires."""
    return InterviewGuide.model_validate(
        {
            "framing": "Framing",
            "question_sections": [
                {
                    "description": "Pets",
                    "questions": [
                        {
                            "id": PETS,
                            "main_question": "Which pets do you have?",
                            "survey_item": {
                                "type": "checkbox",
                                "options": PET_OPTIONS,
                                "with_other": True,
                            },
                        },
                        {
                            "main_question": "Tell me about your dog.",
                            "max_probes_n": 0,
                            "conditions": [
                                _skip_if(
                                    {
                                        "question_context": {
                                            "question_id": PETS,
                                            "question_index": (0, 0),
                                        },
                                        "evaluation": list(evaluations),
                                        "exclusive": exclusive,
                                    }
                                )
                            ],
                        },
                    ],
                }
            ],
        }
    )


async def _skipped(answer: str, *evaluations: dict, exclusive: bool = False) -> bool:
    respondent = Respondent(
        {"Which pets do you have?": [answer], "Tell me about your dog.": ["Woof"]}
    )
    await _interviewer(
        _pets_guide(*evaluations, exclusive=exclusive), respondent
    ).handle_sections()
    return "Tell me about your dog." not in respondent.asked


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("answer", "exclusive", "skipped"),
    [
        ("Dog", False, True),
        ("Cat, Dog", False, True),
        ("Cat", False, False),
        ("Dog", True, True),
        ("Cat, Dog", True, False),
        ("Dog, a lizard", True, False),
    ],
)
async def test_checkbox_includes_or_only(answer, exclusive, skipped):
    assert (
        await _skipped(answer, {"trigger_value": "Dog"}, exclusive=exclusive) is skipped
    )


@pytest.mark.anyio
async def test_checkbox_option_with_a_comma():
    assert await _skipped("Cat, Yes, a fish", {"trigger_value": "Yes, a fish"})
    # "Yes" alone is no option, so it is not read out of "Yes, a fish".
    assert not await _skipped("Yes, a fish", {"trigger_value": "Yes"})
