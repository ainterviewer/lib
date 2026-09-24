import asyncio
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ainterviewer.agents import SecurityAgent
from ainterviewer.agents.config import AgentConfigs
from ainterviewer.agents.security_policy import (
    SecurityAssessmentResult,
    SecurityDecision,
    SecurityPolicy,
    TriggeredDecision,
    default_security_policy,
)
from ainterviewer.config import InterviewConfig
from ainterviewer.exceptions import EndInterviewCondition, SkipSectionCondition
from ainterviewer.interfaces import (
    OutgoingMessage,
    SecurityIntervention,
    SecurityOverride,
)
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.history import HistoryMessage, Turn
from ainterviewer.interview_guides.types import ConditionAction, SecurityAction
from ainterviewer.types import MessageRole, MessageType

# ── Helpers ─────────────────────────────────────────────────────────────────


def _decision(
    name: str = "risk",
    threshold: float = 0.5,
    action: SecurityAction = ConditionAction.END_INTERVIEW,
    action_text: str = "Intervention",
) -> SecurityDecision:
    return SecurityDecision(
        name=name,
        description=f"Is there {name}?",
        threshold=threshold,
        action=action,
        action_text=action_text,
    )


def _assessment(policy: SecurityPolicy, probabilities: dict[str, float]):
    return policy.assessment_model.model_validate(
        {
            decision.name: {
                "reasoning": "",
                "probability": probabilities.get(decision.name, 0.0),
            }
            for decision in policy.decisions
        }
    )


def _result(
    policy: SecurityPolicy, probabilities: dict[str, float]
) -> SecurityAssessmentResult:
    assessment = _assessment(policy, probabilities)
    return SecurityAssessmentResult(
        assessment=assessment,
        triggered=[
            TriggeredDecision(decision, getattr(assessment, decision.name))
            for decision in policy.decisions
            if probabilities.get(decision.name, 0.0) >= decision.threshold
        ],
    )


class Harness:
    """An interviewer with the respondent, the database and every LLM call
    replaced, running a single question with up to two probes.

    `assessments` gives the probabilities the security agent returns, one
    entry per answer in the order they are given, and `override_answers` what
    the respondent answers each intervention they may override.
    """

    def __init__(
        self,
        assessments: list[dict[str, float]] | None = None,
        probe_answers: tuple[str, ...] = ("probe answer 1", "probe answer 2"),
        guide_kwargs: dict | None = None,
        override_answers: tuple[str, ...] = (),
    ):
        self.assessments = list(assessments or [])
        self.probe_answers = list(probe_answers)
        self.override_answers = list(override_answers)
        self.events: list[str] = []
        self.sent: list[tuple[str, bool]] = []
        self.sent_kwargs: list[dict] = []

        guide = InterviewGuide.model_validate(
            {
                "framing": "Framing",
                "question_sections": [
                    {
                        "description": "Section",
                        "questions": [{"main_question": "Q?", "max_probes_n": 2}],
                    }
                ],
                **(guide_kwargs or {}),
            }
        )
        self.db = MagicMock()
        self.io = MagicMock(
            send_data=AsyncMock(),
            receive_message=AsyncMock(side_effect=self.receive_message),
        )
        self.interviewer = iv = AInterviewer(
            io=self.io,
            db=self.db,
            interview_guide=guide,
            config=InterviewConfig.model_validate({}),
            agent_configs=AgentConfigs.model_validate({"security": {"include": True}}),
            project_id=uuid.uuid4(),
            interview_id=uuid.uuid4(),
        )
        assert iv.security_agent is not None
        self.policy = iv.security_agent.policy

        self.patch_agent("assess", self.assess)
        for name, replacement in [
            ("send_data", self.send_data),
            ("ask_question", self.ask_question),
            ("ask_probe", self.ask_probe),
            ("can_probe", self.can_probe),
            ("generate_probe", self.generate_probe),
            ("handle_skip_question_exception", self.handle_skip_question_exception),
            ("_emit_qa_pair", self.noop),
            ("_emit_chunk", self.noop),
            ("send_progress", self.noop),
        ]:
            self.patch(name, replacement)

    def patch(self, name: str, replacement) -> None:
        setattr(self.interviewer, name, replacement)

    def patch_agent(self, name: str, replacement) -> None:
        setattr(self.interviewer.security_agent, name, replacement)

    async def assess(self, **kwargs) -> SecurityAssessmentResult:
        self.events.append("assess")
        return _result(self.policy, self.assessments.pop(0) if self.assessments else {})

    async def receive_message(self, message_id: int, message_type=None):
        # Only the override answer is read from the respondent directly.
        assert message_type == MessageType.SECURITY_OVERRIDE
        assert self.override_answers, "Waited for an unexpected override answer"
        self.events.append("receive_override")
        return self.override_answers.pop(0), message_type, None

    async def send_data(self, text: str, **kwargs):
        self.sent.append((text, kwargs.get("outro", False)))
        self.sent_kwargs.append(kwargs)

    async def ask_question(self, question: Question) -> str:
        self.events.append("ask_question")
        history = self.interviewer.interview_history
        history.add_question(
            question_description=question.description,
            main_question=Turn(question=HistoryMessage(message=question.main_question)),
        )
        history.add_answer(HistoryMessage(message="answer"))
        return "answer"

    async def ask_probe(self, question: Question, probe: str) -> str:
        self.events.append("ask_probe")
        history = self.interviewer.interview_history
        history.add_probe(probe=Turn(question=HistoryMessage(message=probe)))
        answer = self.probe_answers.pop(0)
        history.add_answer(HistoryMessage(message=answer))
        return answer

    async def can_probe(self, question: Question) -> bool:
        return bool(self.probe_answers)

    async def generate_probe(self, **kwargs) -> str:
        self.events.append("generate_probe")
        return "Probe?"

    async def handle_skip_question_exception(self, question: Question):
        self.events.append("handle_skip_question_exception")

    async def noop(self, *args, **kwargs):
        pass

    async def handle_question(self, question: Question | None = None):
        iv = self.interviewer
        iv.interview_history.add_section("Section")
        question = question or iv.interview_guide.question_sections[0].questions[0]
        await iv.handle_question(question, "Section", question_index=0)

    @property
    def stored_overrides(self) -> list[dict]:
        return [
            call.kwargs
            for call in self.db.insert_message.call_args_list
            if call.kwargs["message_type"] == MessageType.SECURITY_OVERRIDE
        ]

    @property
    def logged_assessments(self) -> list:
        return [
            call.kwargs
            for call in self.db.insert_task.call_args_list
            if call.kwargs["task"] == "assess_security"
        ]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skips the pauses the interviewer takes between messages"""
    sleep = asyncio.sleep

    async def no_sleep(*args, **kwargs):
        await sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)


# ── SecurityPolicy ─────────────────────────────────────────────────────────


class TestSecurityPolicy:
    def test_default_policy_is_valid(self):
        policy = default_security_policy()
        assert list(policy.assessment_model.model_fields) == [
            decision.name for decision in policy.decisions
        ]

    def test_default_policy_is_not_shared(self):
        assert AgentConfigs().security.policy is not AgentConfigs().security.policy

    def test_assessment_model_is_cached(self):
        policy = default_security_policy()
        assert policy.assessment_model is policy.assessment_model

    def test_decisions_required(self):
        with pytest.raises(ValidationError):
            SecurityPolicy(decisions=[])

    def test_name_must_be_identifier(self):
        with pytest.raises(ValidationError, match="not a valid identifier"):
            _decision(name="self harm")

    def test_names_must_be_unique(self):
        with pytest.raises(ValidationError, match="Duplicate"):
            SecurityPolicy(decisions=[_decision(), _decision()])

    def test_skip_question_is_not_an_action(self):
        # The question has always been answered by the time it is assessed.
        with pytest.raises(ValidationError):
            SecurityDecision.model_validate(
                {
                    "name": "risk",
                    "description": "",
                    "threshold": 0.5,
                    "action": ConditionAction.SKIP_QUESTION,
                }
            )

    def test_loads_from_config(self):
        configs = AgentConfigs.model_validate(
            {
                "security": {
                    "policy": {
                        "decisions": [
                            {
                                "name": "risk",
                                "description": "d",
                                "threshold": 0.4,
                                "action": "skip_section",
                            }
                        ]
                    }
                }
            }
        )
        (decision,) = configs.security.policy.decisions
        assert decision.action == ConditionAction.SKIP_SECTION


# ── SecurityAssessmentResult.most_severe ───────────────────────────────────


class TestMostSevere:
    def test_none_triggered(self):
        policy = SecurityPolicy(decisions=[_decision()])
        assert _result(policy, {}).most_severe is None

    def test_most_severe_action_wins(self):
        policy = SecurityPolicy(
            decisions=[
                _decision("a", action=ConditionAction.SKIP_PROBES),
                _decision("b", action=ConditionAction.END_INTERVIEW),
                _decision("c", action=ConditionAction.SKIP_SECTION),
            ]
        )
        result = _result(policy, {"a": 1, "b": 1, "c": 1})
        assert result.most_severe is not None
        assert result.most_severe.decision.name == "b"

    def test_first_in_policy_wins_ties(self):
        policy = SecurityPolicy(
            decisions=[
                _decision("a", action=ConditionAction.SKIP_SECTION),
                _decision("b", action=ConditionAction.SKIP_SECTION),
            ]
        )
        result = _result(policy, {"a": 1, "b": 1})
        assert result.most_severe is not None
        assert result.most_severe.decision.name == "a"


# ── SecurityAgent.assess ───────────────────────────────────────────────────


class TestSecurityAgentAssess:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        ("probability", "triggers"), [(0.49, False), (0.5, True), (0.9, True)]
    )
    async def test_threshold(self, monkeypatch, probability, triggers):
        policy = SecurityPolicy(decisions=[_decision(threshold=0.5)])
        agent = SecurityAgent(policy=policy, model="model", language="EN")
        prompts = []

        async def chat_api(messages, response_format=None, **kwargs):
            prompts.append(messages[-1]["content"])
            assert response_format is policy.assessment_model
            return _assessment(policy, {"risk": probability})

        monkeypatch.setattr(agent, "chat_api", chat_api)

        result = await agent.assess(
            interview_framing="Framing",
            section_description="Section",
            question_description="Question",
            transcript="Transcript",
        )

        assert [t.decision.name for t in result.triggered] == (
            ["risk"] if triggers else []
        )
        # The schema is rendered as JSON, not as a Python dict.
        assert '"properties"' in prompts[0]


# ── AInterviewer security checks ───────────────────────────────────────────


class TestInterviewSecurity:
    @pytest.mark.anyio
    async def test_every_answer_is_assessed_and_logged(self):
        harness = Harness()

        await harness.handle_question()

        assert harness.events.count("assess") == 3
        assert harness.events.count("ask_probe") == 2
        assert [log["reason"] for log in harness.logged_assessments] == [None] * 3
        assert harness.sent == []
        assert harness.interviewer._security_check is None

    @pytest.mark.anyio
    async def test_skip_probes_on_main_answer(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("accept",)
        )

        await harness.handle_question()

        assert "ask_probe" not in harness.events
        # Not handled as a skipped question, as it was asked and answered.
        assert "handle_skip_question_exception" not in harness.events
        assert harness.sent == [(harness.policy.decisions[3].action_text, False)]
        assert harness.logged_assessments[0]["reason"] == "uncomfortable"

    @pytest.mark.anyio
    async def test_intervention_is_sent_for_a_modal(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("accept",)
        )

        await harness.handle_question()

        (kwargs,) = harness.sent_kwargs
        assert kwargs["security_intervention"] == SecurityIntervention(
            action=ConditionAction.SKIP_PROBES, respondent_override=True
        )
        # Answered in the modal, never in the chat.
        assert kwargs["can_answer"] is False

    @pytest.mark.anyio
    async def test_intervention_is_stored_and_sent_without_decision(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("accept",)
        )
        iv = harness.interviewer
        harness.db.insert_message.return_value = 1
        harness.patch("send_data", partial(AInterviewer.send_data, iv))

        await harness.handle_question()

        (payload,) = [call.args[0] for call in harness.io.send_data.call_args_list]
        assert isinstance(payload, OutgoingMessage)
        assert payload.model_dump()["security_intervention"] == {
            "action": "skip_probes",
            "respondent_override": True,
        }
        assert "uncomfortable" not in payload.model_dump_json()
        # Stored, so a resumed interview replays it in a modal too.
        stored = harness.db.insert_message.call_args_list[0].kwargs
        assert stored["security_intervention"] == payload.security_intervention

    @pytest.mark.anyio
    async def test_skip_probes_on_probe_answer(self):
        harness = Harness(
            assessments=[{}, {"uncomfortable": 0.9}], override_answers=("accept",)
        )

        await harness.handle_question()

        assert harness.events.count("ask_probe") == 1
        assert len(harness.sent) == 1

    @pytest.mark.anyio
    async def test_most_severe_decision_is_acted_on(self):
        harness = Harness(assessments=[{"uncomfortable": 0.9, "self_harm": 0.9}])

        with pytest.raises(EndInterviewCondition):
            await harness.handle_question()

        assert harness.sent == [(harness.policy.decisions[0].action_text, False)]
        assert harness.logged_assessments[0]["reason"] == "self_harm"

    @pytest.mark.anyio
    async def test_skip_section(self):
        harness = Harness(assessments=[{"risk": 0.9}])
        assert harness.interviewer.security_agent is not None
        harness.interviewer.security_agent.policy = harness.policy = SecurityPolicy(
            decisions=[_decision(action=ConditionAction.SKIP_SECTION)]
        )

        with pytest.raises(SkipSectionCondition):
            await harness.handle_question()

    @pytest.mark.anyio
    async def test_survey_item_answer_is_not_assessed(self):
        harness = Harness(assessments=[{}, {}])
        question = Question.model_validate(
            {"main_question": "Q?", "max_probes_n": 1, "survey_item": {"type": "date"}}
        )
        harness.probe_answers = ["probe answer"]

        await harness.handle_question(question)

        # Only the free-text probe answer.
        assert harness.events.count("assess") == 1

    @pytest.mark.anyio
    async def test_runs_alongside_probe_generation(self):
        harness = Harness()
        probe_generation_started = asyncio.Event()
        generate_probe = harness.generate_probe
        assess = harness.assess

        async def concurrent_generate_probe(**kwargs):
            probe_generation_started.set()
            return await generate_probe(**kwargs)

        async def blocking_assess(**kwargs):
            # The main answer's assessment only finishes if the probe is
            # generated while it runs. The last one has nothing to overlap.
            if "assess" not in harness.events:
                await asyncio.wait_for(probe_generation_started.wait(), timeout=1)
            return await assess(**kwargs)

        harness.patch("generate_probe", concurrent_generate_probe)
        harness.patch_agent("assess", blocking_assess)
        harness.probe_answers = ["probe answer"]

        await harness.handle_question()

        assert harness.events == [
            "ask_question",
            "generate_probe",
            "assess",
            "ask_probe",
            "assess",
        ]

    @pytest.mark.anyio
    async def test_probe_is_not_asked_before_check_resolves(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("accept",)
        )

        await harness.handle_question()

        # Generated alongside the check, but never sent.
        assert harness.events == [
            "ask_question",
            "generate_probe",
            "assess",
            "receive_override",
        ]

    @pytest.mark.anyio
    async def test_pending_check_is_cancelled_on_failure(self):
        harness = Harness()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def hanging_assess(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def failing_generate_probe(**kwargs):
            await started.wait()
            raise RuntimeError("probe generation failed")

        harness.patch_agent("assess", hanging_assess)
        harness.patch("generate_probe", failing_generate_probe)

        # Bounded, as waiting on the check instead of cancelling it never returns.
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(harness.handle_question(), timeout=1)

        assert harness.interviewer._security_check is None
        await asyncio.wait_for(cancelled.wait(), timeout=1)

    @pytest.mark.anyio
    async def test_end_interview_replaces_outros(self):
        harness = Harness(
            assessments=[{"self_harm": 0.9}],
            guide_kwargs={"alt_outro": "Alternative outro", "outro": "Outro"},
        )
        harness.interviewer.interview_guide.introduction = ""

        await harness.interviewer.interview()

        action_text = harness.policy.decisions[0].action_text
        sent = [text for text, _ in harness.sent]
        # The intervention is the last thing the respondent reads.
        assert sent[:1] == [action_text]
        assert "Alternative outro" not in sent
        assert "Outro" not in sent
        assert harness.interviewer.ended_by_security_check


# ── Respondent override ────────────────────────────────────────────────────


class TestRespondentOverride:
    @pytest.mark.anyio
    async def test_override_carries_on_probing(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("override",)
        )

        await harness.handle_question()

        assert harness.events.count("ask_probe") == 2
        (stored,) = harness.stored_overrides
        assert stored["content"] == SecurityOverride.OVERRIDE
        assert stored["role"] == MessageRole.USER
        # Never part of the transcript the agents read.
        assert stored["include_in_history"] is False

    @pytest.mark.anyio
    async def test_accepted_termination_ends_interview(self):
        harness = Harness(
            assessments=[{"termination": 0.9}], override_answers=("accept",)
        )

        with pytest.raises(EndInterviewCondition):
            await harness.handle_question()

        assert harness.interviewer.ended_by_security_check

    @pytest.mark.anyio
    async def test_overridden_termination_carries_on(self):
        harness = Harness(
            assessments=[{"termination": 0.9}], override_answers=("override",)
        )

        await harness.handle_question()

        assert not harness.interviewer.ended_by_security_check
        assert harness.events.count("ask_probe") == 2

    @pytest.mark.anyio
    async def test_no_answer_awaited_without_override(self):
        # The harness fails on any wait it has no answer queued for.
        harness = Harness(assessments=[{"self_harm": 0.9}])

        with pytest.raises(EndInterviewCondition):
            await harness.handle_question()

        assert "receive_override" not in harness.events

    @pytest.mark.anyio
    async def test_unknown_answer_is_rejected(self):
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("maybe",)
        )

        with pytest.raises(ValueError):
            await harness.handle_question()

    @pytest.mark.anyio
    async def test_messages_get_their_own_ids(self):
        # Neither message is part of the transcript, but both are stored, so
        # they must still be counted -- or the next message would reuse an id.
        harness = Harness(
            assessments=[{"uncomfortable": 0.9}], override_answers=("accept",)
        )
        harness.db.insert_message.return_value = 1
        harness.patch("send_data", partial(AInterviewer.send_data, harness.interviewer))

        await harness.handle_question()

        intervention, override = [
            call.kwargs["message_id"]
            for call in harness.db.insert_message.call_args_list
        ]
        # The main question and its answer are 1 and 2.
        assert (intervention, override) == (3, 4)
        assert harness.interviewer.interview_history.current_message_id == 4


# ── Resuming at an intervention ────────────────────────────────────────────


@dataclass
class StoredMessage:
    """The shape a resumed interview reads a stored message as."""

    content: str
    role: MessageRole
    section: int | None = None
    main_question: int | None = None
    sub_question: int | None = None
    message_type: MessageType = MessageType.TEXT
    security_intervention: SecurityIntervention | None = None
    survey_item: Any = None
    is_introduction: bool = False
    outro: bool = False
    timed: bool = False
    skipped_by_condition: bool = False
    include_in_history: bool = True
    image: Any = None
    can_answer: bool = True


def _stored_history(
    action: SecurityAction,
    respondent_override: bool,
    answer: SecurityOverride | None = None,
) -> list[StoredMessage]:
    """A main question and its answer, interrupted by an intervention."""
    history = [
        StoredMessage("Q?", MessageRole.ASSISTANT, 0, 0, 0),
        StoredMessage("answer", MessageRole.USER, 0, 0, 0),
        StoredMessage(
            "Intervention",
            MessageRole.ASSISTANT,
            can_answer=False,
            security_intervention=SecurityIntervention(
                action=action, respondent_override=respondent_override
            ),
        ),
    ]
    if answer is not None:
        history.append(
            StoredMessage(
                answer,
                MessageRole.USER,
                message_type=MessageType.SECURITY_OVERRIDE,
                include_in_history=False,
            )
        )
    return history


class TestResumeAtIntervention:
    @pytest.mark.anyio
    async def test_pending_answer_is_awaited(self):
        harness = Harness(override_answers=("accept",))

        await harness.interviewer.process_history(
            _stored_history(ConditionAction.SKIP_PROBES, respondent_override=True)
        )

        assert harness.events == ["receive_override"]
        # Counted after the intervention it answers.
        assert harness.stored_overrides[0]["message_id"] == 4
        assert harness.interviewer.resume_from_history

    @pytest.mark.anyio
    async def test_pending_override_carries_on_probing(self):
        harness = Harness(override_answers=("override",))

        await harness.interviewer.process_history(
            _stored_history(ConditionAction.SKIP_PROBES, respondent_override=True)
        )

        assert harness.events.count("ask_probe") == 2
        assert harness.interviewer.resume_from_history

    @pytest.mark.anyio
    async def test_given_answer_is_not_asked_again(self):
        harness = Harness()

        with pytest.raises(EndInterviewCondition):
            await harness.interviewer.process_history(
                _stored_history(
                    ConditionAction.END_INTERVIEW,
                    respondent_override=True,
                    answer=SecurityOverride.ACCEPT,
                )
            )

        assert "receive_override" not in harness.events

    @pytest.mark.anyio
    async def test_intervention_without_override_is_applied(self):
        harness = Harness()

        with pytest.raises(EndInterviewCondition):
            await harness.interviewer.process_history(
                _stored_history(
                    ConditionAction.END_INTERVIEW, respondent_override=False
                )
            )

    @pytest.mark.anyio
    async def test_resumed_end_skips_outros(self):
        harness = Harness(guide_kwargs={"alt_outro": "Alternative outro"})

        await harness.interviewer.interview(
            interview_history=_stored_history(
                ConditionAction.END_INTERVIEW, respondent_override=False
            )
        )

        sent = [text for text, _ in harness.sent]
        assert "Alternative outro" not in sent
        assert harness.interviewer.interview_history.is_finished

    @pytest.mark.anyio
    async def test_skipped_section_is_not_resumed(self):
        harness = Harness(
            guide_kwargs={
                "question_sections": [
                    {
                        "description": "First",
                        "questions": [
                            {"main_question": "Q1?"},
                            {"main_question": "Q2?"},
                        ],
                    },
                    {"description": "Second", "questions": [{"main_question": "Q3?"}]},
                ]
            }
        )
        asked = []

        async def ask_question(question):
            asked.append(question.main_question)
            return await harness.ask_question(question)

        harness.patch("ask_question", ask_question)

        await harness.interviewer.process_history(
            _stored_history(ConditionAction.SKIP_SECTION, respondent_override=False)
        )
        await harness.interviewer.handle_sections()

        # Q2 is the rest of the section the intervention skipped.
        assert asked == ["Q3?"]
