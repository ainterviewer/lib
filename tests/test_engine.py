"""`AInterviewer` driven end to end through an `EngineHost`.

The engine's only contact with the outside world is the events it emits and
the inputs it asks for, so these pin both: which events an interview produces,
in which order, and what it asks the respondent for.
"""

import asyncio
import uuid

import pytest

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.config import InterviewConfig
from ainterviewer.events import (
    ChunkEmitted,
    GuideUpdated,
    Init,
    InputKind,
    InputRequested,
    InterviewEnded,
    MessageEmitted,
    Progress,
    SecurityOverrideAnswer,
    UserMessage,
    UserMessageRecorded,
)
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.interview_guides.types import SecurityOverride
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import EmbeddingKind, MessageRole, MessageType
from tests.fakes import FakeHost

PROJECT_ID = uuid.uuid4()
INTERVIEW_ID = uuid.uuid4()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skips the pauses the interviewer takes between messages"""
    sleep = asyncio.sleep

    async def no_sleep(*args, **kwargs):
        await sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)


def make_guide() -> InterviewGuide:
    """An introduction, a free-text question, a survey question and an outro,
    none of which needs a language model."""
    return InterviewGuide.model_validate(
        {
            "introduction": "Hello",
            "outro": "Goodbye",
            "question_sections": [
                {
                    "description": "Section",
                    "questions": [
                        {
                            "main_question": "Free?",
                            "max_probes_n": 0,
                            "check_if_answered": False,
                            "create_segue": False,
                        },
                        {
                            "main_question": "Agree?",
                            "survey_item": {
                                "type": "likert",
                                "options": ["No", "Yes"],
                            },
                        },
                    ],
                }
            ],
        }
    )


def answering(*answers: UserMessage | str) -> FakeHost:
    """A host answering each request with the next of `answers`; a string is
    sent as plain text."""
    queue = list(answers)

    async def answer(request: InputRequested):
        assert queue, f"Unexpected input request: {request}"
        given = queue.pop(0)
        if isinstance(given, str):
            return UserMessage(message_id=request.message_id, content=given)
        return given.model_copy(update={"message_id": request.message_id})

    return FakeHost(answer=answer)


def interviewer(host: FakeHost, guide: InterviewGuide | None = None) -> AInterviewer:
    return AInterviewer(
        host=host,
        interview_guide=guide or make_guide(),
        config=InterviewConfig.model_validate({}),
        agent_configs=AgentConfigs.model_validate({}),
        project_id=PROJECT_ID,
        interview_id=INTERVIEW_ID,
    )


async def run(host: FakeHost, history: list[HistoryRecord] | None = None) -> None:
    await interviewer(host).interview(interview_history=history)


def summary(host: FakeHost) -> list[str]:
    """The emitted events, compactly."""
    out = []
    for event in host.events:
        match event:
            case MessageEmitted():
                out.append(f"message {event.message_id} {event.content}")
            case UserMessageRecorded():
                out.append(f"answer {event.message_id} {event.content}")
            case ChunkEmitted():
                out.append(f"chunk {event.chunk.kind}")
            case Progress():
                out.append(f"progress {event.progress:g}")
            case _:
                out.append(event.type)
    return out


class TestEventSequence:
    @pytest.mark.anyio
    async def test_a_whole_interview(self):
        host = answering("My answer", "Yes")

        await run(host)

        assert summary(host) == [
            "message 1 Hello",
            "message 2 Free?",
            "answer 3 My answer",
            "chunk message",
            "chunk qa_pair",
            "message 4 Agree?",
            "answer 5 Yes",
            "message 6 Goodbye",
            "progress 100",
            "chunk interview",
            "interview_ended",
        ]

    @pytest.mark.anyio
    async def test_inputs_are_requested_by_kind(self):
        host = answering("My answer", "Yes")

        await run(host)

        assert host.requests == [
            InputRequested(message_id=3, expects=InputKind.FREE_TEXT),
            InputRequested(message_id=5, expects=InputKind.SURVEY_ITEM),
        ]
        # Requests are delivered by the host, never emitted as well.
        assert not host.of(InputRequested)

    @pytest.mark.anyio
    async def test_answers_are_recorded_in_place(self):
        host = answering(
            UserMessage(
                message_id=0, content="  spoken  ", message_type=MessageType.AUDIO
            ),
            "Yes",
        )

        await run(host)

        free, survey = host.of(UserMessageRecorded)
        assert (free.content, free.message_type) == ("spoken", MessageType.AUDIO)
        assert (free.section, free.main_question, free.sub_question) == (0, 0, 0)
        # The engine knows a survey answer is one; the host's word is not
        # needed for it.
        assert survey.message_type == MessageType.SURVEY_ITEM
        assert (survey.section, survey.main_question) == (0, 1)

    @pytest.mark.anyio
    async def test_messages_carry_their_question(self):
        host = answering("My answer", "Yes")
        guide = make_guide()

        await interviewer(host, guide).interview()

        intro, free, survey, outro = host.of(MessageEmitted)
        assert intro.is_introduction and intro.question_id is None
        assert free.question_id == guide.question_sections[0].questions[0].id
        assert survey.survey_item is not None
        assert survey.message_type == MessageType.SURVEY_ITEM
        assert survey.progress == 50
        assert outro.outro and not outro.can_answer

    @pytest.mark.anyio
    async def test_message_chunk_follows_its_answer(self):
        host = answering("My answer", "Yes")

        await run(host)

        index, recorded = next(
            (i, e)
            for i, e in enumerate(host.events)
            if isinstance(e, UserMessageRecorded)
        )
        chunk = host.events[index + 1]
        assert isinstance(chunk, ChunkEmitted)
        assert chunk.chunk.kind == EmbeddingKind.MESSAGE
        assert chunk.chunk.message_id == recorded.message_id

    @pytest.mark.anyio
    async def test_an_unfinished_session_does_not_end_the_interview(self):
        # The host tells an unfinished session from a finished one by the
        # absence of `InterviewEnded` alone.
        host = answering("My answer")

        with pytest.raises(AssertionError, match="Unexpected input request"):
            await run(host)

        assert not host.of(InterviewEnded)


class TestInputs:
    @pytest.mark.anyio
    async def test_skip_is_recorded_as_the_skip_token(self):
        host = answering(UserMessage(message_id=0, skip=True), "Yes")

        await run(host)

        skipped, _ = host.of(UserMessageRecorded)
        assert skipped.content == CustomToken.skip_question
        # A control token is not embeddable text, and a survey answer is not
        # either, which leaves nothing at all to embed.
        assert host.of(ChunkEmitted) == []

    @pytest.mark.anyio
    async def test_an_answer_to_another_message_is_refused(self):
        async def stale(request: InputRequested):
            return UserMessage(message_id=request.message_id - 1, content="x")

        with pytest.raises(ValueError, match="expected one to 3"):
            await run(FakeHost(answer=stale))

    @pytest.mark.anyio
    async def test_the_wrong_kind_of_answer_is_refused(self):
        async def override(request: InputRequested):
            return SecurityOverrideAnswer(
                message_id=request.message_id, answer=SecurityOverride.ACCEPT
            )

        with pytest.raises(TypeError, match="Expected UserMessage"):
            await run(FakeHost(answer=override))


class TestResume:
    @pytest.mark.anyio
    async def test_resumes_after_the_last_answer(self):
        guide = make_guide()
        history = [
            HistoryRecord(
                message_id=1,
                role=MessageRole.ASSISTANT,
                content="Hello",
                is_introduction=True,
                can_answer=False,
            ),
            HistoryRecord(
                message_id=2,
                role=MessageRole.ASSISTANT,
                content="Free?",
                section=0,
                main_question=0,
                sub_question=0,
            ),
            HistoryRecord(
                message_id=3,
                role=MessageRole.USER,
                content="My answer",
                section=0,
                main_question=0,
                sub_question=0,
            ),
        ]
        host = answering("Yes")

        await interviewer(host, guide).interview(interview_history=history)

        # Nothing already stored is emitted again.
        assert summary(host)[:2] == [
            "progress 0",
            "message 4 Agree?",
        ]
        assert host.requests == [
            InputRequested(message_id=5, expects=InputKind.SURVEY_ITEM)
        ]

    @pytest.mark.anyio
    async def test_waits_for_the_pending_survey_answer(self):
        history = [
            HistoryRecord(
                message_id=1,
                role=MessageRole.ASSISTANT,
                content="Free?",
                section=0,
                main_question=0,
                sub_question=0,
            ),
            HistoryRecord(
                message_id=2,
                role=MessageRole.USER,
                content="My answer",
                section=0,
                main_question=0,
                sub_question=0,
            ),
            HistoryRecord(
                message_id=3,
                role=MessageRole.ASSISTANT,
                content="Agree?",
                section=0,
                main_question=1,
                sub_question=0,
            ),
        ]
        host = answering("Yes")

        await run(host, history)

        # The survey item comes from the guide; the record has none.
        assert host.requests == [
            InputRequested(message_id=4, expects=InputKind.SURVEY_ITEM)
        ]
        assert host.of(UserMessageRecorded)[0].message_type == MessageType.SURVEY_ITEM


class TestGuideUpdated:
    @pytest.mark.anyio
    async def test_is_a_snapshot(self):
        host = FakeHost()
        iv = interviewer(host)

        await iv.emit_guide()
        iv.interview_guide.question_sections[0].questions[0].main_question = "Later"

        (event,) = host.of(GuideUpdated)
        assert event.guide.question_sections[0].questions[0].main_question == "Free?"


class TestFromInit:
    def test_builds_the_session_init_describes(self):
        init = Init(
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            language="DA",
            guide=make_guide(),
            config=InterviewConfig.model_validate({}),
            agent_configs=AgentConfigs.model_validate({}),
            prompt_overrides={"probing_agent/system.jinja": "overridden"},
            previous_time_spent=30,
        )

        iv = AInterviewer.from_init(FakeHost(), init)

        assert (iv.project_id, iv.interview_id) == (PROJECT_ID, INTERVIEW_ID)
        assert iv.language == "DA"
        assert iv.referable_values["interview_id"] == INTERVIEW_ID
        env = iv.probing_agent.prompts.env
        assert env.get_template("probing_agent/system.jinja").render() == "overridden"
        # Everything not overridden still comes from the package.
        assert env.list_templates() != ["probing_agent/system.jinja"]
