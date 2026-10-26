"""The event contract between an engine and its host.

Hosts talk to engines of other library versions, so what is pinned here is
what the contract promises: every event parses back as itself through its
union, carries a `schema_version`, and tolerates fields it does not know yet.
"""

import typing
from uuid import uuid4

import pytest
from pydantic import BaseModel

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.config import InterviewConfig
from ainterviewer.embedding import EmbeddingChunk
from ainterviewer.events import (
    SCHEMA_VERSION,
    ChunkEmitted,
    Error,
    ErrorCode,
    GuideUpdated,
    Init,
    InputKind,
    InputRequested,
    InterviewEnded,
    MessageEmitted,
    Progress,
    SecurityOverrideAnswer,
    StatusChanged,
    TaskRecorded,
    UserMessage,
    UserMessageRecorded,
    inbound_event_adapter,
    input_event_adapter,
    outbound_event_adapter,
)
from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import LikertItem
from ainterviewer.interview_guides.types import (
    ConditionAction,
    SecurityIntervention,
    SecurityOverride,
)
from ainterviewer.types import EmbeddingKind, InterviewStatus, MessageRole, MessageType

PROJECT_ID = uuid4()
INTERVIEW_ID = uuid4()


def guide() -> InterviewGuide:
    return InterviewGuide(
        question_sections=[
            QuestionSection[Question](
                description="s",
                questions=[
                    Question(description="d", main_question="q0"),
                    Question(
                        description="d",
                        main_question="q1",
                        survey_item=LikertItem(options=["No", "Yes"]),
                    ),
                ],
            )
        ]
    )


INBOUND = [
    Init(
        project_id=PROJECT_ID,
        interview_id=INTERVIEW_ID,
        language="da",
        guide=guide(),
        config=InterviewConfig(),
        agent_configs=AgentConfigs(),
        prompt_overrides={"probing.jinja": "x"},
        history=[
            HistoryRecord(
                message_id=1,
                role=MessageRole.ASSISTANT,
                content="q0",
                section=0,
                main_question=0,
                sub_question=0,
            ),
            HistoryRecord(
                message_id=2,
                role=MessageRole.ASSISTANT,
                content="stop",
                security_intervention=SecurityIntervention(
                    action=ConditionAction.END_INTERVIEW, respondent_override=True
                ),
            ),
        ],
        referable_values={"name": "x"},
        previous_time_spent=12,
    ),
    UserMessage(message_id=3, content="an answer"),
    UserMessage(message_id=3, skip=True),
    SecurityOverrideAnswer(message_id=3, answer=SecurityOverride.OVERRIDE),
]

OUTBOUND = [
    MessageEmitted(
        message_id=1,
        content="q1",
        message_type=MessageType.SURVEY_ITEM,
        survey_item=LikertItem(options=["No", "Yes"]),
        section=0,
        main_question=1,
        sub_question=0,
        progress=50,
        question_id=uuid4(),
    ),
    InputRequested(message_id=2, expects=InputKind.SURVEY_ITEM),
    UserMessageRecorded(
        message_id=2,
        content="Yes",
        message_type=MessageType.SURVEY_ITEM,
        section=0,
        main_question=1,
        sub_question=0,
    ),
    TaskRecorded(message_id=2, task="assess_security", response="{}", model="m"),
    ChunkEmitted(
        chunk=EmbeddingChunk(
            kind=EmbeddingKind.MESSAGE,
            text="x",
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            message_id=2,
        )
    ),
    StatusChanged(status=InterviewStatus.INACTIVE, time_spent=3),
    GuideUpdated(guide=guide()),
    Progress(progress=100),
    InterviewEnded(),
    Error(code=ErrorCode.INFERENCE_ERROR, detail="timeout"),
]


@pytest.mark.parametrize("event", INBOUND, ids=lambda e: e.type)
def test_inbound_round_trips(event: BaseModel):
    parsed = inbound_event_adapter.validate_json(event.model_dump_json())

    assert type(parsed) is type(event)
    assert parsed == event


@pytest.mark.parametrize("event", OUTBOUND, ids=lambda e: e.type)
def test_outbound_round_trips(event: BaseModel):
    parsed = outbound_event_adapter.validate_json(event.model_dump_json())

    assert type(parsed) is type(event)
    assert parsed == event


@pytest.mark.parametrize("event", INBOUND[1:], ids=lambda e: e.type)
def test_inputs_parse_as_inputs(event: BaseModel):
    assert input_event_adapter.validate_json(event.model_dump_json()) == event


def test_input_union_refuses_init():
    with pytest.raises(ValueError):
        input_event_adapter.validate_json(INBOUND[0].model_dump_json())


@pytest.mark.parametrize("event", INBOUND + OUTBOUND, ids=lambda e: e.type)
def test_every_event_carries_its_schema_version(event: BaseModel):
    assert event.model_dump()["schema_version"] == SCHEMA_VERSION


def test_unknown_fields_are_ignored():
    # A newer peer may send fields this version does not know yet.
    data = InputRequested(message_id=1, expects=InputKind.FREE_TEXT).model_dump(
        mode="json"
    ) | {"added_later": True}

    assert outbound_event_adapter.validate_python(data) == InputRequested(
        message_id=1, expects=InputKind.FREE_TEXT
    )


def test_history_record_carries_no_guide_data():
    # Resume looks guide data up in the engine's own guide, so a host never
    # has to produce guide models in another version's shape.
    guide_types = {InterviewGuide, Question}

    def mentions(annotation) -> set[type]:
        found = {annotation} if isinstance(annotation, type) else set()
        for arg in typing.get_args(annotation):
            found |= mentions(arg)
        return found

    for name, field in HistoryRecord.model_fields.items():
        referenced = mentions(field.annotation)
        assert not referenced & guide_types, name
        assert not any(
            t.__module__.startswith(
                (
                    "ainterviewer.interview_guides.survey_items",
                    "ainterviewer.interview_guides.media",
                )
            )
            for t in referenced
        ), name
