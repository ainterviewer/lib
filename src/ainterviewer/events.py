"""The events an interview exchanges with its host.

`AInterviewer` reads input events and emits output events through an
`EngineHost`, and does nothing else with the outside world: no database, no
sockets, no embedding backend. Whatever a host does with an event -- store it,
forward it to a respondent, queue it for embedding -- is its own business.

These models are a contract between a host and engines of *other* library
versions, so they only ever change additively: new optional fields and new
event types are fine, renames and removals are not. Every event carries the
`schema_version` it was built under, which is what `ainterviewer.compat` keys
its upgrades on when a breaking change cannot be avoided.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import UUID4, BaseModel, Field, TypeAdapter

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.compat import SCHEMA_VERSION
from ainterviewer.config import InterviewConfig
from ainterviewer.embedding import EmbeddingChunk
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.interview_guides.media import Image
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.interview_guides.types import SecurityIntervention, SecurityOverride
from ainterviewer.types import LanguageCode, MessageType


class _Event(BaseModel):
    schema_version: int = SCHEMA_VERSION


# ── Inbound ──────────────────────────────────────────────────────────────────


class Init(_Event):
    """Everything an engine needs to run, or resume, one interview session."""

    type: Literal["init"] = "init"
    project_id: UUID4
    interview_id: UUID4
    language: LanguageCode = "EN"
    # The interview's own guide, already rendered for `language` and shuffled.
    guide: InterviewGuide
    config: InterviewConfig
    agent_configs: AgentConfigs
    # Prompt templates by name, taking precedence over the packaged ones.
    prompt_overrides: dict[str, str] = Field(default_factory=dict)
    # Empty for a new interview.
    history: list[HistoryRecord] = Field(default_factory=list)
    referable_values: dict[str, Any] = Field(default_factory=dict)
    previous_time_spent: int = 0


class UserMessage(_Event):
    """The respondent's answer to an `InputRequested` expecting free text or a
    survey item."""

    type: Literal["user_message"] = "user_message"
    # The `InputRequested.message_id` this answers.
    message_id: int
    content: str = ""
    message_type: MessageType = MessageType.TEXT
    # The respondent chose to skip the question; `content` is ignored.
    skip: bool = False


class SecurityOverrideAnswer(_Event):
    """The respondent's answer to an `InputRequested` expecting a security
    override."""

    type: Literal["security_override_answer"] = "security_override_answer"
    message_id: int
    answer: SecurityOverride


type InputEvent = Annotated[
    UserMessage | SecurityOverrideAnswer, Field(discriminator="type")
]
type InboundEvent = Annotated[
    Init | UserMessage | SecurityOverrideAnswer, Field(discriminator="type")
]


# ── Outbound ─────────────────────────────────────────────────────────────────


class MessageEmitted(_Event):
    """A message from the interviewer.

    One with `skipped_by_condition` set was never shown: it records a question
    a condition skipped, and is for storage only.
    """

    type: Literal["message_emitted"] = "message_emitted"
    message_id: int
    content: str
    message_type: MessageType = MessageType.TEXT
    can_answer: bool = True
    include_in_history: bool = True
    survey_item: SurveyItem | None = None
    image: Image | list[Image] | None = None
    section: int | None = None
    main_question: int | None = None
    sub_question: int | None = None
    is_introduction: bool = False
    outro: bool = False
    timed: bool = False
    skipped_by_condition: bool = False
    security_intervention: SecurityIntervention | None = None
    progress: float | None = Field(default=None, ge=0, le=100)
    is_probe: bool = False
    user_image: bool = False
    # The guide question this asks, as its main question or one of its probes;
    # None for anything else (introduction, outro, timed messages).
    question_id: UUID | None = None


class InputKind(StrEnum):
    FREE_TEXT = "free_text"
    SURVEY_ITEM = "survey_item"
    SECURITY_OVERRIDE = "security_override"


class InputRequested(_Event):
    """The engine is waiting for the respondent.

    Answered with a `SecurityOverrideAnswer` when `expects` is
    `SECURITY_OVERRIDE`, and with a `UserMessage` otherwise.
    """

    type: Literal["input_requested"] = "input_requested"
    # The id the answer will be recorded under.
    message_id: int
    expects: InputKind


class UserMessageRecorded(_Event):
    """An answer, as processed by the engine and placed in the interview."""

    type: Literal["user_message_recorded"] = "user_message_recorded"
    message_id: int
    content: str
    message_type: MessageType = MessageType.TEXT
    section: int | None = None
    main_question: int | None = None
    sub_question: int | None = None
    include_in_history: bool = True


class TaskRecorded(_Event):
    """A decision an agent made along the way, for the record.

    May arrive out of step with the messages around it: some tasks, such as
    `assess_security`, run in the background.
    """

    type: Literal["task_recorded"] = "task_recorded"
    # The message the task belongs to.
    message_id: int
    task: str
    reason: str | None = None
    context: str | None = None
    content: str | None = None
    response: str | None = None
    model: str | None = None
    time_spend: int | None = None


class ChunkEmitted(_Event):
    """Interview text ready to be embedded. A `MESSAGE` chunk always follows
    the event that recorded its message."""

    type: Literal["chunk_emitted"] = "chunk_emitted"
    chunk: EmbeddingChunk


class GuideUpdated(_Event):
    """The engine extended the interview's guide, e.g. with a generated
    question. Replaces the guide the host passes in `Init` from now on."""

    type: Literal["guide_updated"] = "guide_updated"
    guide: InterviewGuide


class Progress(_Event):
    type: Literal["progress"] = "progress"
    progress: float = Field(ge=0, le=100)


class InterviewEnded(_Event):
    """The interview is complete; nothing follows.

    The only status the engine reports. Whether a session is active, when it
    stopped and how long it took are the host's to record: it owns the
    connection, and a session can end -- a crash, a dropped socket -- without
    the engine being able to say so.
    """

    type: Literal["interview_ended"] = "interview_ended"


class ErrorCode(StrEnum):
    # A language model could not be reached or gave an unusable reply.
    INFERENCE_ERROR = "inference_error"
    INTERNAL = "internal"


class Error(_Event):
    """The session failed and ends here. The interview can be resumed from
    what was recorded."""

    type: Literal["error"] = "error"
    code: ErrorCode
    detail: str | None = None


type OutboundEvent = Annotated[
    MessageEmitted
    | InputRequested
    | UserMessageRecorded
    | TaskRecorded
    | ChunkEmitted
    | GuideUpdated
    | Progress
    | InterviewEnded
    | Error,
    Field(discriminator="type"),
]

inbound_event_adapter: TypeAdapter[InboundEvent] = TypeAdapter(InboundEvent)
input_event_adapter: TypeAdapter[InputEvent] = TypeAdapter(InputEvent)
outbound_event_adapter: TypeAdapter[OutboundEvent] = TypeAdapter(OutboundEvent)
