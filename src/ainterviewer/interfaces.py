from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import UUID4, BaseModel, Field, model_validator

from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.media import Audio, Image, Video
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.types import (
    EmbeddingKind,
    Feedback,
    InterviewStatus,
    LanguageCode,
    MessageRole,
    MessageType,
)


class ReceivedData(BaseModel):
    type: Literal["message", "image", "audio"]
    message_type: MessageType | None = None
    content: str
    filename: str | None = Field(None, description="filename for media asset")

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.filename is not None and self.type == "message":
            raise ValueError(
                "Cannot specify filename for messages with `type == 'message'`"
            )
        return self


class _OutgoingData(BaseModel):
    type: Literal["history", "message"]
    content: str
    role: MessageRole
    message_id: int
    feedback: Feedback | None = None
    image: Image | list[Image] | None = None
    survey_item: SurveyItem | None = None

    def model_dump(self, **kwargs):
        # NOTE:
        # Set 'json' mode as default if not explicitly overridden
        if "mode" not in kwargs:
            kwargs["mode"] = "json"
        return super().model_dump(**kwargs)


class OutgoingHistoryMessage(_OutgoingData):
    type: Literal["history"] = "history"


class OutgoingMessage(_OutgoingData):
    type: Literal["message"] = "message"
    can_answer: bool = True
    user_image: bool = False
    progress: float | None = Field(default=None, ge=0, le=100)
    is_probe: bool = False


class OutgoingData(BaseModel):
    type: Literal["data"] = "data"
    content: str | None = None
    progress: float | None = Field(default=None, ge=0, le=100)
    error: Literal["InstanceInitializing", "InferenceError"] | None = None

    def model_dump(self, **kwargs):
        # NOTE:
        # Sets 'json' mode as default if not explicitly overridden
        if "mode" not in kwargs:
            kwargs["mode"] = "json"

        return super().model_dump(**kwargs)


class IOProtocol(Protocol):
    async def send_data(self, data: OutgoingData | OutgoingMessage) -> None: ...

    async def receive_message(
        self,
        message_id: int,
        message_type: MessageType | None = None,
    ) -> tuple[str, MessageType, str | None]:
        """Returns the message text, its type, and the filename of the media
        asset backing it (e.g. the audio recording a transcript came from),
        if any."""
        ...


class PersistenceProtocol(Protocol):
    def update_interview_status(
        self,
        project_id: UUID4,
        interview_id: UUID4,
        status: InterviewStatus,
        time_spent: int = 0,
    ): ...

    def update_interview_guide(
        self,
        project_id: UUID4,
        interview_id: UUID4,
        interview_guide: InterviewGuide,
    ): ...

    def insert_message(
        self,
        message_id: int,
        content: str,
        role: MessageRole,
        interview_id: UUID4,
        project_id: UUID4,
        message_type: MessageType = MessageType.TEXT,
        can_answer: bool = True,
        include_in_history: bool = True,
        attachment: Path | None = None,
        audio_file: str | None = None,
        survey_item: SurveyItem | None = None,
        image: Image | list[Image] | None = None,
        section: int | None = None,
        main_question: int | None = None,
        sub_question: int | None = None,
        is_introduction: bool = False,
        outro: bool = False,
        timed: bool = False,
        skipped_by_condition: bool = False,
    ) -> int: ...

    def insert_task(
        self,
        message_id: int,
        interview_id: UUID4,
        project_id: UUID4,
        task: str,
        reason: str | None = None,
        context: str | None = None,
        content: str | None = None,
        response: str | None = None,
        model: str | None = None,
        time_spend: int | None = None,
    ): ...

    async def save_media(self, image: Image | Audio | Video): ...


class EmbeddingChunk(BaseModel):
    """One unit of interview text handed to an embedding backend.

    The chunk identifies itself *structurally* -- by interview plus the
    ``(section, main_question, sub_question)`` coordinates the interview loop
    already tracks -- rather than by a database row id. `InterviewHistory` holds
    no row ids, so a QA pair assembled from it has none to give; and a
    consumer that stores messages can resolve the coordinates back to its own
    rows unambiguously, since a turn has exactly one respondent answer.

    ``message_id`` is the per-interview counter also passed to
    ``PersistenceProtocol.insert_message``, set only for ``MESSAGE`` chunks.
    """

    kind: EmbeddingKind
    text: str
    # The `ChunkPolicy.format_version` that produced `text`. Stored with the
    # vector so a consumer can tell a corpus built under changed inclusion or
    # rendering rules from one that is current, instead of discovering the
    # mismatch through search results that quietly got worse.
    format_version: str = "1"
    project_id: UUID4
    interview_id: UUID4
    language: LanguageCode = "EN"
    message_id: int | None = None
    section: int | None = None
    main_question: int | None = None
    sub_question: int | None = None


class EmbeddingProtocol(Protocol):
    """Sink for embeddable chunks produced during an interview.

    Implementations must be cheap and must not raise: `AInterviewer` calls this
    on the interview's critical path, and an interview must never fail because
    an embedding backend is unreachable. Queue the chunk and return.
    """

    async def embed_chunk(self, chunk: EmbeddingChunk) -> None: ...
