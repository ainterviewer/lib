import html
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import UUID4, BaseModel, Field, field_validator, model_validator

from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.media import Audio, Image, Video
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.lpm.types import CustomTokens
from ainterviewer.types import Feedback, InterviewStatus, MessageRole, MessageType


class ReceivedData(BaseModel):
    type: Literal["message", "image", "audio"]
    message_type: MessageType | None = None
    content: str
    filename: str | None = Field(None, description="filename for media asset")

    @field_validator("content", mode="before")
    @classmethod
    def escape_html(cls, v: str) -> str:
        if v and v not in CustomTokens.all:
            return html.escape(v)
        else:
            return v

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


class OutgoingData(BaseModel):
    type: Literal["data"] = "data"
    content: str | None = None
    progress: float | None = Field(default=None, ge=0, le=100)
    error: Literal["InstanceInitializing"] | None = None

    def model_dump(self, **kwargs):
        # NOTE:
        # Sets 'json' mode as default if not explicitly overridden
        if "mode" not in kwargs:
            kwargs["mode"] = "json"

        return super().model_dump(**kwargs)


# NOTE:
# Currently IOProtocol and PersistenceProtocol are separated mainly due to need
# for insert_task which is done based on logic other than receiving/sending
# data


class IOProtocol(Protocol):
    async def send_data(self, data: OutgoingData | OutgoingMessage) -> None: ...

    async def receive_message(
        self,
        message_id: int,
        message_type: MessageType | None = None,
    ) -> tuple[str, MessageType]: ...


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
