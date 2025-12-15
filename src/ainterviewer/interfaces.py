import html
from pathlib import Path
from typing import Literal, Optional, Protocol

from pydantic import UUID4, BaseModel, Field, FilePath, field_validator

from ainterviewer.interview_guides import Image
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.types import Feedback, MessageRole, MessageType


class ReceivedData(BaseModel):
    type: Literal["message", "image"]
    message_type: MessageType | None = None
    content: str | None = None
    file: FilePath | None = None

    @field_validator("content", mode="before")
    @classmethod
    def escape_html(cls, v: str | None) -> str | None:
        return html.escape(v) if v else v


class _OutgoingData(BaseModel):
    content: str
    role: MessageRole
    interview_id: UUID4
    message_id: int
    include_in_history: bool = True
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
    interview_id: UUID4 | None = None
    project_id: UUID4 | None = None
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
        self, message_type: MessageType | None = None
    ) -> tuple[str, MessageType]: ...


class PersistenceProtocol(Protocol):
    def update_interview_status(
        self,
        project_id: UUID4,
        interview_id: UUID4,
        is_active: bool | None = None,
        is_complete: bool | None = None,
        time_spent: int = 0,
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
        image: Optional[Image | list[Image]] = None,
        section: Optional[int] = None,
        main_question: Optional[int] = None,
        sub_question: Optional[int] = None,
        is_introduction: bool = False,
        outro: bool = False,
        timed: bool = False,
    ) -> int: ...

    def insert_task(
        self,
        message_id: int,
        interview_id: UUID4,
        project_id: UUID4,
        task: str,
        reason: Optional[str] = None,
        context: Optional[str] = None,
        content: Optional[str] = None,
        response: Optional[str] = None,
        model: Optional[str] = None,
        time_spend: Optional[int] = None,
    ): ...

    def save_image(self, image: Image): ...
