from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ainterviewer.agents.types import ProbingStrategy
from ainterviewer.types import LanguageCode


class InterviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    default_language: LanguageCode = "EN"

    with_consent: bool = Field(
        True,
        description="Whether to ask for consent before starting the interview",
    )
    with_welcome: bool = Field(
        True,
        description="Whether to show a welcome message before starting the interview",
    )
    with_audio: bool = Field(
        True,
        description="Allows the respondents to record their answers as audio messages which are transcribed before send as answers to the AInterviewer.",
    )

    probing_strategy: set[ProbingStrategy] = Field(
        default_factory=lambda: {ProbingStrategy.STANDARD}
    )
