from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ainterviewer.lpm.types import Temperature
from ainterviewer.settings import settings


class AgentConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    probing: ProbingAgentConfig = Field(default_factory=lambda: ProbingAgentConfig())
    classification: AgentConfig = Field(
        default_factory=lambda: AgentConfig(temperature=0.0)
    )
    guide: AgentConfig = Field(default_factory=lambda: AgentConfig())
    history: AgentConfig = Field(default_factory=lambda: AgentConfig())
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())
    visual: VisualConfig = Field(default_factory=lambda: VisualConfig())
    answering: AgentConfig = Field(default_factory=lambda: AgentConfig())
    reformulation: AgentConfig = Field(default_factory=lambda: AgentConfig())


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    model: str = settings.llm.default_model
    temperature: Temperature = Field(default=0.7)
    include: bool = True

    @property
    def chat_kwargs(self) -> dict:
        return {"temperature": self.temperature}


class ProbingAgentConfig(AgentConfig):
    few_shot_examples: list[str] | None = None


class SecurityConfig(AgentConfig):
    sensitive_subjects: list | None = None
    include: bool = False

    @model_validator(mode="after")
    def check_sensitive_subjects(self):
        if self.include and not self.sensitive_subjects:
            raise ValueError(
                "'sensitive_subjects' must be provided when include is True"
            )
        return self


class VisualConfig(AgentConfig):
    model: str = "llava"
    include: bool = False
