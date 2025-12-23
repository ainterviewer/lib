from __future__ import annotations

from pathlib import Path
from typing import Generator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ainterviewer.lpm.types import Temperature
from ainterviewer.settings import settings
from ainterviewer.types import LanguageCode

CONFIG_FOLDER = Path(__file__).parent.parent.parent / "data" / "configs"


class AgentConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    probing: ProbingAgentConfig = Field(default_factory=lambda: ProbingAgentConfig())
    classification: AgentConfig = Field(
        default_factory=lambda: AgentConfig(temperature=0.0)
    )
    history: AgentConfig = Field(default_factory=lambda: AgentConfig())
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())
    visual: VisualConfig = Field(default_factory=lambda: VisualConfig())
    answering: AgentConfig = Field(default_factory=lambda: AgentConfig())

    def __iter__(self) -> Generator[tuple[str, AgentConfig], None, None]:
        yield from self.__dict__.items()


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    model: str = settings.llm.default_model
    temperature: Temperature = Field(default=0.7)
    lang: LanguageCode = "EN"
    include: bool = True


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


def read_agent_configs(path: Path | str) -> AgentConfigs:
    with open(path) as f:
        agents_config = yaml.safe_load(f)

    if agents_config.get("base"):
        base_config = agents_config.pop("base")

        # Apply base config to all agents
        for agent_name in AgentConfigs.model_fields:
            if agent_name not in agents_config:
                agents_config[agent_name] = base_config.copy()

                # Exclude translation and security by default
                if agent_name in ("translation", "security"):
                    agents_config[agent_name]["include"] = False
            else:
                agents_config[agent_name] = {**base_config, **agents_config[agent_name]}

    return AgentConfigs(**agents_config)
