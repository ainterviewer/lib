from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Generator

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ainterviewer.exceptions import ConfigError
from ainterviewer.lpm.types import Temperature
from ainterviewer.settings import settings
from ainterviewer.types import LanguageCode

CONFIG_FOLDER = Path(__file__).parent.parent.parent / "data" / "configs"


class InterviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    default_language: LanguageCode = "EN"
    referer_id_key: str | None = Field(
        None, description="Key for the value of the id sent in the referer header"
    )
    with_consent: bool = Field(
        True, description="Whether to ask for consent before starting the interview"
    )
    with_welcome: bool = Field(
        True,
        description="Whether to show a welcome message before starting the interview",
    )


class AgentConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    probing: ProbingAgentConfig = Field(default_factory=lambda: ProbingAgentConfig())
    classification: AgentConfig = Field(default_factory=lambda: AgentConfig())
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


@cache
def get_configs(name: str) -> tuple[InterviewConfig, AgentConfigs]:
    return read_configs(CONFIG_FOLDER / f"{name}_config.yaml")


def create_template_config():
    config_folder = Path(__file__).parent / "configs"

    dump = read_configs(config_folder / "default_config.yaml").model_dump(
        mode="json", serialize_as_any=True
    )

    with open(config_folder / "config_template.yaml", "w") as f:
        yaml.dump(dump, f)


def read_configs(path: Path | str) -> tuple[InterviewConfig, AgentConfigs]:
    with open(path) as f:
        config = yaml.safe_load(f)

    return create_config(config)


def read_interview_config(path: Path | str) -> InterviewConfig:
    with open(path) as f:
        config = yaml.safe_load(f)

    return InterviewConfig(**config)


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


def create_config(config: dict) -> tuple[InterviewConfig, AgentConfigs]:
    agents_config = config.pop("agents")

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

    return InterviewConfig(**config), AgentConfigs(**agents_config)
