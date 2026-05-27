from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.agents.types import ProbingStrategy
from ainterviewer.types import LanguageCode

CONFIG_FOLDER = Path(__file__).parent.parent.parent / "data" / "configs"


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


@cache
def get_configs(name: str) -> tuple[InterviewConfig, AgentConfigs]:
    return read_configs(CONFIG_FOLDER / f"{name}_config.yaml")


def read_configs(path: Path | str) -> tuple[InterviewConfig, AgentConfigs]:
    with open(path) as f:
        config = yaml.safe_load(f)

    return create_config(config)


def read_interview_config(path: Path | str) -> InterviewConfig:
    with open(path) as f:
        config = yaml.safe_load(f)

    return InterviewConfig(**config)


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
