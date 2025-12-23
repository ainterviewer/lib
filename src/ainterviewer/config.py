from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ainterviewer.types import LanguageCode
from ainterviewer.agents.config import AgentConfigs

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


@cache
def get_configs(name: str) -> tuple[InterviewConfig, AgentConfigs]:
    return read_configs(CONFIG_FOLDER / f"{name}_config.yaml")


def create_template_config():
    config_folder = Path(__file__).parent / "configs"

    for config in read_configs(config_folder / "default_config.yaml"):
        dump = config.model_dump(mode="json", serialize_as_any=True)

        with open(config_folder / f"{config.__name__}_template.yaml", "w") as f:
            yaml.dump(dump, f)


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
