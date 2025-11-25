from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, computed_field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def BaseSettingsConfigDict(**kwargs) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        use_enum_values=True,
        validate_assignment=True,
        frozen=True,
        **kwargs,
    )


class LLMSettings(BaseModel):
    llm_host: str = "0.0.0.0"
    llm_port: int = 8880
    model_storage: Literal["local", "s3_bucket"] = "local"
    vllm_api_key: str = ""
    available_models: list[str] = Field(default_factory=lambda: ["gpt-5-mini"])
    default_model: str = "gpt-5-mini"
    seed: int = 4268

    @computed_field
    @property
    def llm_endpoint(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}"


class Secrets(BaseSettings):
    openai_api_key: SecretStr | None = None
    google_ai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None

    model_config = BaseSettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    debug: bool = False

    llm: LLMSettings = LLMSettings()
    secrets: Secrets = Secrets()  # ty: ignore[missing-argument]

    model_config = BaseSettingsConfigDict(
        toml_file="config.toml",
        env_nested_delimiter="__",
        pyproject_toml_table_header=("tool", "ainterviewer"),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )


settings = Settings()  # ty: ignore[missing-argument]

if __name__ == "__main__":
    print(settings.model_dump_json(indent=4))
    print(settings.llm.llm_endpoint)
    print(settings.debug)
