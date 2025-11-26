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
        env_nested_delimiter="__",
        validate_assignment=True,
        frozen=True,
        env_file_encoding="utf-8",
        **kwargs,
    )


class LLMSettings(BaseModel):
    llm_host: str = "0.0.0.0"
    llm_port: int = 8880
    model_storage: Literal["local", "s3_bucket"] = "local"
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
    vllm_api_key: SecretStr = SecretStr("")

    model_config = BaseSettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    debug: bool = False

    llm: LLMSettings = LLMSettings()
    secrets: Secrets = Secrets()  # ty: ignore[missing-argument]

    model_config = BaseSettingsConfigDict(
        toml_file="config.toml",
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
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            PyprojectTomlConfigSettingsSource(settings_cls),
        )


settings = Settings()  # ty: ignore[missing-argument]

if __name__ == "__main__":
    print(settings.model_dump_json(indent=4))
    print(settings.llm.llm_endpoint)
    print(settings.debug)
    print(settings.secrets.openai_api_key.get_secret_value())
