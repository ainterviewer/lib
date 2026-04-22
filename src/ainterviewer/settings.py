from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ainterviewer.storage import ExperimentStorage, InterviewStorage, ProjectStorage


class MediaStorageSettings(BaseModel):
    """Central configuration for all media storage."""

    base_path: Path = Field(default=Path("storage/"))

    @field_validator("base_path")
    @classmethod
    def validate_base_path(cls, v: Path) -> Path:
        """Ensure base storage path exists."""
        path = v.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def project_storage(self) -> ProjectStorage:
        """Get project storage instance."""
        return ProjectStorage(base_path=self.base_path / "projects")

    @property
    def interview_storage(self) -> InterviewStorage:
        """Get interview storage instance."""
        return InterviewStorage(base_path=self.base_path / "interviews")

    @property
    def experiment_storage(self) -> ExperimentStorage:
        """Get interview storage instance."""
        return ExperimentStorage(base_path=self.base_path / "experiments")


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

    available_models: list[str] = Field(default_factory=lambda: ["gpt-5-mini"])
    default_model: str = "gpt-5-mini"

    demo_models: list[str] | None = Field(None, validate_default=True)
    default_demo_model: str | None = Field(None, validate_default=True)

    seed: int = 4268

    @computed_field
    @property
    def llm_endpoint(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}"

    @field_validator("available_models")
    @classmethod
    def unique_models(cls, v: Any):
        if len(v) != len(set(v)):
            raise ValueError("available_models contains duplicate entries")

        return v

    @model_validator(mode="after")
    def finalize(self):
        if not self.available_models:
            raise ValueError("available_models cannot be empty")

        if self.default_model not in self.available_models:
            raise ValueError("default_model must be one of available_models")

        if self.demo_models is None:
            self.demo_models = list(self.available_models)

        if self.default_demo_model is None:
            self.default_demo_model = self.default_model

        if self.default_demo_model not in self.demo_models:
            raise ValueError("default_demo_model must be one of demo_models")

        return self


class Secrets(BaseSettings):
    openai_api_key: SecretStr | None = None
    google_ai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    vllm_api_key: SecretStr = SecretStr("")

    model_config = BaseSettingsConfigDict(env_prefix="")


class Settings(BaseSettings):
    debug: bool = False

    tzinfo: ZoneInfo = "Europe/Copenhagen"  # ty: ignore[invalid-assignment]

    llm: LLMSettings = LLMSettings()
    secrets: Secrets = Secrets()
    storage: MediaStorageSettings = MediaStorageSettings()

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


settings = Settings()

if __name__ == "__main__":
    print(settings.model_dump_json(indent=4))
    print(settings.llm.llm_endpoint)
    print(settings.debug)
    # print(settings.secrets.openai_api_key.get_secret_value())
