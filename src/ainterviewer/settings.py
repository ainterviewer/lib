from __future__ import annotations

from pathlib import Path
from typing import Any
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
    # Full OpenAI-compatible base URL of a vLLM server, e.g.
    # "https://llm.example.org/v1". Overrides llm_host/llm_port for models
    # without a provider prefix, and allows HTTPS and path prefixes.
    vllm_base_url: str | None = None

    # Two lists, so whoever adds a model has to say which it is; a model in
    # neither is treated as test only.
    #: Models respondents' answers may be sent to: usable everywhere,
    #: distributed interviews included.
    models: list[str] = Field(default_factory=list)
    #: Models for drafts and test interviews only (manual and synthetic): a
    #: project whose agents use one cannot be published or distributed.
    test_only_models: list[str] = Field(
        default_factory=lambda: ["openrouter:openai/gpt-oss-120b"]
    )
    default_model: str = "openrouter:openai/gpt-oss-120b"

    demo_models: list[str] | None = Field(None, validate_default=True)
    default_demo_model: str | None = Field(None, validate_default=True)

    seed: int = 4268

    @computed_field
    @property
    def llm_endpoint(self) -> str:
        return f"http://{self.llm_host}:{self.llm_port}"

    @property
    def vllm_api_base(self) -> str:
        """Base URL that models without a provider prefix are sent to."""
        if self.vllm_base_url:
            return self.vllm_base_url.rstrip("/")
        return f"{self.llm_endpoint}/v1"

    @property
    def available_models(self) -> list[str]:
        """Every model a project may configure, the distributable ones first."""
        return [*self.models, *self.test_only_models]

    def is_test_only(self, model: str) -> bool:
        """Whether `model` is barred from distributed interviews: anything
        not in `models`, so a model nobody listed is never distributed."""
        return model not in self.models

    @model_validator(mode="before")
    @classmethod
    def reject_available_models(cls, data: Any):
        # Ignoring it (extra="ignore") would silently fall back to the
        # defaults, under which nothing can be published.
        if isinstance(data, dict) and "available_models" in data:
            raise ValueError(
                "available_models was split into models (usable in distributed "
                "interviews) and test_only_models"
            )
        return data

    @field_validator("models", "test_only_models")
    @classmethod
    def unique_models(cls, v: list[str]):
        if len(v) != len(set(v)):
            raise ValueError("model lists cannot contain duplicate entries")

        return v

    @model_validator(mode="after")
    def finalize(self):
        if both := set(self.models) & set(self.test_only_models):
            raise ValueError(
                f"models and test_only_models overlap: {', '.join(sorted(both))}"
            )

        if not self.available_models:
            raise ValueError("models and test_only_models cannot both be empty")

        if self.default_model not in self.available_models:
            raise ValueError("default_model must be one of models or test_only_models")

        if self.demo_models is None:
            self.demo_models = list(self.available_models)

        if self.default_demo_model is None:
            self.default_demo_model = self.default_model

        if self.default_demo_model not in self.demo_models:
            raise ValueError("default_demo_model must be one of demo_models")

        return self


class Secrets(BaseSettings):
    openai_api_key: SecretStr = SecretStr("")
    google_ai_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    vllm_api_key: SecretStr = SecretStr("")
    alex_api_key: SecretStr = SecretStr("")

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
