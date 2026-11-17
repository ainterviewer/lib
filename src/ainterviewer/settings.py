from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
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


class ConfigSection(BaseModel):
    """A table of a config file, such as `[llm]`.

    Each table has one owner, so a key in it that nothing reads is a typo or a
    leftover, and is refused rather than silently ignored. Whole tables are
    the unit a file is shared at -- the backend's config.toml also carries the
    library's `[llm]` -- so an unknown *top-level* key is only warned about,
    see `ConfigSettings`.
    """

    model_config = ConfigDict(extra="forbid")


class MediaStorageSettings(ConfigSection):
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
    """The shared settings config; `kwargs` add to or override it."""
    defaults = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        use_enum_values=True,
        env_nested_delimiter="__",
        validate_assignment=True,
        frozen=True,
        env_file_encoding="utf-8",
        # Settings validate the environment and .env, which hold the secrets,
        # and a validation error prints the values it refused: one model
        # handed lines it did not expect dumps them all to the log. The error
        # still names the field and what it expected. Pydantic honours this
        # on the outermost model only, which is why it lives here, on every
        # settings class, rather than on the sections.
        hide_input_in_errors=True,
    )
    return SettingsConfigDict(**{**defaults, **kwargs})


class ConfigError(ValueError):
    """A config file holds something it must not."""


class ConfigWarning(UserWarning):
    """A config file holds a key nothing reads."""


_secret_classes: list[type[SecretSettings]] = []

# Key names that hold a secret wherever they appear, for secrets of classes
# this process never defines: the proxy reading a file that also carries the
# backend's. Deliberately no `_token` suffix: `registration_requires_token`.
_SECRET_NAMES = {"password", "secret", "api_key", "apikey"}
_SECRET_SUFFIXES = ("_password", "_secret", "_secret_key", "_api_key")


class SecretSettings(BaseSettings):
    """Base for secrets, which are read from the environment and `.env` only.

    Never make one a field of a `ConfigSettings`: that would read it from
    config.toml too, which is what lets config.toml be committed. Keep it
    beside the settings, as its own instance.

    Every subclass is registered, so a config file naming one of its fields
    is refused with the variable to set instead.
    """

    model_config = BaseSettingsConfigDict()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _secret_classes.append(cls)

    @classmethod
    def env_names(cls) -> dict[str, str]:
        """Each field's environment variable, by field name."""
        prefix = cls.model_config.get("env_prefix", "")
        return {name: f"{prefix}{name}".upper() for name in cls.model_fields}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, file_secret_settings


def check_config_file(
    data: Mapping[str, Any], where: str, known: Iterable[str] | None = None
) -> None:
    """Refuse secrets in a config file, and warn about unknown top-level keys.

    `known` is every top-level key some settings in this process reads; None
    skips the warning, for settings that cannot know who else shares the file.
    """
    secrets = {
        name: env for cls in _secret_classes for name, env in cls.env_names().items()
    }

    def walk(node: Mapping[str, Any], path: str) -> None:
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            name = key.lower()
            if (
                name in secrets
                or name in _SECRET_NAMES
                or name.endswith(_SECRET_SUFFIXES)
            ):
                where_to = secrets.get(name, "its environment variable")
                raise ConfigError(
                    f"{where}: `{here}` is a secret. Secrets are read from the "
                    f"environment and .env only: set {where_to} there instead."
                )
            if isinstance(value, Mapping):
                walk(value, here)

    if "secrets" in data:
        raise ConfigError(
            f"{where}: there is no [secrets] table. Secrets are read from the "
            "environment and .env only."
        )
    walk(data, "")

    if known is not None and (unknown := sorted(set(data) - set(known))):
        warnings.warn(
            f"{where}: nothing reads {', '.join(unknown)}; remove "
            f"{'it' if len(unknown) == 1 else 'them'}",
            ConfigWarning,
            stacklevel=2,
        )


class ConfigSettings(BaseSettings):
    """Settings read from the environment, `.env`, config.toml and
    `[tool.ainterviewer]` in pyproject.toml, in that order of precedence.

    Neither file may hold a secret (`check_config_file`), so both can be
    committed. Secrets are a `SecretSettings`.
    """

    #: Other settings that read the same file in this process. Their top-level
    #: keys are not unknown, so they are not warned about.
    shares_file_with: ClassVar[tuple[type[BaseSettings], ...]] = ()
    #: Whether to warn about unknown top-level keys at all. Off for settings
    #: that cannot know who else reads the file.
    warn_unknown_keys: ClassVar[bool] = True

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
        known = None
        if cls.warn_unknown_keys:
            known = {
                name
                for c in (settings_cls, *cls.shares_file_with)
                for name in c.model_fields
            }

        toml = TomlConfigSettingsSource(settings_cls)
        pyproject = PyprojectTomlConfigSettingsSource(settings_cls)
        check_config_file(toml.toml_data, "config.toml", known)
        check_config_file(
            pyproject.toml_data, "pyproject.toml [tool.ainterviewer]", known
        )

        return env_settings, dotenv_settings, toml, pyproject


class LLMSettings(ConfigSection):
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


class ApiKeys(SecretSettings):
    """The model providers' keys, under the names their own SDKs read."""

    openai_api_key: SecretStr = SecretStr("")
    google_ai_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    vllm_api_key: SecretStr = SecretStr("")
    alex_api_key: SecretStr = SecretStr("")


class Settings(ConfigSettings):
    # The backend's config.toml carries these beside its own tables, and this
    # class cannot know them; the backend warns for both.
    warn_unknown_keys: ClassVar[bool] = False

    debug: bool = False

    tzinfo: ZoneInfo = "Europe/Copenhagen"  # ty: ignore[invalid-assignment]

    llm: LLMSettings = LLMSettings()
    storage: MediaStorageSettings = MediaStorageSettings()


settings = Settings()
api_keys = ApiKeys()

if __name__ == "__main__":
    print(settings.model_dump_json(indent=4))
    print(settings.llm.llm_endpoint)
    print(settings.debug)
    # print(settings.secrets.openai_api_key.get_secret_value())
