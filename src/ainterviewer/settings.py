# TODO: Separate settings between app and lib
from __future__ import annotations

import datetime
from typing import Literal, final
from zoneinfo import ZoneInfo

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    SecretStr,
    ValidationInfo,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)
from types_aiobotocore_ec2.literals import InstanceTypeType

from ainterviewer.types import DatabaseType, EC2Access, TimeDelta


class Settings(BaseSettings):
    debugging: bool = False

    app: AppSettings
    database: DatabaseSettings
    llm: LLMSettings
    aws: AWSSettings
    secrets: Secrets
    services: ServiceSettings

    @model_validator(mode="after")
    def validate_aws(self, info: ValidationInfo):
        if (
            self.aws.ec2_inference
            and self.secrets.aws_access_key_id is None
            or self.secrets.aws_secret_access_key is None
        ):
            raise ValueError(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for EC2 inference"
            )

        if self.app.ainterviewer_host is None:
            self.app.ainterviewer_host = f"127.0.0.1:{self.app.ainterviewer_port}"

        return self

    @final
    class Config:
        extra = "ignore"
        env_file = ".env"
        toml_file = "config.toml"
        use_enum_values = True
        validate_assignment = True
        env_nested_delimiter = "__"
        frozen = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls), env_settings, dotenv_settings)


class AppSettings(BaseModel):
    fastapi_env: Literal["production", "staging", "development"] = "development"
    ainterviewer_port: int = 8666
    ainterviewer_host: str | None = None
    jwt_secret_key: str
    jwt_interview_token_expiration: dict[str, float] = Field(
        default_factory=lambda: TimeDelta(days=3).model_dump()
    )
    jwt_auth_token_expiration: dict[str, float] = Field(
        default_factory=lambda: TimeDelta(days=1).model_dump()
    )
    session_secret_key: str
    registration_requires_token: bool = True


class DatabaseSettings(BaseModel):
    db: DatabaseType = DatabaseType.SQLITE

    db_username: str | None = None
    db_password: SecretStr | None = None
    db_url: str = "localhost"
    db_port: str = "5432"
    db_name: str = "ainterviewer"
    default_team_id: UUID4

    @computed_field
    @property
    def database_file(self) -> str | None:
        return "db.sqlite" if self.db == DatabaseType.SQLITE else None

    @computed_field
    @property
    def connection_string(self) -> str:
        if self.db == DatabaseType.SQLITE:
            connection_string = f"sqlite:///app/{self.database_file}"
        else:
            if not self.db_username or not self.db_password:
                raise ValueError(
                    "`db_username` and `db_password` must be set for PostgreSQL"
                )
            connection_string = f"postgresql://{self.db_username}:{self.db_password.get_secret_value()}@{self.db_url}:{self.db_port}/{self.db_name}"

        return connection_string


class LLMSettings(BaseModel):
    llm_host: str = "0.0.0.0"
    llm_port: int = 8880
    model_storage: Literal["local", "s3_bucket"] = "local"
    vllm_api_key: str = ""
    available_models: list[str] = Field(default_factory=lambda: ["gpt-5-mini"])
    default_model: str = "gpt-5-mini"
    seed: int = 4268

    @field_validator("llm_host")
    def set_llm_host(cls, v: str, values: ValidationInfo):
        if not v:
            return f"0.0.0.0:{values.data['llm_port']}"
        return v


class AWSSettings(BaseModel):
    aws_default_region: str = "eu-north-1"
    aws_bedrock_region: str = "eu-west-3"
    aws_s3_bucket: str = "ainterviewer"

    ec2_access: EC2Access = EC2Access.PRIVATE
    ec2_instance_type: InstanceTypeType
    ec2_inference: bool = False
    ec2_ami: str
    ec2_template: str
    ec2_template_version: str | None = None
    ec2_unload_time: int = 3600
    ec2_default_instance_id: str | None = None
    ec2_min_instances_running: int = 0
    ec2_max_concurrent_connections: int | None = Field(
        3,
        description="The maximum number of queued conenctions to a GPU instance, before a new one is started",
    )
    ec2_max_instances: int | None = Field(
        None, description="The maximum number of GPU instances to run"
    )
    ec2_check_stop_interval: int = Field(
        60, description="The interval to check if any GPU instances needs to be stopped"
    )
    ec2_check_start_interval: int = Field(
        30, description="The interval to check if any GPU instances needs to be started"
    )
    ec2_downtime: tuple[datetime.time, datetime.time] | None = Field(
        None,
        description="The time range during which the GPU instances can be stopped regardless of minimum number of instances",
    )

    @property
    def is_downtime(self) -> bool:
        """
        Check if the current time is within the EC2 down time period.
        """
        if not self.ec2_downtime:
            return False

        start_time, end_time = self.ec2_downtime
        check_time = datetime.datetime.now(ZoneInfo("Europe/Copenhagen")).time()

        if start_time <= end_time:
            return start_time <= check_time <= end_time
        else:
            return check_time >= start_time or check_time <= end_time


class Secrets(BaseModel):
    openai_api_key: str | None = None
    google_ai_api_key: str | None = None
    deepl_api_key: str | None = None
    deepl_api_url: str | None = None
    hugging_face_hub_token: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    openrouter_api_key: str | None = None


class ServiceSettings(BaseModel):
    email: EmailSettings


class EmailSettings(BaseModel):
    smtp_server: str
    smtp_port: int = 587
    smtp_use_ssl: bool = False
    sender: EmailAccount
    recipient: EmailAccount


class EmailAccount(BaseModel):
    email: str
    password: SecretStr


settings = Settings()  # pyright: ignore[reportCallIssue]

if __name__ == "__main__":
    print(settings.aws.ec2_downtime)
    print(settings.aws.is_downtime)
    print(settings.model_dump_json(indent=4))
