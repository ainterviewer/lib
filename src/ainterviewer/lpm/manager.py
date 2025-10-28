from __future__ import annotations

import asyncio
from threading import Lock
from typing import Any, Literal

import aiohttp
import requests
from pydantic import BaseModel, model_validator

from ainterviewer.settings import settings


class LLMServerHealth(BaseModel):
    version: str | None = None
    models: list[ModelHealth] | None = None
    status: Literal["success", "error"] = "success"
    error_details: ErrorDetails | None = None


class ErrorDetails(BaseModel):
    type: str
    message: str


class ModelHealth(BaseModel):
    name: str
    model: str | None = None
    size: int | None
    details: ModelDetails

    @model_validator(mode="after")
    def validate_model(self):
        if self.model is None:
            self.model = self.name

        return self


class ModelDetails(BaseModel):
    context_size: int | None = None
    quantization_level: str | None = None


class LLMServerManager:
    def __init__(self, async_session: aiohttp.ClientSession, server_address: str):
        self.session = requests.Session()
        self.server_address = server_address
        self.server_health_endpoint = f"http://{server_address}/v1/models"

        self.alock = asyncio.Lock()
        self.lock = Lock()
        self.async_session = async_session

    async def get_server_health(self) -> LLMServerHealth:
        try:
            async with self.async_session.get(
                "http://" + self.server_address + "/version"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                if not isinstance(data, dict):
                    raise ValueError("Invalid response from server")
                # WARNING: vllm may use 0.10.0.1 which is incompatible with
                # pydantic's SemanticVersion validation
                # vllm_version = SemanticVersion.parse(
                #     # NOTE: Strips dev versions since SemanticVersion doesn't support them
                #     re.split(r"\.dev*", data["version"])[0]
                # )
                vllm_version = data["version"]

            async with self.async_session.get(
                self.server_health_endpoint,
                timeout=aiohttp.ClientTimeout(total=5),
                headers={"Authorization": f"Bearer {settings.llm.vllm_api_key}"},
            ) as response:
                response.raise_for_status()
                server_health = self.parse_server_health(await response.json())

            server_health.version = vllm_version
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            server_health = LLMServerHealth(
                version=None,
                status="error",
                error_details=ErrorDetails(
                    type=str(e.__class__.__name__), message=str(e)
                ),
            )

        return server_health

    def parse_server_health(self, data: dict[Any, Any]) -> LLMServerHealth:
        models = []
        server_data = data["data"] if data["data"] else []

        for model in server_data:
            models.append(
                {
                    "name": model["id"],
                    "model": model.get("root"),
                    "size": None,
                    "details": {"context_size": model["max_model_len"]},
                }
            )

        return LLMServerHealth(models=models)
