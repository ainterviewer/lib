from __future__ import annotations

import re
from functools import partial
from types import CoroutineType
from typing import Any, Callable, TypeVar, overload

import requests
from any_llm import acompletion
from any_llm.types.completion import ChatCompletion
from pydantic import BaseModel

from ainterviewer.lpm.types import CustomTokens, Message, Temperature
from ainterviewer.lpm.utils import map_system_to_user
from ainterviewer.lpm.vllm import VLLM_MODEL_CONFIGS
from ainterviewer.settings import settings
from ainterviewer.types import MessageRole
from ainterviewer.utils import encode_image

_DUMMY_MESSAGES: list[Message] = [{"role": MessageRole.USER, "content": "Hello"}]

T = TypeVar("T", bound=BaseModel)


@overload
async def chat(
    messages: list[Message],
    model: str,
    response_format: type[T],
    temperature: Temperature = 0.7,
    stop_tokens: list[str] | str | None = None,
    include_stop_token: bool = False,
    sanitize: bool = True,
    top_logprobs: int | None = None,
    **model_kwargs,
) -> T: ...


@overload
async def chat(
    messages: list[Message],
    model: str,
    temperature: Temperature = 0.7,
    stop_tokens: list[str] | str | None = None,
    include_stop_token: bool = False,
    sanitize: bool = True,
    top_logprobs: int | None = None,
    response_format: None = None,
    **model_kwargs,
) -> str: ...


async def chat(
    messages: list[Message],
    model: str,
    temperature: Temperature = 0.7,
    stop_tokens: list[str] | str | None = None,
    include_stop_token: bool = False,
    sanitize: bool = True,
    top_logprobs: int | None = None,
    response_format: type[T] | None = None,
    **model_kwargs,
) -> str | T:
    if stop_tokens:
        if isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens]

    chat: Callable[..., CoroutineType[Any, Any, ChatCompletion]] = partial(  # ty: ignore[invalid-assignment]
        acompletion,
        messages=messages,
        temperature=temperature,
        stop=stop_tokens,
        top_logprobs=top_logprobs,
        seed=settings.llm.seed,
        stream=False,
        response_format=response_format,
        **model_kwargs,
    )

    if model.startswith("openrouter:"):
        chat_completion = await chat(
            model=model,
            extra_body={"provider": {"order": ["deepinfra"]}},
            api_key=settings.secrets.openrouter_api_key.get_secret_value(),
            reasoning_effort="low",
        )
    elif model.startswith("openai:"):
        chat_completion = await chat(
            model=model,
            api_key=settings.secrets.openai_api_key.get_secret_value(),
            reasoning_effort="none",
        )
    elif model.startswith("gemini:"):
        # map system roles for gemini compatability
        messages = [
            {"role": map_system_to_user(message["role"]), "content": message["content"]}
            for message in messages
        ]
        chat_completion = await chat(
            messages=messages,
            model=model,
            api_key=settings.secrets.google_ai_api_key.get_secret_value(),
        )
    else:
        server_endpoint = f"{settings.llm.llm_endpoint}/v1"

        extra_model_kwargs = {}

        model = "vllm:" + (
            served_model_name
            if (served_model_name := VLLM_MODEL_CONFIGS[model].served_model_name)
            else model
        )

        if model in ("gpt-oss-120b"):
            extra_model_kwargs["reasoning_effort"] = "low"
            extra_model_kwargs["extra_body"] = {"top_k": 3}

            if response_format is None:
                # TODO:
                # - this should be model based and maybe also question based.
                # - maybe they should be words, and tokens fetched and cached from the api.
                # - how to implement in interface.
                print("Applying logit bias")
                extra_model_kwargs["logit_bias"] = {
                    # Negative
                    "4157": -3,  #  kun
                    "65512": -7,  # Kan
                    "98936": -5,  # Kunne
                    "11": -7,  # ,
                    "80750": -15,  # konkre
                    "102719": -15,  #  konkret
                    "12855": -15,  #  specif
                    # Positive
                    "73760": 5,  # Tak
                    "30": 7,  # ?
                }

        chat_completion = await chat(
            model=model,
            api_base=server_endpoint,
            api_key=settings.secrets.vllm_api_key.get_secret_value(),
            **extra_model_kwargs,
        )

    if response_format:
        return chat_completion.choices[0].message.parsed

    # TODO:
    # - Use the returned log probs
    # classification_tokens = get_classification_response_tokens(model)

    message = chat_completion.choices[0].message.content.strip()

    message = message.encode().decode()

    if message in CustomTokens.all:
        return message

    if sanitize:
        message = re.sub("^Q:|^Question:", "", message)
        message = re.sub("^A:|^Answer:", "", message)
        message = message.strip().strip('"').strip()

    return message.strip() + (
        stop_tokens[0] if stop_tokens and include_stop_token else ""
    )


def visual_chat(
    model: str,
    messages: list[Message],
    stream=False,
    session=requests.Session(),
):
    encoded_messages = [
        {
            k: v if k != "images" else [encode_image(image) for image in v]
            for k, v in message.items()
        }
        for message in messages
    ]

    data = {
        "model": model,
        "messages": encoded_messages,
        "stream": stream,
    }

    response = session.post(f"http://{settings.llm.llm_endpoint}/api/chat", json=data)
    response.raise_for_status()

    return response.json()["message"]["content"]


async def main(
    model: str,
):
    response = await chat(
        messages=_DUMMY_MESSAGES,
        model=model,
        temperature=0.7,
    )

    print(response)


if __name__ == "__main__":
    import asyncio

    # model = "openrouter:openai/gpt-oss-120b"
    # model = "openrouter:openai/gpt-oss-120b"
    # model = "openai:gpt-5.2"
    model = "gpt-oss-120b"

    asyncio.run(main(model=model))
