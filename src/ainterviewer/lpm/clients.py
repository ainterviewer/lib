from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from types import CoroutineType
from typing import Any, overload

import requests
from any_llm import acompletion
from any_llm.types.completion import ChatCompletion
from pydantic import BaseModel

from ainterviewer.lpm.types import CustomToken, Message, Temperature
from ainterviewer.lpm.utils import map_system_to_user
from ainterviewer.settings import settings
from ainterviewer.types import MessageRole
from ainterviewer.utils import encode_image

_DUMMY_MESSAGES: list[Message] = [{"role": MessageRole.USER, "content": "Hello"}]


@overload
async def chat[T: BaseModel](
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


async def chat[T: BaseModel](
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
    if stop_tokens and isinstance(stop_tokens, str):
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
    elif model.startswith("alex:"):
        chat_completion = await chat(
            messages=messages,
            model=model.replace("alex:", "openai:"),
            api_key=settings.secrets.alex_api_key.get_secret_value(),
            api_base="https://inference.alexandra.dk/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        server_endpoint = f"{settings.llm.llm_endpoint}/v1"

        # TODO: re-enable once logit bias is validated against gpt-oss-120b in staging
        extra_model_kwargs = {}  # get_extra_model_kwargs(model, response_format is not None)

        chat_completion = await chat(
            model="vllm:" + model,
            api_base=server_endpoint,
            api_key=settings.secrets.vllm_api_key.get_secret_value(),
            **extra_model_kwargs,
        )

    if response_format:
        return chat_completion.choices[0].message.parsed  # ty:ignore[unresolved-attribute]

    # TODO:
    # - Use the returned log probs
    # classification_tokens = get_classification_response_tokens(model)

    if chat_completion.choices[0].message.content is None:
        # FIXME: This should probably raise an exception
        return ""

    message = chat_completion.choices[0].message.content.strip()

    message = message.encode().decode()

    if message in CustomToken:
        return message

    if sanitize:
        message = re.sub("^Q:|^Question:", "", message)
        message = re.sub("^A:|^Answer:", "", message)
        message = message.strip().strip('"').strip()

    return message.strip() + (
        stop_tokens[0] if stop_tokens and include_stop_token else ""
    )


# FIXME: This needs a new rewrite
def visual_chat(
    model: str,
    messages: list[Message],
    stream=False,
    session: requests.Session | None = None,
):
    session = session or requests.Session()

    encoded_messages = [
        {
            k: v if k != "images" else [encode_image(image) for image in v]  # ty:ignore[not-iterable]
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
    model = "openai/gpt-oss-120b"

    asyncio.run(main(model=model))
