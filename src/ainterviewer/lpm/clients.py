from __future__ import annotations

import re
from functools import partial
from types import CoroutineType
from typing import Any, Callable, Optional

import litellm
import requests
from litellm import acompletion
from litellm.types.utils import ModelResponse

from ainterviewer.lpm.types import CustomTokens, Message, Temperature
from ainterviewer.lpm.utils import map_system_to_user
from ainterviewer.lpm.vllm import VLLM_MODEL_CONFIGS
from ainterviewer.settings import settings
from ainterviewer.types import MessageRole
from ainterviewer.utils import encode_image

litellm.suppress_debug_info = True

_DUMMY_MESSAGES: list[Message] = [{"role": MessageRole.USER, "content": "Hello"}]


async def chat(
    messages: list[Message],
    model: str,
    temperature: Temperature = 0.7,
    stop_tokens: Optional[list[str] | str] = None,
    include_stop_token: bool = False,
    sanitize: bool = True,
    guided_choice: Optional[list[str]] = None,
    top_logprogs: Optional[int] = None,
    **model_kwargs,
) -> str:
    if stop_tokens:
        if isinstance(stop_tokens, str):
            stop_tokens = [stop_tokens]

    chat: Callable[..., CoroutineType[Any, Any, ModelResponse]] = partial(
        acompletion,
        messages=messages,
        temperature=temperature,
        stop=stop_tokens,
        top_logprobs=top_logprogs,
        seed=settings.llm.seed,
        drop_params=True,
        stream=False,
        **model_kwargs,
    )  # type: ignore

    if model.startswith("openrouter/"):
        chat_completion: ModelResponse = await chat(
            model=model,
            provider={"order": ["deepinfra"]},
            api_key=settings.secrets.openrouter_api_key.get_secret_value(),
            reasoning_effort="minimal",
        )
    elif model.startswith("openai/"):
        # TODO:
        # Implement logit_bias instead of guided_choice for openai...
        # see ainterviewer.lpm.utils.get_classification_response_tokens
        chat_completion: ModelResponse = await chat(
            model=model,
            api_key=settings.secrets.openai_api_key,
            reasoning_effort="minimal",
        )
    elif model.startswith("gemini/"):
        # map system roles for gemini compatability
        messages = [
            {"role": map_system_to_user(message["role"]), "content": message["content"]}
            for message in messages
        ]
        chat_completion: ModelResponse = await chat(
            messages=messages,
            model=model,
            api_key=settings.secrets.google_ai_api_key.get_secret_value(),
        )
    else:
        server_endpoint = f"{settings.llm.llm_endpoint}/v1"

        model_kwargs = {}

        model_kwargs["model"] = "hosted_vllm/" + (
            served_model_name
            if (served_model_name := VLLM_MODEL_CONFIGS[model].served_model_name)
            else model
        )

        if guided_choice:
            model_kwargs["extra_body"] = dict(guided_choice=guided_choice)

        if model in ("gpt-oss-120b"):
            model_kwargs["reasoning_effort"] = "low"

        chat_completion: ModelResponse = await chat(
            api_base=server_endpoint,
            api_key=settings.secrets.vllm_api_key,
            **model_kwargs,
        )

    # TODO:
    # - Use the returned log probs
    # classification_tokens = get_classification_response_tokens(model)

    message = chat_completion.choices[0].message.content.strip()  # type: ignore

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

    response = session.post(f"http://{settings.llm.llm_host}/api/chat", json=data)
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

    model = "gemma3-27b"

    asyncio.run(main(model=model))
