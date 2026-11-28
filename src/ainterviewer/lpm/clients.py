from __future__ import annotations

import math
import re
from collections.abc import Callable
from functools import partial
from types import CoroutineType
from typing import Any, cast, overload

import requests
from any_llm import acompletion
from any_llm.types.completion import ChatCompletion
from pydantic import BaseModel

from ainterviewer.exceptions import (
    ModelUnavailable,
    StructuredOutputError,
    UnavailableReason,
)
from ainterviewer.lpm.types import CustomToken, Message, Temperature
from ainterviewer.lpm.utils import map_system_to_user
from ainterviewer.settings import api_keys, settings
from ainterviewer.types import MessageRole
from ainterviewer.utils import encode_image

_DUMMY_MESSAGES: list[Message] = [{"role": MessageRole.USER, "content": "Hello"}]


# Provider routing for OpenRouter models that need it, keyed by the model name
# without the `openrouter:` prefix. Any other model is left to OpenRouter's own
# routing: a preferred provider that does not serve the model only adds a
# failed attempt before the fallback.
OPENROUTER_PROVIDERS: dict[str, dict[str, Any]] = {
    "openai/gpt-oss-120b": {"order": ["amazon-bedrock/eu-west-1"]},
}


def openrouter_model_kwargs(model: str) -> dict[str, Any]:
    """The extra request arguments `model` takes on OpenRouter."""
    provider = OPENROUTER_PROVIDERS.get(model.removeprefix("openrouter:"))
    return {"extra_body": {"provider": provider}} if provider else {}


#: The prefixes `chat` sends to a hosted provider, one branch each. A model
#: with none of them goes to the self-hosted vLLM server, behind our EC2
#: proxy: `chat` routes there by `is_self_hosted`, and so does the backend's
#: warm-up. Add a provider's prefix here with its branch; a prefix with no
#: branch raises rather than reaching vLLM.
HOSTED_PREFIXES = ("openrouter:", "openai:", "gemini:", "alex:")


def is_self_hosted(model: str) -> bool:
    """Whether `chat` sends `model` to the self-hosted vLLM server."""
    return not model.startswith(HOSTED_PREFIXES)


#: Seconds to wait on a provider that said it cannot serve now (a 503 or a 429)
#: without saying for how long.
DEFAULT_RETRY_AFTER = 10

# The codes our EC2 proxy puts on what it refuses (`Refusal` in the proxy
# repo). The OpenAI SDK reads `code` off an error body and any-llm carries it
# on, so they arrive as the error's `code`.
_PROXY_REFUSALS = {
    "capacity_starting": UnavailableReason.STARTING,
    "capacity_busy": UnavailableReason.BUSY,
    "capacity_unavailable": UnavailableReason.DOWN,
    "inference_disabled": UnavailableReason.DOWN,
}


def model_unavailable(error: Exception, model: str) -> ModelUnavailable | None:
    """`error` as a `ModelUnavailable`, when it is a provider saying it cannot
    serve now (503) or will not for a while (429); None for anything else.

    Reads the error by attribute -- `status_code`, `code`, the response's
    `Retry-After` -- which any-llm's unified errors and the OpenAI SDK's own
    both carry, so it holds whether or not unified exceptions are on.

    A refusal the proxy says is final is `DOWN`, with no `retry_after`. Any
    other 503 or 429 is worth waiting for, if not for how long it said, then
    for `DEFAULT_RETRY_AFTER`: a hosted provider that is overloaded says so
    with a bare 503 as often as with a header.
    """
    if getattr(error, "status_code", None) not in (429, 503):
        return None

    reason = _PROXY_REFUSALS.get(str(getattr(error, "code", None)))
    detail = _error_detail(error)

    if reason is UnavailableReason.DOWN:
        return ModelUnavailable(model, reason, retry_after=None, detail=detail)

    return ModelUnavailable(
        model,
        reason or UnavailableReason.BUSY,
        retry_after=_retry_after(error) or DEFAULT_RETRY_AFTER,
        detail=detail,
    )


def _sources(error: Exception) -> list[object]:
    """The error and the SDK error any-llm wrapped, which keeps the response."""
    original = getattr(error, "original_exception", None)
    return [error] if original is None else [error, original]


def _retry_after(error: Exception) -> int | None:
    """`Retry-After` in whole seconds, if any source of `error` carries one.

    Only the delta-seconds form: every provider we use sends that, and an
    unparsable value is no reason to turn a wait into a failure.
    """
    for source in _sources(error):
        # any-llm sets `retry_after` on its RateLimitError only.
        value = getattr(source, "retry_after", None)
        if value is None:
            headers = getattr(getattr(source, "response", None), "headers", None)
            value = headers.get("retry-after") if headers is not None else None
        if value is None:
            continue
        try:
            return max(math.ceil(float(value)), 0)
        except (TypeError, ValueError):
            continue
    return None


def _error_detail(error: Exception) -> str:
    """The `detail` of the error's JSON body, as the proxy writes it, else its
    message."""
    for source in _sources(error):
        body = getattr(source, "body", None)
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            return body["detail"]
    return str(error)


async def _completion(**kwargs: Any) -> ChatCompletion:
    """`acompletion`, with a provider that cannot serve raised as such."""
    try:
        # `chat` never streams, so this is never the chunk iterator.
        return cast(ChatCompletion, await acompletion(**kwargs))
    except Exception as error:
        if (unavailable := model_unavailable(error, kwargs["model"])) is not None:
            raise unavailable from error
        raise


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

    chat: Callable[..., CoroutineType[Any, Any, ChatCompletion]] = partial(
        _completion,
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
            api_key=api_keys.openrouter_api_key.get_secret_value(),
            reasoning_effort="low",
            **openrouter_model_kwargs(model),
        )
    elif model.startswith("openai:"):
        chat_completion = await chat(
            model=model,
            api_key=api_keys.openai_api_key.get_secret_value(),
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
            api_key=api_keys.google_ai_api_key.get_secret_value(),
        )
    elif model.startswith("alex:"):
        extra_model_kwargs = {}

        match model.removeprefix("alex:"):
            case "glm-5.3-flash":
                extra_model_kwargs["reasoning_effort"] = "low"
                extra_model_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"clear_thinking": True}
                }
            case "qwen3.8-flash-next":
                extra_model_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }

        chat_completion = await chat(
            messages=messages,
            model=model.replace("alex:", "openai:"),
            api_key=api_keys.alex_api_key.get_secret_value(),
            api_base="https://inference.alexandra.dk/v1",
            **extra_model_kwargs,
        )
    elif is_self_hosted(model):
        server_endpoint = settings.llm.vllm_api_base

        # TODO: re-enable once logit bias is validated against gpt-oss-120b in staging
        extra_model_kwargs = {}  # get_extra_model_kwargs(model, response_format is not None)

        # Only as a default: an argument given here would replace one the
        # caller chose, as keywords passed at the call win over the partial's.
        if (
            settings.llm.vllm_max_tokens is not None
            and "max_tokens" not in model_kwargs
        ):
            extra_model_kwargs["max_tokens"] = settings.llm.vllm_max_tokens
        if (
            settings.llm.vllm_reasoning_effort is not None
            and "reasoning_effort" not in model_kwargs
        ):
            extra_model_kwargs["reasoning_effort"] = settings.llm.vllm_reasoning_effort

        chat_completion = await chat(
            model="vllm:" + model,
            api_base=server_endpoint,
            api_key=api_keys.vllm_api_key.get_secret_value(),
            # No retries in the SDK. Our proxy answers a cold start with a 503
            # and `Retry-After`, which the SDK honours twice over -- a minute
            # and a half of silence per call before anyone hears of it, and
            # then the next call does the same. Raised at once as
            # `ModelUnavailable`, the caller can tell the respondent and
            # decide how long to wait.
            client_args={"max_retries": 0},
            **extra_model_kwargs,
        )
    else:
        # A prefix in HOSTED_PREFIXES with no branch above. Sending it to vLLM
        # would be wrong, and the backend's warm-up would not have started it.
        raise ValueError(
            f"No client for {model!r}: its prefix is in HOSTED_PREFIXES but "
            "`chat` has no branch for it"
        )

    if response_format:
        reply = chat_completion.choices[0].message
        parsed = getattr(reply, "parsed", None)
        if parsed is None:
            reason = getattr(reply, "refusal", None) or "no parsable output"
            raise StructuredOutputError(
                f"{model} returned no {response_format.__name__}: {reason}"
            )
        return parsed

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

    data: dict[str, Any] = {
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
