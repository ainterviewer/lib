"""A provider that cannot serve is raised as `ModelUnavailable`, with what it
said about why and for how long.

The refusals are pushed through the real OpenAI SDK and any-llm over a mock
transport, because what is tested is that the proxy's body survives both:
the SDK reads `code` off it, any-llm carries it on, the response keeps the
`Retry-After` header.
"""

import asyncio

import httpx
import pytest
from any_llm import acompletion

from ainterviewer.exceptions import ModelUnavailable, UnavailableReason
from ainterviewer.lpm import clients
from ainterviewer.lpm.clients import DEFAULT_RETRY_AFTER, model_unavailable
from ainterviewer.lpm.types import Message, MessageRole

MESSAGES = [Message(role=MessageRole.USER, content="Hi")]


def _refusing(status: int, body: dict | None = None, headers=None):
    """A transport answering every request with `status`, counting them."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, json=body or {}, headers=headers)

    return httpx.MockTransport(handler), calls


def _chat(monkeypatch, transport, model="gpt-oss-120b"):
    """`clients.chat` with the real any-llm call, over `transport`."""

    async def over_transport(**kwargs):
        client_args = {
            **(kwargs.pop("client_args", None) or {}),
            "http_client": httpx.AsyncClient(transport=transport),
        }
        return await acompletion(**kwargs, client_args=client_args)

    monkeypatch.setattr(clients, "acompletion", over_transport)
    return asyncio.run(clients.chat(MESSAGES, model=model))


@pytest.fixture(params=[True, False], ids=["unified", "sdk-errors"])
def unified_exceptions(request, monkeypatch):
    """Both error shapes: the backend turns any-llm's unified errors on, but
    the library must not depend on it."""
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1" if request.param else "0")


@pytest.mark.usefixtures("unified_exceptions")
@pytest.mark.parametrize(
    ("body", "headers", "reason", "retry_after"),
    [
        (
            {"detail": "EC2 instance initializing", "code": "capacity_starting"},
            {"Retry-After": "87"},
            UnavailableReason.STARTING,
            87,
        ),
        (
            {"detail": "Capacity saturated", "code": "capacity_busy"},
            {"Retry-After": "5"},
            UnavailableReason.BUSY,
            5,
        ),
        # Final: nothing to wait for, whatever a header might say.
        (
            {"detail": "No EC2 instance available.", "code": "capacity_unavailable"},
            None,
            UnavailableReason.DOWN,
            None,
        ),
        (
            {"detail": "EC2 inference not enabled", "code": "inference_disabled"},
            None,
            UnavailableReason.DOWN,
            None,
        ),
        # Someone else's 503, saying nothing: worth a short wait.
        ({}, None, UnavailableReason.BUSY, DEFAULT_RETRY_AFTER),
    ],
)
def test_a_refusal_is_raised_with_its_reason(
    monkeypatch, body, headers, reason, retry_after
):
    transport, _ = _refusing(503, body, headers)

    with pytest.raises(ModelUnavailable) as raised:
        _chat(monkeypatch, transport)

    assert raised.value.reason is reason
    assert raised.value.retry_after == retry_after
    assert raised.value.worth_waiting is (retry_after is not None)
    if "detail" in body:
        assert raised.value.detail == body["detail"]


@pytest.mark.usefixtures("unified_exceptions")
def test_the_self_hosted_model_is_asked_once(monkeypatch):
    """No retries in the SDK: it would sit through `Retry-After` twice before
    anyone heard the model was not there."""
    transport, calls = _refusing(
        503, {"code": "capacity_starting"}, {"Retry-After": "1"}
    )

    with pytest.raises(ModelUnavailable):
        _chat(monkeypatch, transport)

    assert len(calls) == 1


@pytest.mark.usefixtures("unified_exceptions")
def test_a_rate_limit_is_busy(monkeypatch):
    transport, _ = _refusing(
        429, {"error": {"message": "slow down"}}, {"Retry-After": "0"}
    )

    with pytest.raises(ModelUnavailable) as raised:
        _chat(monkeypatch, transport)

    assert raised.value.reason is UnavailableReason.BUSY
    # Zero is "now", and a wait of nothing is no wait: the default instead.
    assert raised.value.retry_after == DEFAULT_RETRY_AFTER


@pytest.mark.usefixtures("unified_exceptions")
@pytest.mark.parametrize("status", [500, 502, 400])
def test_other_failures_are_left_alone(monkeypatch, status):
    """Only 503 and 429 say "not now"; anything else is the provider's own
    error, for whoever handles it today."""
    transport, _ = _refusing(status, {"detail": "nope", "code": "upstream_unreachable"})

    with pytest.raises(Exception) as raised:
        _chat(monkeypatch, transport)

    assert not isinstance(raised.value, ModelUnavailable)


def test_errors_without_a_status_are_not_refusals():
    assert model_unavailable(TimeoutError("slow"), "m") is None
    assert model_unavailable(ValueError("bad"), "m") is None
