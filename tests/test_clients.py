import asyncio
from types import SimpleNamespace

import pytest

from ainterviewer.exceptions import StructuredOutputError
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.lpm import clients
from ainterviewer.lpm.clients import openrouter_model_kwargs
from ainterviewer.lpm.types import Message, MessageRole


def test_gpt_oss_prefers_bedrock_eu():
    assert openrouter_model_kwargs("openrouter:openai/gpt-oss-120b") == {
        "extra_body": {"provider": {"order": ["amazon-bedrock/eu-west-1"]}}
    }


def test_other_models_use_openrouter_routing():
    assert openrouter_model_kwargs("openrouter:z-ai/glm-5.3-flash") == {}


def _completion(**message):
    """A minimal completion with one choice carrying `message`."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(**message))]
    )


def _structured_chat(monkeypatch, **message):
    async def acompletion(**kwargs):
        return _completion(**message)

    monkeypatch.setattr(clients, "acompletion", acompletion)
    return asyncio.run(
        clients.chat(
            [Message(role=MessageRole.USER, content="Hi")],
            model="openai:test",
            response_format=QuestionBase,
        )
    )


def test_structured_chat_returns_the_parsed_model(monkeypatch):
    question = QuestionBase(main_question="How do you get to work?")
    assert _structured_chat(monkeypatch, parsed=question) is question


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ({"parsed": None, "refusal": "I can't help with that."}, "I can't help"),
        ({"parsed": None, "refusal": None}, "no parsable output"),
        ({"content": "{}"}, "no parsable output"),
    ],
)
def test_structured_chat_raises_when_nothing_parsed(monkeypatch, message, reason):
    # Returning None here used to surface as an AttributeError in the caller.
    with pytest.raises(StructuredOutputError, match=reason):
        _structured_chat(monkeypatch, **message)


def _sent_kwargs(monkeypatch, model: str, **chat_kwargs) -> dict:
    """The arguments `chat()` hands the provider for one plain call."""
    sent: dict = {}

    async def acompletion(**kwargs):
        sent.update(kwargs)
        return _completion(content="Hello")

    monkeypatch.setattr(clients, "acompletion", acompletion)
    asyncio.run(
        clients.chat(
            [Message(role=MessageRole.USER, content="Hi")],
            model=model,
            **chat_kwargs,
        )
    )
    return sent


def test_vllm_calls_get_the_max_tokens_ceiling(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_max_tokens", 4096)
    assert _sent_kwargs(monkeypatch, "openai/gpt-oss-120b")["max_tokens"] == 4096


def test_a_callers_max_tokens_wins_over_the_ceiling(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_max_tokens", 4096)
    sent = _sent_kwargs(monkeypatch, "openai/gpt-oss-120b", max_tokens=200)
    assert sent["max_tokens"] == 200


def test_no_ceiling_when_switched_off(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_max_tokens", None)
    assert "max_tokens" not in _sent_kwargs(monkeypatch, "openai/gpt-oss-120b")


def test_hosted_providers_get_no_ceiling(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_max_tokens", 4096)
    assert "max_tokens" not in _sent_kwargs(monkeypatch, "openai:gpt-5-mini")


def test_vllm_calls_reason_at_the_configured_effort(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_reasoning_effort", "low")
    sent = _sent_kwargs(monkeypatch, "openai/gpt-oss-120b")
    assert sent["reasoning_effort"] == "low"


def test_a_callers_reasoning_effort_wins(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_reasoning_effort", "low")
    sent = _sent_kwargs(monkeypatch, "openai/gpt-oss-120b", reasoning_effort="high")
    assert sent["reasoning_effort"] == "high"


def test_reasoning_effort_left_to_the_model_when_unset(monkeypatch):
    monkeypatch.setattr(clients.settings.llm, "vllm_reasoning_effort", None)
    assert "reasoning_effort" not in _sent_kwargs(monkeypatch, "openai/gpt-oss-120b")
