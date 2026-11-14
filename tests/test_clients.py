import asyncio
from types import SimpleNamespace

import pytest

from ainterviewer.exceptions import StructuredOutputError
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.lpm import clients
from ainterviewer.lpm.clients import openrouter_model_kwargs
from ainterviewer.lpm.types import Message, MessageRole


def test_gpt_oss_prefers_deepinfra():
    assert openrouter_model_kwargs("openrouter:openai/gpt-oss-120b") == {
        "extra_body": {"provider": {"order": ["deepinfra"]}}
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
