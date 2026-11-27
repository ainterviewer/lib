"""How closely the probes follow a question's suggested probes
(`Question.probe_steering`)."""

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from ainterviewer.agents.probing_agent import (
    ProbingAgent,
    SuggestedProbe,
    suggested_probe_model,
)
from ainterviewer.interview_guides.history import HistoryMessage, Turn
from ainterviewer.interview_guides.types import ProbeSteering
from tests.test_condition_branches import Respondent, _interviewer
from tests.test_interview_loop import _plain_guide

SUGGESTIONS = ["Why?", "When?", "Who else?"]


def _setup(steering="loose", probes=SUGGESTIONS, answered=None, asked=()):
    guide = _plain_guide(probes=probes, max_probes_n=3, probe_steering=steering)
    interviewer = _interviewer(guide, Respondent({}))
    interviewer.classification_agent.answered_suggestions = AsyncMock(
        return_value=answered or []
    )
    interviewer.probing_agent.generate_probe = AsyncMock(return_value="Own probe?")
    interviewer.probing_agent.generate_suggested_probe = AsyncMock(
        side_effect=lambda **kwargs: SuggestedProbe(
            suggestion=kwargs["suggestions"][0], question=kwargs["suggestions"][0]
        )
    )
    question = guide.question_sections[0].questions[0]
    history = interviewer.interview_history
    history.add_section("Section")
    history.add_question(
        question_description="",
        main_question=Turn(question=HistoryMessage(message="First?")),
    )
    for probe in asked:
        history.add_probe(probe=Turn(question=HistoryMessage(message=probe)))
    return interviewer, question


async def _generate(interviewer, question):
    return await interviewer.generate_probe(
        section_description="Section", question=question, transcript="Q: First?"
    )


@pytest.mark.anyio
async def test_loose_asks_the_general_agent():
    interviewer, question = _setup()

    assert await _generate(interviewer, question) == "Own probe?"
    interviewer.probing_agent.generate_suggested_probe.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("steering", ["reword", "strict"])
async def test_reword_and_strict_pick_from_the_open_suggestions(steering):
    interviewer, question = _setup(steering, answered=["Why?"])

    assert await _generate(interviewer, question) == "When?"
    pick = interviewer.probing_agent.generate_suggested_probe.call_args.kwargs
    assert pick["steering"] == steering
    assert pick["suggestions"] == ["When?", "Who else?"]
    interviewer.probing_agent.generate_probe.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("steering", ["reword", "strict"])
async def test_nothing_is_asked_once_every_suggestion_is_used(steering):
    interviewer, question = _setup(steering, answered=SUGGESTIONS)

    assert await _generate(interviewer, question) is None
    interviewer.probing_agent.generate_suggested_probe.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("steering", ["reword", "strict"])
async def test_without_suggestions_they_probe_as_loose(steering):
    interviewer, question = _setup(steering, probes=None)

    assert await _generate(interviewer, question) == "Own probe?"


@pytest.mark.anyio
async def test_the_pick_is_logged():
    interviewer, question = _setup("strict")

    await _generate(interviewer, question)

    task = interviewer.db.insert_task.call_args.kwargs
    assert task["task"] == "suggested_probe"
    assert task["reason"] == "strict"
    assert json.loads(task["content"]) == SUGGESTIONS
    assert json.loads(task["response"]) == {"suggestion": "Why?", "question": "Why?"}


@pytest.mark.anyio
async def test_a_failed_pick_asks_the_first_suggestion_as_written():
    interviewer, question = _setup("reword")
    interviewer.probing_agent.generate_suggested_probe = AsyncMock(
        side_effect=RuntimeError("model down")
    )

    assert await _generate(interviewer, question) == "Why?"
    response = json.loads(interviewer.db.insert_task.call_args.kwargs["response"])
    assert response["error"] == "RuntimeError: model down"


@pytest.mark.anyio
async def test_suggestions_asked_word_for_word_are_left_out_at_every_level():
    interviewer, question = _setup(asked=["why", "  When? "])
    interviewer.classification_agent.answered_suggestions.side_effect = RuntimeError

    # Left out even though the classifier failed.
    assert await interviewer.unanswered_suggestions(question, "Q: First?") == [
        "Who else?"
    ]


@pytest.mark.anyio
async def test_a_strict_probe_skips_the_message_preprocessing():
    interviewer, question = _setup("strict")
    interviewer.preprocess_message = AsyncMock(return_value="cut")
    interviewer.receive_data = AsyncMock(return_value="answer")

    await interviewer.ask_probe(question, " Why? And how? ", verbatim=True)

    interviewer.preprocess_message.assert_not_awaited()
    assert interviewer.interview_history.current_question.probes[
        -1
    ].question.message == ("Why? And how?")


def _parse(model, **fields):
    """The fields of a reply to a model built at runtime."""
    return model.model_validate(fields).model_dump()


def test_strict_only_accepts_the_suggestions_as_written():
    model = suggested_probe_model(ProbeSteering.STRICT, SUGGESTIONS)

    assert model.model_json_schema()["properties"]["suggestion"]["enum"] == SUGGESTIONS
    assert _parse(model, suggestion="When?") == {"suggestion": "When?"}
    with pytest.raises(ValidationError):
        _parse(model, suggestion="When did it start?")


def test_reword_picks_by_number():
    model = suggested_probe_model(ProbeSteering.REWORD, SUGGESTIONS)

    assert _parse(model, suggestion=3, question="Who else?")["suggestion"] == 3
    with pytest.raises(ValidationError):
        _parse(model, suggestion=4, question="x")


def _agent(*replies):
    agent = ProbingAgent(language="EN", interview_framing="Framing", model="test")
    agent.chat_api = AsyncMock(side_effect=list(replies))
    return agent


async def _pick(agent, steering, suggestions=SUGGESTIONS):
    return await agent.generate_suggested_probe(
        steering=steering,
        section_description="Section",
        question_description="",
        main_question="First?",
        transcript="Q: First?\nA: Yes",
        suggestions=suggestions,
        previous_probes=[],
    )


@pytest.mark.anyio
async def test_the_agent_maps_a_reworded_pick_back_to_its_suggestion():
    model = suggested_probe_model(ProbeSteering.REWORD, SUGGESTIONS)
    agent = _agent(
        model.model_validate({"suggestion": 2, "question": "When was that, roughly?"})
    )

    assert await _pick(agent, ProbeSteering.REWORD) == SuggestedProbe(
        suggestion="When?", question="When was that, roughly?"
    )
    prompt = agent.chat_api.call_args.args[0][-1]["content"]
    assert "1. Why?\n2. When?\n3. Who else?" in prompt
    assert "Reword the chosen suggestion" in prompt


@pytest.mark.anyio
async def test_the_agent_tries_again_when_the_reply_does_not_fit():
    model = suggested_probe_model(ProbeSteering.STRICT, SUGGESTIONS)
    try:
        model.model_validate({"suggestion": "nope"})
    except ValidationError as error:
        invalid = error

    agent = _agent(invalid, model.model_validate({"suggestion": "Who else?"}))

    picked = await _pick(agent, ProbeSteering.STRICT)
    assert picked == SuggestedProbe(suggestion="Who else?", question="Who else?")
    assert agent.chat_api.await_count == 2


@pytest.mark.anyio
async def test_duplicate_suggestions_are_shown_once():
    model = suggested_probe_model(ProbeSteering.STRICT, ["Why?"])
    agent = _agent(model.model_validate({"suggestion": "Why?"}))

    await _pick(agent, ProbeSteering.STRICT, suggestions=["Why?", "Why?"])

    prompt = agent.chat_api.call_args.args[0][-1]["content"]
    assert "1. Why?\n" in prompt
    assert "2. " not in prompt
