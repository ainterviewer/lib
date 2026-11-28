"""Suggested probes the respondent has already answered are left out of the
probing prompt."""

from unittest.mock import AsyncMock

import pytest

from ainterviewer.agents.classification_agent import (
    AnsweredSuggestionsModel,
    ClassificationAgent,
)
from ainterviewer.exceptions import ModelUnavailable, UnavailableReason
from ainterviewer.interview_guides.history import HistoryMessage, Turn
from tests.test_condition_branches import Respondent, _interviewer
from tests.test_interview_loop import _plain_guide

SUGGESTIONS = ["Why?", "When?", "Who else?"]
UNAVAILABLE = ModelUnavailable("test", UnavailableReason.STARTING, retry_after=60)


def _probing_setup(answered=None, error=None, probes=SUGGESTIONS):
    guide = _plain_guide(probes=probes, max_probes_n=2)
    interviewer = _interviewer(guide, Respondent({}))
    interviewer.classification_agent.answered_suggestions = AsyncMock(
        return_value=answered or [], side_effect=error
    )
    interviewer.probing_agent.generate_probe = AsyncMock(return_value="Probe?")
    question = guide.question_sections[0].questions[0]
    history = interviewer.interview_history
    history.add_section("Section")
    history.add_question(
        question_description="",
        main_question=Turn(question=HistoryMessage(message="First?")),
    )
    return interviewer, question


async def _suggested_probes(interviewer, question) -> str | None:
    await interviewer.generate_probe(
        section_description="Section", question=question, transcript="Q: First?"
    )
    return interviewer.probing_agent.generate_probe.call_args.kwargs["suggested_probes"]


@pytest.mark.anyio
async def test_answered_suggestions_are_left_out():
    interviewer, question = _probing_setup(answered=["When?"])

    assert await _suggested_probes(interviewer, question) == "- Why?\n- Who else?"
    classify = interviewer.classification_agent.answered_suggestions
    assert classify.call_args.kwargs == {
        "transcript": "Q: First?",
        "main_question": "First?",
        "suggestions": SUGGESTIONS,
    }


@pytest.mark.anyio
async def test_the_dropped_suggestions_are_logged():
    interviewer, question = _probing_setup(answered=["When?"])

    await _suggested_probes(interviewer, question)

    task = interviewer.db.insert_task.call_args.kwargs
    assert task["task"] == "answered_suggestions"
    assert task["response"] == '["When?"]'


@pytest.mark.anyio
async def test_no_suggestions_are_shown_when_all_are_answered():
    interviewer, question = _probing_setup(answered=SUGGESTIONS)

    assert await _suggested_probes(interviewer, question) is None


@pytest.mark.anyio
async def test_all_suggestions_are_kept_when_the_classification_fails():
    interviewer, question = _probing_setup(error=RuntimeError("model down"))

    assert (
        await _suggested_probes(interviewer, question) == "- Why?\n- When?\n- Who else?"
    )


@pytest.mark.anyio
async def test_a_missing_model_is_not_fallen_back_on():
    """The probe needs the same model; carrying on would spend its call
    finding out, and keep the respondent waiting for both."""
    interviewer, question = _probing_setup(error=UNAVAILABLE)

    with pytest.raises(ModelUnavailable):
        await _suggested_probes(interviewer, question)
    interviewer.probing_agent.generate_probe.assert_not_awaited()


@pytest.mark.anyio
async def test_nothing_is_classified_without_suggestions():
    interviewer, question = _probing_setup(probes=None)

    assert await _suggested_probes(interviewer, question) is None
    interviewer.classification_agent.answered_suggestions.assert_not_awaited()


@pytest.mark.anyio
async def test_the_agent_maps_the_numbers_back_to_suggestions():
    agent = ClassificationAgent(model="test", language="EN")
    agent.chat_api = AsyncMock(
        # 9 is not a suggestion's number and is ignored.
        return_value=AnsweredSuggestionsModel(answered=[3, 1, 9])
    )

    answered = await agent.answered_suggestions(
        transcript="Q: First?\nA: Yes", main_question="First?", suggestions=SUGGESTIONS
    )

    assert answered == ["Why?", "Who else?"]
    system, user = agent.chat_api.call_args.args[0]
    assert system["role"] == "system"
    prompt = user["content"]
    assert '"First?"' in prompt
    assert "1. Why?\n2. When?\n3. Who else?" in prompt
