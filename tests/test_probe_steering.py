"""How closely the probes follow a question's suggested probes
(`Question.probe_steering`)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from ainterviewer.agents.probing_agent import (
    ProbingAgent,
    SuggestedProbe,
    suggested_probe_model,
)
from ainterviewer.exceptions import ModelUnavailable, UnavailableReason
from ainterviewer.interview_guides.history import HistoryMessage, Turn
from ainterviewer.interview_guides.types import ProbeSteering
from ainterviewer.types import MessageRole
from tests.test_condition_branches import Respondent, _interviewer
from tests.test_history import StoredMessage
from tests.test_interview_loop import _plain_guide, _stored_rows

SUGGESTIONS = ["Why?", "When?", "Who else?"]


def _setup(
    steering="loose",
    probes=SUGGESTIONS,
    answered=None,
    asked=(),
    respondent=None,
    **question,
):
    guide = _plain_guide(
        probes=probes, probe_steering=steering, **{"max_probes_n": 3, **question}
    )
    interviewer = _interviewer(guide, respondent or Respondent({}))
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
        # A pair is a reworded probe and the suggestion it was asked from.
        message, suggestion = probe if isinstance(probe, tuple) else (probe, None)
        history.add_probe(
            probe=Turn(
                question=HistoryMessage(message=message), suggested_probe=suggestion
            )
        )
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

    assert await _generate(interviewer, question) == SuggestedProbe(
        suggestion="When?", question="When?"
    )
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

    assert await _generate(interviewer, question) == SuggestedProbe(
        suggestion="Why?", question="Why?"
    )
    response = json.loads(interviewer.db.insert_task.call_args.kwargs["response"])
    assert response["error"] == "RuntimeError: model down"


@pytest.mark.anyio
async def test_a_missing_model_does_not_ask_the_first_suggestion():
    """Asking a suggestion as written would carry on as if nothing were the
    matter, and leave the respondent no wiser about the outage."""
    interviewer, question = _setup("reword")
    interviewer.probing_agent.generate_suggested_probe = AsyncMock(
        side_effect=ModelUnavailable("test", UnavailableReason.BUSY, retry_after=5)
    )

    with pytest.raises(ModelUnavailable):
        await _generate(interviewer, question)
    tasks = [call.kwargs["task"] for call in interviewer.db.insert_task.call_args_list]
    assert "suggested_probe" not in tasks


@pytest.mark.anyio
async def test_suggestions_asked_word_for_word_are_left_out_at_every_level():
    interviewer, question = _setup(asked=["why", "  When? "])
    interviewer.classification_agent.answered_suggestions.side_effect = RuntimeError

    # Left out even though the classifier failed.
    assert await interviewer.unanswered_suggestions(question, "Q: First?") == [
        "Who else?"
    ]


@pytest.mark.anyio
async def test_suggestions_are_filled_in_before_they_are_pruned():
    interviewer, question = _setup(
        "strict",
        probes=["Why {{ product }}?", "When did {{ product }} start?"],
        asked=["Why Acme?"],
    )
    interviewer.referable_values["product"] = "Acme"

    # Asked as filled in, so pruned as filled in.
    assert await interviewer.unanswered_suggestions(question, "Q: First?") == [
        "When did Acme start?"
    ]
    picked = await _generate(interviewer, question)
    assert picked == SuggestedProbe(
        suggestion="When did Acme start?", question="When did Acme start?"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("classifier_fails", [False, True])
async def test_a_reworded_suggestion_is_not_asked_again(classifier_fails):
    # Reworded, and left unanswered: neither the text nor the classifier
    # rules it out, only the mark.
    interviewer, question = _setup(
        "reword", asked=[("So why was that, do you think?", "Why?")]
    )
    if classifier_fails:
        interviewer.classification_agent.answered_suggestions.side_effect = RuntimeError

    assert await interviewer.unanswered_suggestions(question, "Q: First?") == [
        "When?",
        "Who else?",
    ]


def _probing(interviewer):
    """Probe without the classifiers and the security check, which would
    reach a model."""
    interviewer.contains_refusal = AsyncMock(return_value=False)
    interviewer.has_main_question_been_exhausted = AsyncMock(return_value=False)
    interviewer.start_security_check = MagicMock()


def _reworded(**kwargs):
    suggestion = kwargs["suggestions"][0]
    return SuggestedProbe(suggestion=suggestion, question=f"Reworded: {suggestion}")


async def _probe(interviewer, question):
    await interviewer.probe(question, "Section")
    return [
        probe.question.message
        for probe in interviewer.interview_history.current_question.probes
    ]


@pytest.mark.anyio
async def test_reword_asks_each_suggestion_once_and_then_stops():
    answers = {f"Reworded: {s}": ["Not sure"] for s in SUGGESTIONS}
    interviewer, question = _setup(
        "reword", respondent=Respondent(answers), max_probes_n=10
    )
    interviewer.probing_agent.generate_suggested_probe.side_effect = _reworded
    _probing(interviewer)

    assert await _probe(interviewer, question) == [
        "Reworded: Why?",
        "Reworded: When?",
        "Reworded: Who else?",
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("steering", ["reword", "strict"])
async def test_steering_keeps_to_the_probe_limit(steering):
    interviewer, question = _setup(
        steering, respondent=Respondent({"Why?": ["a"]}), max_probes_n=1
    )
    _probing(interviewer)

    assert await _probe(interviewer, question) == ["Why?"]


@pytest.mark.anyio
@pytest.mark.parametrize("steering", ["reword", "strict"])
async def test_steering_stops_once_the_question_is_exhausted(steering):
    interviewer, question = _setup(
        steering,
        respondent=Respondent({"Why?": ["a"]}),
        max_probes_n=10,
        check_if_exhausted=True,
    )
    _probing(interviewer)
    interviewer.has_main_question_been_exhausted.return_value = True

    assert await _probe(interviewer, question) == ["Why?"]


@pytest.mark.anyio
async def test_the_suggestion_a_probe_was_asked_from_is_stored():
    interviewer, question = _setup(
        "reword", respondent=Respondent({"Reworded: Why?": ["Not sure"]})
    )
    interviewer.probing_agent.generate_suggested_probe.side_effect = _reworded
    _probing(interviewer)
    interviewer.can_probe = AsyncMock(side_effect=[True, False])

    await interviewer.probe(question, "Section")

    probe, _answer = _stored_rows(interviewer)[-2:]
    assert (probe.content, probe.suggested_probe) == ("Reworded: Why?", "Why?")


@pytest.mark.anyio
async def test_a_resumed_interview_knows_which_suggestion_was_asked():
    stored = [
        StoredMessage("First?", MessageRole.ASSISTANT),
        StoredMessage("Yes", MessageRole.USER),
        StoredMessage(
            "So why was that?",
            MessageRole.ASSISTANT,
            sub_question=1,
            suggested_probe="Why?",
        ),
        StoredMessage("Not sure", MessageRole.USER, sub_question=1),
    ]
    guide = _plain_guide(probes=SUGGESTIONS, probe_steering="reword")
    resumed = _interviewer(guide, Respondent({}))
    resumed.classification_agent.answered_suggestions = AsyncMock(return_value=[])

    resumed.interview_history.process_history(stored, guide)

    question = guide.question_sections[0].questions[0]
    assert await resumed.unanswered_suggestions(question, "Q: First?") == [
        "When?",
        "Who else?",
    ]


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
