"""The generators that write interview-guide parts with an LLM, driven offline.

Two paths generate guide parts: `interview_guides.generate`, which the
dashboard and its assistant call with a user's instructions, and
`GuideAgent`, which writes sections and questions mid-interview from the
transcript. Both are run here with the LLM call stubbed, recording the
messages it was sent.
"""

import asyncio

import pytest

from ainterviewer.agents.guide_agent import GuideAgent
from ainterviewer.interview_guides import (
    InterviewGuide,
    Question,
    QuestionSection,
    generate,
)
from ainterviewer.interview_guides.questions import QuestionBase

SECTION = QuestionSection(
    description="Background",
    questions=[Question(main_question="What brought you here?")],
)
GUIDE = InterviewGuide(framing="A study of commuting.", question_sections=[SECTION])
GENERATED = QuestionBase(main_question="How do you get to work?", max_probes_n=None)
# What the stubbed LLM answers, by the kind of part asked for: a guide holds
# sections, a section holds questions, and a question is `GENERATED`.
ANSWERS = {
    "InterviewGuideTemplate": {
        "framing": "Commuting",
        "introduction": "Hello",
        "question_sections": [
            {"description": "Travel", "questions": [GENERATED.model_dump()]}
        ],
    },
    "QuestionSectionTemplate": {
        "description": "Travel",
        "questions": [GENERATED.model_dump()],
    },
}


@pytest.fixture
def sent(monkeypatch) -> list[list]:
    """Stub the LLM, recording the messages sent."""
    calls: list[list] = []

    async def chat(messages, model, response_format, **kwargs):
        calls.append(messages)
        title = response_format.model_config.get("title")
        return response_format.model_validate(
            ANSWERS.get(title, GENERATED.model_dump())
        )

    monkeypatch.setattr(generate, "chat", chat)
    monkeypatch.setattr("ainterviewer.agents.base.chat", chat)
    return calls


def test_generate_question_sets_each_probe_limit_on_its_own_field(sent):
    question = asyncio.run(
        generate.generate_question(
            "Ask about the commute",
            "openai:test",
            GUIDE,
            max_probes_n=2,
            max_probes_time=90,
        )
    )

    assert (question.max_probes_n, question.max_probes_time) == (2, 90)


def test_guide_agent_sets_each_probe_limit_on_its_own_field(sent):
    agent = GuideAgent(model="openai:test", language="EN")

    question = asyncio.run(
        agent.generate_main_question(
            "Interviewer: Hello", GUIDE, SECTION, max_probes_n=2, max_probes_time=90
        )
    )

    assert (question.max_probes_n, question.max_probes_time) == (2, 90)


def prompts(messages) -> tuple[str, str]:
    """The (system, user) prompt of one LLM call."""
    (system, user) = messages
    return system["content"], user["content"]


def test_guide_agent_shows_the_guide_and_section_as_their_text(sent):
    agent = GuideAgent(model="openai:test", language="EN")

    asyncio.run(agent.generate_main_question("Interviewer: Hello", GUIDE, SECTION))
    asyncio.run(agent.generate_question_section("Interviewer: Hello", GUIDE))

    for call in sent:
        _, user = prompts(call)
        assert GUIDE.prompt_json() in user
        assert str(SECTION.id) not in user
    assert SECTION.prompt_json() in prompts(sent[0])[1]
