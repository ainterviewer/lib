"""Interview-guide parts written by an LLM to a researcher's instructions.

These run before any interview -- the dashboard's generate buttons and its
guide assistant call them. They share the guide agent's system prompt, which
holds what makes a good interview question, with `GuideAgent`, which writes
sections and questions mid-interview from the transcript instead.
"""

import asyncio
import json
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ainterviewer.agents.prompts.agent_prompts import GuideAgentPrompts
from ainterviewer.agents.prompts.get_prompts import get_agent_prompts
from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewGuideTemplate,
    Question,
    QuestionSection,
)
from ainterviewer.interview_guides.questions import QuestionBaseExtended
from ainterviewer.interview_guides.sections import QuestionSectionBaseExtendedTemplate
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message, MessageRole
from ainterviewer.types import LanguageCode


@cache
def _prompts(language: LanguageCode) -> GuideAgentPrompts:
    return get_agent_prompts("GuideAgent", lang=language)


async def _generate[T: BaseModel](
    component: Literal["interview guide", "question section", "main question"],
    instructions: str,
    model: str,
    schema: type[T],
    language: LanguageCode | None,
    guide: InterviewGuide | None = None,
    section: QuestionSection[Question] | None = None,
) -> T:
    prompts = _prompts(language or "EN")
    messages = [
        Message(role=MessageRole.SYSTEM, content=prompts.system_prompt),
        Message(
            role=MessageRole.USER,
            content=prompts.generate_from_instructions_prompt(
                component,
                instructions,
                schema,
                interview_guide=guide,
                section=section,
            ),
        ),
    ]
    return await chat(messages=messages, model=model, response_format=schema)


async def generate_interview_guide(
    prompt: str,
    model: str,
    *,
    output_path: str | None = None,
    language: LanguageCode | None = None,
) -> InterviewGuide:
    """A whole interview guide, written to `prompt` in `language` (English by
    default), and saved to `output_path` as JSON when one is given."""
    template = await _generate(
        "interview guide", prompt, model, InterviewGuideTemplate, language
    )
    interview_guide = InterviewGuide.model_validate(template.model_dump())

    if output_path:
        await asyncio.to_thread(
            Path(output_path).write_text,
            json.dumps(interview_guide.model_dump(mode="json"), indent=4),
        )

    return interview_guide


async def generate_section(
    instruction: str,
    model: str,
    guide: InterviewGuide,
    *,
    language: LanguageCode | None = None,
) -> QuestionSection[Question]:
    """A new section for `guide`, written to `instruction` in `language`."""
    # TODO: Let the caller constrain the number of questions, and probably the
    # total time spent.
    template = await _generate(
        "question section",
        instruction,
        model,
        QuestionSectionBaseExtendedTemplate,
        language,
        guide=guide,
    )
    return QuestionSection[Question].model_validate(template.model_dump())


async def generate_question(
    instructions: str,
    model: str,
    guide: InterviewGuide,
    section: QuestionSection[Question] | None = None,
    max_probes_n: int | None = None,
    max_probes_time: int | None = None,
    *,
    language: LanguageCode | None = None,
) -> Question:
    """A new question for `guide`, written to `instructions` in `language`, to
    go at the end of `section` when one is given."""
    base = await _generate(
        "main question",
        instructions,
        model,
        QuestionBaseExtended,
        language,
        guide=guide,
        section=section,
    )
    question = Question.model_validate(base.model_dump())

    if max_probes_n is not None:
        question.max_probes_n = max_probes_n

    if max_probes_time is not None:
        question.max_probes_time = max_probes_time

    return question
