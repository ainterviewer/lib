import asyncio
from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model

from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewMessage,
)
from ainterviewer.interview_guides.questions import Question
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message, MessageRole


_MAX_BATCH_SIZE = 8
_MAX_TRANSLATION_RETRIES = 3
_MAX_TOKENS = 100_000
_TEMPERATURE = 0


def _build_translation_batch_model(size: int) -> type[BaseModel]:
    if size < 1:
        raise ValueError("size must be >= 1")

    fields: dict[str, Any] = {
        f"t{i}": (
            str,
            ...,
        )
        for i in range(size)
    }
    return cast(type[BaseModel], create_model(f"TranslationBatch_{size}", **fields))


async def _translate_batch(
    sources: list[str],
    target_language: str,
    model: str,
    *,
    context: str | None = None,
) -> list[str]:
    if not sources:
        return []

    numbered = "\n".join(f"{i + 1}. {source!r}" for i, source in enumerate(sources))
    key_guide = "\n".join(
        f"t{i}: translation of item {i + 1}" for i in range(len(sources))
    )
    response_model = _build_translation_batch_model(len(sources))

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                f"Translate each numbered source string into {target_language}. "
                "Preserve meaning and tone. "
                "Return JSON only, with exactly one field per expected key. "
                "Do not add explanations."
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                (f"# Context\n{context}\n\n" if context else "")
                + "# Sources\n"
                + numbered
                + "\n\n# Output keys\n"
                + key_guide
            ),
        ),
    ]

    for attempt in range(_MAX_TRANSLATION_RETRIES):
        try:
            batch = await chat(
                messages=messages,
                model=model,
                response_format=response_model,
                max_tokens=_MAX_TOKENS,
                temperature=_TEMPERATURE,
                reasoning_effort="low",
                # extra_body={"provider": "DeepInfra"},
            )
            return [getattr(batch, f"t{i}") for i in range(len(sources))]
        except (ValidationError, ValueError):
            if attempt == _MAX_TRANSLATION_RETRIES - 1:
                raise
            await asyncio.sleep(0.35 * (2**attempt))

    raise RuntimeError("translation retries exhausted")


async def _translate_strings(
    strings: dict[str, str],
    target_language: str,
    model: str,
    *,
    context: str | None = None,
) -> dict[str, str]:
    """Translate a keyed map of strings.

    Uses positional matching (numbered list in, list of translations out) so
    the schema stays simple and compatible with strict structured outputs.
    """
    if not strings:
        return {}

    keys = list(strings.keys())
    sources = [strings[k] for k in keys]
    translated_values: list[str] = []

    for start in range(0, len(sources), _MAX_BATCH_SIZE):
        chunk = sources[start : start + _MAX_BATCH_SIZE]
        batch_context = (
            (f"{context}\n" if context else "")
            + f"Batch item range: {start + 1}-{start + len(chunk)} of {len(sources)} total strings."
        )
        translated_values.extend(
            await _translate_batch(
                chunk,
                target_language=target_language,
                model=model,
                context=batch_context,
            )
        )

    return {
        key: translated_values[i] if i < len(translated_values) else strings[key]
        for i, key in enumerate(keys)
    }


def _collect_top_level_strings(guide: InterviewGuide) -> dict[str, str]:
    out: dict[str, str] = {}

    for field in ("introduction", "outro", "alt_outro"):
        val = getattr(guide, field)
        if isinstance(val, str):
            out[field] = val
        elif isinstance(val, InterviewMessage):
            out[f"{field}.message"] = val.message

    for ti, tm in enumerate(guide.timed_messages or []):
        out[f"timed_messages[{ti}].message"] = tm.message

    return out


def _collect_question_strings(q: Question) -> dict[str, str]:
    out: dict[str, str] = {"main_question": q.main_question}
    if q.description:
        out["description"] = q.description
    for pi, p in enumerate(q.probes or []):
        out[f"probes[{pi}]"] = p
    for ai, a in enumerate(q.alternative_main_questions or []):
        out[f"alternative_main_questions[{ai}]"] = a
    return out


def _apply_top_level_strings(guide: InterviewGuide, t: dict[str, str]) -> None:
    for field in ("introduction", "outro", "alt_outro"):
        val = getattr(guide, field)
        if isinstance(val, str) and field in t:
            setattr(guide, field, t[field])
        elif isinstance(val, InterviewMessage) and f"{field}.message" in t:
            val.message = t[f"{field}.message"]

    for ti, tm in enumerate(guide.timed_messages or []):
        key = f"timed_messages[{ti}].message"
        if key in t:
            tm.message = t[key]


def _apply_question_strings(q: Question, t: dict[str, str]) -> None:
    if "main_question" in t:
        q.main_question = t["main_question"]
    if "description" in t:
        q.description = t["description"]
    if q.probes:
        q.probes = [
            t[f"probes[{pi}]"] if f"probes[{pi}]" in t else p
            for pi, p in enumerate(q.probes)
        ]
    if q.alternative_main_questions:
        q.alternative_main_questions = [
            t[f"alternative_main_questions[{ai}]"]
            if f"alternative_main_questions[{ai}]" in t
            else a
            for ai, a in enumerate(q.alternative_main_questions)
        ]


async def translate_interview_guide(
    guide: InterviewGuide,
    target_language: str,
    model: str,
) -> InterviewGuide:
    """Return a copy of the interview guide with user-facing text translated.

    Top-level fields and each section are translated in parallel LLM calls to
    keep each call short and avoid the model drifting back to the source
    language on long outputs. Non-user-facing fields (framing, configs,
    indices, variables, conditions) are left untouched.
    """
    translated = guide.model_copy(deep=True)

    base_context = (
        "These strings come from an interview guide used by an AI interviewer "
        "in a social science qualitative interview."
        + (f"\nFraming: {guide.framing}" if guide.framing else "")
    )

    tasks: list = []
    appliers: list = []

    top_strings = _collect_top_level_strings(translated)
    if top_strings:
        tasks.append(
            _translate_strings(
                top_strings,
                target_language=target_language,
                model=model,
                context=base_context
                + "\nThese are top-level messages (introduction, outro, timed messages).",
            )
        )
        appliers.append(lambda r: _apply_top_level_strings(translated, r))

    section_descriptions: dict[str, str] = {
        f"sections[{si}].description": section.description
        for si, section in enumerate(translated.question_sections)
    }
    if section_descriptions:

        def _apply_section_descriptions(r: dict[str, str]) -> None:
            for si, section in enumerate(translated.question_sections):
                key = f"sections[{si}].description"
                if key in r:
                    section.description = r[key]

        tasks.append(
            _translate_strings(
                section_descriptions,
                target_language=target_language,
                model=model,
                context=base_context + "\nThese are section descriptions.",
            )
        )
        appliers.append(_apply_section_descriptions)

    for si, section in enumerate(translated.question_sections):
        for qi, question in enumerate(section.questions):
            q_strings = _collect_question_strings(question)
            if not q_strings:
                continue
            tasks.append(
                _translate_strings(
                    q_strings,
                    target_language=target_language,
                    model=model,
                    context=(
                        base_context
                        + f"\nThese strings belong to question {qi + 1} in "
                        f"section {si + 1} of {len(translated.question_sections)}. "
                        f"Section description: {section.description!r}."
                    ),
                )
            )
            appliers.append(lambda r, q=question: _apply_question_strings(q, r))

    results = await asyncio.gather(*tasks)
    for applier, result in zip(appliers, results):
        applier(result)

    return translated


async def translate_consent(
    consent: Consent,
    target_language: str,
    model: str,
) -> Consent:
    strings = {"title": consent.title, "text": consent.text}
    result = await _translate_strings(
        strings,
        target_language=target_language,
        model=model,
        context="Consent text shown to an interviewee before starting an interview.",
    )
    return consent.model_copy(update=result)


async def translate_welcome(
    welcome: Welcome,
    target_language: str,
    model: str,
) -> Welcome:
    strings = {"title": welcome.title, "text": welcome.text}
    result = await _translate_strings(
        strings,
        target_language=target_language,
        model=model,
        context="Welcome screen text shown to an interviewee before starting an interview.",
    )
    return welcome.model_copy(update=result)


if __name__ == "__main__":
    raise NotImplementedError("CLI not implemented because of async")
