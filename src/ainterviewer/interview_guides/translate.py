import asyncio
from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model

from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewMessage,
)
from ainterviewer.interview_guides.questions import Question
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import (
    CheckboxItem,
    LikertItem,
    RadioItem,
    SliderItem,
)
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


def _collect_section_strings(section: QuestionSection[Question]) -> dict[str, str]:
    out: dict[str, str] = {"description": section.description}
    for qi, q in enumerate(section.questions):
        base = f"questions[{qi}]"
        out[f"{base}.main_question"] = q.main_question
        if q.description:
            out[f"{base}.description"] = q.description
        for pi, p in enumerate(q.probes or []):
            out[f"{base}.probes[{pi}]"] = p
        for ai, a in enumerate(q.alternative_main_questions or []):
            out[f"{base}.alternative_main_questions[{ai}]"] = a
        if isinstance(q.survey_item, (RadioItem, CheckboxItem, LikertItem)):
            for oi, option in enumerate(q.survey_item.options):
                out[f"{base}.survey_item.options[{oi}]"] = option
        elif isinstance(q.survey_item, SliderItem):
            if q.survey_item.min_label:
                out[f"{base}.survey_item.min_label"] = q.survey_item.min_label
            if q.survey_item.max_label:
                out[f"{base}.survey_item.max_label"] = q.survey_item.max_label
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


def _apply_section_strings(
    section: QuestionSection[Question], t: dict[str, str]
) -> None:
    if "description" in t:
        section.description = t["description"]
    for qi, q in enumerate(section.questions):
        base = f"questions[{qi}]"
        if (k := f"{base}.main_question") in t:
            q.main_question = t[k]
        if (k := f"{base}.description") in t:
            q.description = t[k]
        if q.probes:
            q.probes = [
                t[f"{base}.probes[{pi}]"] if f"{base}.probes[{pi}]" in t else p
                for pi, p in enumerate(q.probes)
            ]
        if isinstance(q.survey_item, (RadioItem, CheckboxItem, LikertItem)):
            q.survey_item.options = [
                t[f"{base}.survey_item.options[{oi}]"]
                if f"{base}.survey_item.options[{oi}]" in t
                else option
                for oi, option in enumerate(q.survey_item.options)
            ]
        elif isinstance(q.survey_item, SliderItem):
            if (k := f"{base}.survey_item.min_label") in t:
                q.survey_item.min_label = t[k]
            if (k := f"{base}.survey_item.max_label") in t:
                q.survey_item.max_label = t[k]
        if q.alternative_main_questions:
            q.alternative_main_questions = [
                t[f"{base}.alternative_main_questions[{ai}]"]
                if f"{base}.alternative_main_questions[{ai}]" in t
                else a
                for ai, a in enumerate(q.alternative_main_questions)
            ]


async def translate_interview_guide(
    guide: InterviewGuide,
    target_language: str,
    model: str,
) -> InterviewGuide:
    """Return a copy of the interview guide with user-facing text translated.

    Runs one LLM call for top-level messages and one call per section
    (description + all its questions) in parallel. Non-user-facing fields
    (framing, configs, indices, variables, conditions) are left untouched;
    framing is passed as context so translations stay consistent in tone.
    """
    translated = guide.model_copy(deep=True)

    base_context = (
        "These strings come from an interview guide used by an AI interviewer "
        "in a social science qualitative interview."
    )
    framing_context = f"\n\n# Framing\n{guide.framing}" if guide.framing else ""

    tasks: list = []
    appliers: list = []

    top_strings = _collect_top_level_strings(translated)
    if top_strings:
        tasks.append(
            _translate_strings(
                top_strings,
                target_language=target_language,
                model=model,
                context=(
                    base_context
                    + "\nThese are top-level messages (introduction, outro, timed messages)."
                    + framing_context
                ),
            )
        )
        appliers.append(lambda r: _apply_top_level_strings(translated, r))

    for si, section in enumerate(translated.question_sections):
        strings = _collect_section_strings(section)
        tasks.append(
            _translate_strings(
                strings,
                target_language=target_language,
                model=model,
                context=(
                    base_context + f"\nThese strings belong to section {si + 1} of "
                    f"{len(translated.question_sections)} (its description and questions)."
                    + framing_context
                ),
            )
        )
        appliers.append(lambda r, s=section: _apply_section_strings(s, r))

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
