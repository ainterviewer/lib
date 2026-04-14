import json

from pydantic import BaseModel, Field

from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewMessage,
)
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message, MessageRole


class TranslationItem(BaseModel):
    key: str = Field(description="The exact key from the input map.")
    translated: str = Field(description="The translated text.")


class TranslationBatch(BaseModel):
    items: list[TranslationItem]


async def _translate_strings(
    strings: dict[str, str],
    target_language: str,
    model: str,
    *,
    context: str | None = None,
) -> dict[str, str]:
    """Translate a keyed map of strings, preserving keys and placeholders."""
    if not strings:
        return {}

    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                f"Translate the provided user-facing strings into {target_language}. "
                "Preserve meaning and tone. "
                "Preserve any Jinja2 templating placeholders (e.g. {{ uuid }}) "
                "and literal '{}' placeholders exactly as they appear. "
                "Return one item per input key, reusing the exact same keys."
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                (f"# Context\n{context}\n\n" if context else "")
                + "# Strings to translate\n"
                + f"```json\n{json.dumps(strings, ensure_ascii=False, indent=2)}\n```"
            ),
        ),
    ]

    batch = await chat(
        messages=messages,
        model=model,
        response_format=TranslationBatch,
    )

    translated = {item.key: item.translated for item in batch.items}
    return {key: translated.get(key, value) for key, value in strings.items()}


def _collect_guide_strings(guide: InterviewGuide) -> dict[str, str]:
    out: dict[str, str] = {}

    for field in ("introduction", "outro", "alt_outro"):
        val = getattr(guide, field)
        if isinstance(val, str):
            out[field] = val
        elif isinstance(val, InterviewMessage):
            out[f"{field}.message"] = val.message

    for si, section in enumerate(guide.question_sections):
        out[f"sections[{si}].description"] = section.description
        for qi, q in enumerate(section.questions):
            base = f"sections[{si}].questions[{qi}]"
            out[f"{base}.main_question"] = q.main_question
            if q.description:
                out[f"{base}.description"] = q.description
            for pi, p in enumerate(q.probes or []):
                out[f"{base}.probes[{pi}]"] = p
            for ai, a in enumerate(getattr(q, "alternative_main_questions", None) or []):
                out[f"{base}.alternative_main_questions[{ai}]"] = a

    for ti, tm in enumerate(guide.timed_messages or []):
        out[f"timed_messages[{ti}].message"] = tm.message

    return out


def _apply_guide_strings(guide: InterviewGuide, t: dict[str, str]) -> None:
    for field in ("introduction", "outro", "alt_outro"):
        val = getattr(guide, field)
        if isinstance(val, str) and field in t:
            setattr(guide, field, t[field])
        elif isinstance(val, InterviewMessage) and f"{field}.message" in t:
            val.message = t[f"{field}.message"]

    for si, section in enumerate(guide.question_sections):
        key = f"sections[{si}].description"
        if key in t:
            section.description = t[key]
        for qi, q in enumerate(section.questions):
            base = f"sections[{si}].questions[{qi}]"
            if (k := f"{base}.main_question") in t:
                q.main_question = t[k]
            if (k := f"{base}.description") in t:
                q.description = t[k]
            if q.probes:
                q.probes = [
                    t.get(f"{base}.probes[{pi}]", p) for pi, p in enumerate(q.probes)
                ]
            alts = getattr(q, "alternative_main_questions", None)
            if alts:
                q.alternative_main_questions = [
                    t.get(f"{base}.alternative_main_questions[{ai}]", a)
                    for ai, a in enumerate(alts)
                ]

    for ti, tm in enumerate(guide.timed_messages or []):
        key = f"timed_messages[{ti}].message"
        if key in t:
            tm.message = t[key]


async def translate_interview_guide(
    guide: InterviewGuide,
    target_language: str,
    model: str,
) -> InterviewGuide:
    """Return a copy of the interview guide with user-facing text translated.

    Non-user-facing fields (framing, configs, indices, variables, conditions)
    are left untouched.
    """
    translated = guide.model_copy(deep=True)
    strings = _collect_guide_strings(translated)
    result = await _translate_strings(
        strings,
        target_language=target_language,
        model=model,
        context=(
            "These strings come from an interview guide used by an AI interviewer "
            "in a social science qualitative interview."
            + (f"\nFraming: {guide.framing}" if guide.framing else "")
        ),
    )
    _apply_guide_strings(translated, result)
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
