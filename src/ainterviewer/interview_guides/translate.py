import asyncio
from collections import defaultdict
from typing import Any, cast

from pydantic import BaseModel, ValidationError, create_model

from ainterviewer.interview_guides.localized import (
    LocalizedConsent,
    LocalizedInterviewGuide,
    LocalizedText,
    LocalizedWelcome,
)
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message, MessageRole
from ainterviewer.types import LanguageCode
from ainterviewer.utils import get_language_dict

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


_CONTEXT = (
    "These strings come from an interview guide used by an AI interviewer "
    "in a social science qualitative interview."
)


def _group(path: str) -> str:
    """Leaves are translated in one call per section and one for the rest, so
    each call sees strings that belong together."""
    head = path.split(".", 1)[0]
    return head if head.startswith("question_sections[") else "top"


async def translate_missing[
    T: (LocalizedInterviewGuide, LocalizedConsent, LocalizedWelcome)
](
    content: T,
    *,
    target: LanguageCode,
    source: LanguageCode,
    model: str,
) -> T:
    """Return a copy of `content` with every leaf written in `source` but
    missing in `target` translated into it. Leaves `target` already has are
    left alone, so this both seeds a new language and fills in what was added
    since.

    Only text is translated: the structure is shared by every language, and a
    condition matching a choice names the option by id, so nothing else has to
    follow.
    """
    translated = content.model_copy(deep=True)
    target_name = get_language_dict(target)["name"]

    groups: dict[str, dict[str, LocalizedText]] = defaultdict(dict)
    for path, text in translated.texts():
        if not text.get(target) and text.get(source):
            groups[_group(path)][path] = text
    if not groups:
        return translated

    match translated:
        case LocalizedInterviewGuide():
            framing = translated.framing.get(source)
            context = _CONTEXT + (f"\n\n# Framing\n{framing}" if framing else "")
            n_sections = len(translated.question_sections)
        case LocalizedConsent():
            context = (
                "Consent text shown to an interviewee before starting an interview."
            )
            n_sections = 0
        case LocalizedWelcome():
            context = "Welcome screen text shown to an interviewee before starting an interview."
            n_sections = 0

    def group_context(group: str) -> str:
        if group == "top":
            return context
        n = int(group.removeprefix("question_sections[").removesuffix("]"))
        return (
            f"{context}\nThese strings belong to section {n + 1} of {n_sections} "
            "(its description and questions)."
        )

    results = await asyncio.gather(
        *(
            _translate_strings(
                {path: text[source] for path, text in leaves.items()},
                target_language=target_name,
                model=model,
                context=group_context(group),
            )
            for group, leaves in groups.items()
        )
    )
    for leaves, result in zip(groups.values(), results):
        for path, text in leaves.items():
            text[target] = result[path]

    # A translation can break what validation checks (a `{}` placeholder, a
    # template variable), so the result is validated like any other write.
    return type(translated).model_validate(translated.model_dump())


if __name__ == "__main__":
    raise NotImplementedError("CLI not implemented because of async")
