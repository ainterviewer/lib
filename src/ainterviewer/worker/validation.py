"""Whether this runtime version can run what it is given, exactly as given.

The models are lenient where it suits a running interview -- several ignore
fields they do not know -- but a host freezing content for this version needs
the opposite: anything this version would not act on must be named, or a
feature a researcher turned on is silently dropped. So validation here is
strict everywhere, without changing the models themselves: input is validated
as usual, then walked next to what it validated into, and every key the models
let pass without a field to take it is reported.
"""

from collections.abc import Iterator
from typing import Any

from jinja2 import Environment, PackageLoader, TemplateError, TemplateNotFound
from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from ainterviewer.agents.config import AgentConfigs
from ainterviewer.config import InterviewConfig
from ainterviewer.interview_guides import InterviewGuide

TEMPLATES = PackageLoader("ainterviewer.agents.prompts.templates", "EN")


class Problem(BaseModel):
    """Something at `path` this version cannot accept."""

    # Dotted, from the top of the request: `guide.question_sections.0.foo`.
    path: str
    message: str


def _join(path: str, key: str | int) -> str:
    return f"{path}.{key}" if path else str(key)


def _names(field_name: str, field: FieldInfo) -> set[str]:
    names = {field_name}
    if isinstance(field.alias, str):
        names.add(field.alias)
    return names


def unknown_paths(data: Any, value: Any, path: str = "") -> Iterator[str]:
    """The paths in `data` that validating it into `value` ignored."""
    if isinstance(value, BaseModel) and isinstance(data, dict):
        fields = {
            name: field_name
            for field_name, field in type(value).model_fields.items()
            for name in _names(field_name, field)
        }
        for key, item in data.items():
            if key not in fields:
                yield _join(path, key)
            else:
                yield from unknown_paths(
                    item, getattr(value, fields[key]), _join(path, key)
                )

    elif isinstance(value, list | tuple) and isinstance(data, list | tuple):
        for index, (item, validated) in enumerate(zip(data, value, strict=False)):
            yield from unknown_paths(item, validated, _join(path, index))

    elif isinstance(value, dict) and isinstance(data, dict):
        for key, item in data.items():
            if key in value:
                yield from unknown_paths(item, value[key], _join(path, key))


def _path_in(data: Any, loc: tuple[str | int, ...], path: str) -> str:
    """Pydantic's location of an error, as a path into `data`.

    A location also names the union member it was validated against, which
    is no key in the data; those parts are dropped. A missing field is the
    exception, as the last part of its location.
    """
    current = data

    for position, part in enumerate(loc):
        is_key = isinstance(current, dict) and part in current
        is_index = (
            isinstance(current, list)
            and isinstance(part, int)
            and 0 <= part < len(current)
        )
        is_missing = isinstance(current, dict) and position == len(loc) - 1

        if is_key or is_index:
            current = current[part]
        elif not is_missing:
            continue

        path = _join(path, part)

    return path


def check[M: BaseModel](model: type[M], data: Any, path: str) -> list[Problem]:
    """The problems with `data` as a `model`, strictly."""
    try:
        value = model.model_validate(data)
    except ValidationError as error:
        return [
            Problem(path=_path_in(data, detail["loc"], path), message=detail["msg"])
            for detail in error.errors()
        ]

    return [
        Problem(path=unknown, message="Not supported by this version")
        for unknown in unknown_paths(data, value, path)
    ]


def check_prompt_overrides(overrides: Any, path: str) -> list[Problem]:
    """Overrides must replace templates this version has, with templates it
    can compile."""
    if not isinstance(overrides, dict) or not all(
        isinstance(name, str) and isinstance(source, str)
        for name, source in overrides.items()
    ):
        return [Problem(path=path, message="Must map template names to templates")]

    environment = Environment(loader=TEMPLATES)
    known = {name for name in TEMPLATES.list_templates() if name.endswith(".jinja")}
    problems = []

    for name, source in overrides.items():
        if name not in known:
            problems.append(
                Problem(
                    path=_join(path, name),
                    message="No such template in this version",
                )
            )
            continue
        try:
            environment.parse(source)
        except (TemplateError, TemplateNotFound) as error:
            problems.append(Problem(path=_join(path, name), message=str(error)))

    return problems


def validate_payload(payload: dict[str, Any]) -> list[Problem]:
    """Every problem with a payload's parts, each of which is optional."""
    problems: list[Problem] = []

    for key, model in (
        ("guide", InterviewGuide),
        ("config", InterviewConfig),
        ("agent_configs", AgentConfigs),
    ):
        if key in payload:
            problems.extend(check(model, payload[key], key))

    if "prompt_overrides" in payload:
        problems.extend(
            check_prompt_overrides(payload["prompt_overrides"], "prompt_overrides")
        )

    problems.extend(
        Problem(path=key, message="Not supported by this version")
        for key in payload
        if key not in {"guide", "config", "agent_configs", "prompt_overrides"}
    )

    return problems
