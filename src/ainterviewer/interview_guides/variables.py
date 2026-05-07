from typing import Any

from jinja2 import Environment, meta

BUILTIN_VARIABLES: frozenset[str] = frozenset({"project_id", "interview_id"})


def extract_placeholders(text: str) -> set[str]:
    """Return the set of {{ name }} placeholders referenced in `text`."""
    if not text:
        return set()
    ast = Environment().parse(text)
    return meta.find_undeclared_variables(ast)


def fill_variables_in_message(
    text: str,
    referable_values: dict[str, Any],
) -> str:
    """Fill in the variables in the main question and probes"""
    env = Environment()
    ast = env.parse(text)
    placeholders = meta.find_undeclared_variables(ast)

    values = {
        name: referable_values[name]
        for name in placeholders
        if name in referable_values
    }

    return env.from_string(text).render(**values)
