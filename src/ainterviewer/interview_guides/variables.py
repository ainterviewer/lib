from typing import Any

from jinja2 import Template


def fill_variables_in_message(
    text: str,
    variables: list[str],
    referable_values: dict[str, Any],
) -> str:
    """Fill in the variables in the main question and probes"""
    values = {}
    for variable in variables:
        if value := referable_values.get(variable):
            values[variable] = value

    template = Template(text)
    return template.render(**values)
