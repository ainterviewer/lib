import random
from typing import Any, Protocol, TypeVar

from jinja2 import Template


class Shuffleable(Protocol):
    shuffle: bool


T = TypeVar("T", bound=Shuffleable)


def shuffle_items(items: list[T]) -> list[T]:
    """Shuffle a list of items, keeping the order of items that have shuffle=False"""

    # Get indices and items where shuffle is True
    shuffleable_indices = [i for i, item in enumerate(items) if item.shuffle]
    shuffleable_items = [items[i] for i in shuffleable_indices]

    # Shuffle the selected items
    random.shuffle(shuffleable_items)

    # Create new list, replacing shuffleable items with shuffled ones
    result = list(items)  # Make a copy
    for new_item, idx in zip(shuffleable_items, shuffleable_indices):
        result[idx] = new_item

    return result


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
