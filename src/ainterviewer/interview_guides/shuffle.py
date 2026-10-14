import random
from collections.abc import Collection
from typing import Protocol

# Enough to find a valid order whenever constraints leave a reasonable share of
# orders valid, while bounding the cost of a guide whose constraints leave few.
SHUFFLE_ATTEMPTS = 1000


class Shuffleable(Protocol):
    shuffle: bool


def shuffle_items[T: Shuffleable](
    items: list[T],
    must_precede: Collection[tuple[int, int]] = (),
    attempts: int = SHUFFLE_ATTEMPTS,
) -> list[T]:
    """Shuffle a list of items, keeping the order of items that have shuffle=False

    `must_precede` holds pairs `(a, b)` of indices into `items`: wherever the
    shuffle puts them, item `a` must end up before item `b`. An order breaking a
    pair is drawn again rather than repaired, so every valid order stays equally
    likely -- repairing would over-represent the orders repairs land on, which
    matters to anyone analysing order effects. If no valid order is drawn within
    `attempts`, the items keep their given order, which saving the guide checks
    is valid (`InterviewGuide.condition_order_errors`).
    """

    # Get indices of the items where shuffle is True
    shuffleable_indices = [i for i, item in enumerate(items) if item.shuffle]

    for _ in range(attempts):
        # Shuffle the selected indices
        shuffled = random.sample(shuffleable_indices, len(shuffleable_indices))

        # Create new order, replacing shuffleable items with shuffled ones
        order = list(range(len(items)))
        for new_index, slot in zip(shuffled, shuffleable_indices):
            order[slot] = new_index

        position = {index: n for n, index in enumerate(order)}
        if all(position[a] < position[b] for a, b in must_precede):
            return [items[index] for index in order]

    return list(items)
