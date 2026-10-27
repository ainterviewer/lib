"""Compatibility between library versions: converting contract JSON between
schema versions.

The contract between a host and its engines -- the events, `HistoryRecord`,
`EmbeddingChunk`, and the models an engine is handed (`InterviewGuide`,
`InterviewConfig`, `AgentConfigs`) -- is versioned as a whole by
`SCHEMA_VERSION`. A host runs on one library version but talks to engines on
older ones, so it converts in both directions:

- **Up**, for what comes from older versions: events from engines on an older
  runtime version, and its own stored copies after a library upgrade.
- **Down**, for what goes to older versions: content serialised for an engine
  on an older runtime version, and the events it sends one.

A payload frozen for an engine is never converted again; it only changes by
being rebuilt from the host's own, upgraded copy.

Making a breaking change to the contract:

1. Bump `SCHEMA_VERSION`.
2. For every kind the change touches, register both directions of the change
   from the previous version, N: `@upgrades(kind, N)` and `@downgrades(kind,
   N)`. A step takes the JSON of one value of that kind and returns it in the
   other version's shape. A part stored on its own (`SURVEY_ITEM`, `IMAGE`)
   is registered as well as every kind that contains it, whose steps call the
   part's: a guide holds survey items, and so do `MessageEmitted`,
   `GuideUpdated` and `Init`.
3. A downgrade that cannot express what it is given -- a feature the older
   version lacks, a value whose meaning changed -- raises `Unsupported`
   naming the paths, never drops them. The host turns that into "upgrade to
   the newer runtime version".
4. Test both directions on a real payload of the previous version: a lossless
   change must round-trip, `downgrade(upgrade(x)) == x`.

The steps for the change from N are only needed while an engine on schema N or
older is still running. Once none is, delete them and raise
`MIN_SCHEMA_VERSION` past N.
"""

import copy
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any, Self

# Bump on a breaking change to any part of the contract, together with both
# steps for each kind it touches.
SCHEMA_VERSION = 1
# The oldest version still converted from or to.
MIN_SCHEMA_VERSION = 1

type Json = dict[str, Any]
type Step = Callable[[Json], Json]


class Kind(StrEnum):
    EVENT = "event"
    INTERVIEW_GUIDE = "interview_guide"
    INTERVIEW_CONFIG = "interview_config"
    AGENT_CONFIGS = "agent_configs"
    HISTORY_RECORD = "history_record"
    EMBEDDING_CHUNK = "embedding_chunk"
    # Guide parts a host stores on their own as well, outside any guide or
    # event: the survey item and image a message was sent with.
    SURVEY_ITEM = "survey_item"
    IMAGE = "image"


class Unsupported(ValueError):
    """A value uses something the version it is downgraded to cannot express."""

    def __init__(self, paths: Iterable[str]):
        self.paths: list[str] = list(paths)
        super().__init__(f"Not supported: {', '.join(self.paths)}")

    def at(self, prefix: str) -> Self:
        """The same refusal, for a value found at `prefix` in a larger one."""
        return type(self)(f"{prefix}.{path}" if path else prefix for path in self.paths)


# (kind, N) -> the step between versions N and N + 1, in each direction.
UPGRADES: dict[tuple[Kind, int], Step] = {}
DOWNGRADES: dict[tuple[Kind, int], Step] = {}


def _register(
    steps: dict[tuple[Kind, int], Step], kind: Kind, version: int
) -> Callable[[Step], Step]:
    def register(step: Step) -> Step:
        if (kind, version) in steps:
            raise ValueError(f"A step for {kind} at version {version} exists")
        steps[(kind, version)] = step
        return step

    return register


def upgrades(kind: Kind, from_version: int) -> Callable[[Step], Step]:
    """Registers the decorated function as the upgrade of `kind` from
    `from_version` to the version after it."""
    return _register(UPGRADES, kind, from_version)


def downgrades(kind: Kind, to_version: int) -> Callable[[Step], Step]:
    """Registers the decorated function as the downgrade of `kind` from the
    version after `to_version` to `to_version`."""
    return _register(DOWNGRADES, kind, to_version)


def _check(kind: Kind, version: int) -> None:
    if not MIN_SCHEMA_VERSION <= version <= SCHEMA_VERSION:
        raise ValueError(
            f"Cannot convert {kind} at schema version {version}; this library "
            f"converts {MIN_SCHEMA_VERSION} to {SCHEMA_VERSION}"
        )


def upgrade(kind: Kind, data: Json, from_version: int) -> Json:
    """`data`, a `kind` written under `from_version`, in the current shape.

    Leaves `data` itself alone.
    """
    _check(kind, from_version)
    data = copy.deepcopy(data)

    for version in range(from_version, SCHEMA_VERSION):
        # A version that did not touch this kind has nothing to do for it.
        if (step := UPGRADES.get((kind, version))) is not None:
            data = step(data)

    return data


def downgrade(kind: Kind, data: Json, to_version: int) -> Json:
    """`data`, a `kind` in the current shape, as `to_version` writes it.

    Leaves `data` itself alone. Raises `Unsupported` when `data` uses
    something `to_version` cannot express.
    """
    _check(kind, to_version)
    data = copy.deepcopy(data)

    for version in reversed(range(to_version, SCHEMA_VERSION)):
        if (step := DOWNGRADES.get((kind, version))) is not None:
            data = step(data)

    return data


def upgrade_event(data: Json) -> Json:
    """An event's JSON in the current shape, upgraded from the
    `schema_version` it carries."""
    upgraded = upgrade(Kind.EVENT, data, from_version=data["schema_version"])
    upgraded["schema_version"] = SCHEMA_VERSION
    return upgraded


def downgrade_event(data: Json, to_version: int) -> Json:
    """A current event's JSON as an engine on `to_version` reads it."""
    downgraded = downgrade(Kind.EVENT, data, to_version=to_version)
    downgraded["schema_version"] = to_version
    return downgraded
