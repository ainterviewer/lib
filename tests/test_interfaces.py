from typing import Annotated, Union

from pydantic import Field, TypeAdapter

from ainterviewer.interfaces import (
    OutgoingHistoryMessage,
    OutgoingMessage,
    _OutgoingData,
)
from ainterviewer.types import MessageRole


def test_outgoing_types_are_distinct():
    # Nothing parses outgoing data as a union, so build one here: pydantic
    # refuses it if a subclass lacks a `Literal` `type` or reuses another's.
    subclasses = tuple(_OutgoingData.__subclasses__())
    adapter = TypeAdapter(
        Annotated[Union[subclasses], Field(discriminator="type")]  # noqa: UP007  # ty:ignore[invalid-type-form]
    )

    for cls in (OutgoingHistoryMessage, OutgoingMessage):
        data = cls(content="x", role=MessageRole.USER, message_id=1).model_dump()
        assert type(adapter.validate_python(data)) is cls
