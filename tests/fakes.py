from collections.abc import Awaitable, Callable

from ainterviewer.events import (
    InputEvent,
    InputRequested,
    MessageEmitted,
    OutboundEvent,
    UserMessageRecorded,
)
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.types import MessageRole


class FakeHost:
    """An `EngineHost` that keeps every event it is sent and answers each
    input request with `answer`. Without `answer`, any request fails the test.
    """

    def __init__(
        self,
        answer: Callable[[InputRequested], Awaitable[InputEvent]] | None = None,
    ):
        self.answer = answer
        self.events: list[OutboundEvent] = []
        self.requests: list[InputRequested] = []

    async def emit(self, event: OutboundEvent) -> None:
        self.events.append(event)

    async def next_input(self, request: InputRequested) -> InputEvent:
        self.requests.append(request)
        assert self.answer is not None, f"Unexpected input request: {request}"
        return await self.answer(request)

    def of[T](self, kind: type[T]) -> list[T]:
        """The events of one kind, in the order they were emitted."""
        return [event for event in self.events if isinstance(event, kind)]


def records(events: list[OutboundEvent]) -> list[HistoryRecord]:
    """What a host that stores the messages it is sent hands back on resume or
    backfill."""
    out = []
    for event in events:
        match event:
            case MessageEmitted():
                out.append(
                    HistoryRecord(
                        role=MessageRole.ASSISTANT,
                        **event.model_dump(include=set(HistoryRecord.model_fields)),
                    )
                )
            case UserMessageRecorded():
                out.append(
                    HistoryRecord(
                        role=MessageRole.USER,
                        **event.model_dump(include=set(HistoryRecord.model_fields)),
                    )
                )
    return out
