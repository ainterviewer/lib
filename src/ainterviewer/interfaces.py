from typing import Protocol

from ainterviewer.events import InputEvent, InputRequested, OutboundEvent


class EngineHost(Protocol):
    """Everything `AInterviewer` needs from whatever runs it."""

    async def emit(self, event: OutboundEvent) -> None:
        """Takes one output event. May be called concurrently -- background
        work such as the security check emits while the interview runs -- so an
        implementation writing to a single stream must serialise. Events from
        one caller arrive in the order it emitted them."""
        ...

    async def next_input(self, request: InputRequested) -> InputEvent:
        """Puts `request` to the respondent and waits for the answer. The host
        delivers the request itself; the engine does not emit it as well."""
        ...
