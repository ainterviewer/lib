import asyncio

from fastapi import WebSocket
from pydantic import ValidationError

from ainterviewer.compat import SCHEMA_VERSION
from ainterviewer.events import (
    InputEvent,
    InputRequested,
    OutboundEvent,
    input_event_adapter,
)


class ProtocolError(Exception):
    """The host sent something this engine cannot read."""


class WebsocketHost:
    """An `EngineHost` at the other end of a websocket: every event goes out as
    one JSON text frame, and inputs come back the same way."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._sending = asyncio.Lock()

    async def emit(self, event: OutboundEvent) -> None:
        await self._send(event.model_dump_json())

    async def next_input(self, request: InputRequested) -> InputEvent:
        await self.emit(request)

        # Only the interview loop waits for input, so frames are never read
        # from two places at once.
        text = await self.websocket.receive_text()

        try:
            answer = input_event_adapter.validate_json(text)
        except ValidationError as error:
            raise ProtocolError(f"Unreadable input: {error}") from error

        if answer.schema_version != SCHEMA_VERSION:
            raise ProtocolError(
                f"Input at schema version {answer.schema_version}, "
                f"this engine reads {SCHEMA_VERSION}"
            )

        return answer

    async def _send(self, text: str) -> None:
        # Shielded, lock included: background work such as the security check
        # can be cancelled mid-send, and a frame must still go out whole and
        # alone. The caller is cancelled; the send it started is not.
        await asyncio.shield(self._send_alone(text))

    async def _send_alone(self, text: str) -> None:
        async with self._sending:
            await self.websocket.send_text(text)
