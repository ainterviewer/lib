"""The runtime server: one interview engine, at one library version, for a host
to run interviews on.

Only `/run` keeps state or calls a language model. Everything else is a pure
function of its request, answered with this version's own models and rules.
"""

import logging
import os
import re
from typing import Any

from any_llm.exceptions import (
    AnyLLMError,
    GatewayTimeoutError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import UUID4, BaseModel, ValidationError

from ainterviewer import __version__
from ainterviewer.agents.config import AgentConfigs
from ainterviewer.compat import SCHEMA_VERSION
from ainterviewer.config import InterviewConfig
from ainterviewer.embedding import (
    CHUNK_FORMAT_VERSION,
    EmbeddingChunk,
    chunks_for_records,
)
from ainterviewer.events import Error, ErrorCode, Init
from ainterviewer.exceptions import UnexpectedInput
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides import InterviewGuide
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.types import LanguageCode
from ainterviewer.worker.host import ProtocolError, WebsocketHost
from ainterviewer.worker.validation import Problem, check, validate_payload

# Provider failures arrive as any-llm's own exception types, classified below,
# rather than as whatever each provider's SDK raises.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

logger = logging.getLogger(__name__)


def runtime_version(lib_version: str, name: str | None = None) -> str:
    """The runtime version a server on `lib_version` is.

    A release is the library's major and minor version, `0.9`. A preview -- a
    release candidate, a branch build -- is named by whoever builds it, and
    the name must start from the same version, `0.9-rc` or `0.9-feat-probing`,
    so it cannot pass for a different one.
    """
    release = ".".join(lib_version.split(".")[:2])

    if not name or name == release:
        return release

    label = name.removeprefix(f"{release}-")
    if label == name or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", label):
        raise ValueError(
            f"Runtime version {name!r} must be {release!r} or "
            f"{release!r} followed by a lowercase label, e.g. '{release}-rc'"
        )

    return name


# Set by the image build: a preview's name, and the commit it was built from.
RUNTIME_VERSION = runtime_version(__version__, os.environ.get("RUNTIME_VERSION"))
BUILD = os.environ.get("WORKER_BUILD") or None

# Worth retrying: the provider is down, overloaded or slow. Every other
# provider error is a refusal that retrying would only repeat.
_TRANSIENT_ERRORS = (
    RateLimitError,
    GatewayTimeoutError,
    UpstreamProviderError,
    ProviderError,
)

app = FastAPI(title="AInterviewer worker", version=__version__)


class Version(BaseModel):
    lib_version: str
    runtime_version: str
    # The commit the image was built from, when it was built by CI.
    build: str | None
    schema_version: int
    chunk_format_version: str


class Validation(BaseModel):
    ok: bool
    problems: list[Problem]


class ChunksRequest(BaseModel):
    guide: InterviewGuide
    history: list[HistoryRecord]
    project_id: UUID4
    interview_id: UUID4
    language: LanguageCode = "EN"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Version:
    return Version(
        lib_version=__version__,
        runtime_version=RUNTIME_VERSION,
        build=BUILD,
        schema_version=SCHEMA_VERSION,
        chunk_format_version=CHUNK_FORMAT_VERSION,
    )


@app.get("/schema")
def schema() -> dict[str, Any]:
    """The JSON Schema of everything a host serialises for this version."""
    return {
        "interview_guide": InterviewGuide.model_json_schema(),
        "interview_config": InterviewConfig.model_json_schema(),
        "agent_configs": AgentConfigs.model_json_schema(),
    }


@app.post("/validate")
def validate(payload: dict[str, Any]) -> Validation:
    """Whether this version runs `payload` -- any of `guide`, `config`,
    `agent_configs` and `prompt_overrides` -- exactly as given, naming every
    path it cannot accept."""
    problems = validate_payload(payload)
    return Validation(ok=not problems, problems=problems)


@app.post("/prepare")
def prepare(payload: dict[str, Any]) -> dict[str, Any]:
    """One interview's own guide: `guide`, shuffled by this version's rules."""
    if problems := check(InterviewGuide, payload.get("guide"), "guide"):
        raise HTTPException(
            status_code=422, detail=[problem.model_dump() for problem in problems]
        )

    guide = InterviewGuide.model_validate(payload["guide"])
    guide.shuffle()

    return {"guide": guide.model_dump(mode="json")}


@app.post("/chunks")
def chunks(request: ChunksRequest) -> list[EmbeddingChunk]:
    """Every chunk of a stored interview, derived as this version derives them
    live."""
    return chunks_for_records(
        request.guide,
        request.history,
        project_id=request.project_id,
        interview_id=request.interview_id,
        language=request.language,
    )


def classify(error: Exception) -> ErrorCode:
    if isinstance(error, ProtocolError | UnexpectedInput):
        return ErrorCode.PROTOCOL
    if isinstance(error, _TRANSIENT_ERRORS):
        return ErrorCode.INFERENCE_UNAVAILABLE
    if isinstance(error, AnyLLMError):
        return ErrorCode.INFERENCE_ERROR
    return ErrorCode.INTERNAL


async def read_init(websocket: WebSocket) -> Init:
    text = await websocket.receive_text()

    try:
        init = Init.model_validate_json(text)
    except ValidationError as error:
        raise ProtocolError(f"Expected an init event: {error}") from error

    if init.schema_version != SCHEMA_VERSION:
        raise ProtocolError(
            f"Init at schema version {init.schema_version}, "
            f"this engine reads {SCHEMA_VERSION}"
        )

    return init


@app.websocket("/run")
async def run(websocket: WebSocket) -> None:
    """One interview session: an `Init` in, then events out and inputs in
    until the interview ends or fails.

    A failure is reported as an `Error` event, never as a bare close, so the
    host can tell it from a dropped connection. The session can be resumed
    from what was recorded either way.
    """
    await websocket.accept()
    host = WebsocketHost(websocket)
    interview_id = None

    try:
        init = await read_init(websocket)
        interview_id = init.interview_id
        interviewer = AInterviewer.from_init(host, init)
        await interviewer.interview(interview_history=init.history)
    except WebSocketDisconnect:
        # The host ended the session; it records that itself.
        return
    except Exception as error:
        code = classify(error)

        if code == ErrorCode.INTERNAL:
            logger.exception("Interview %s failed", interview_id)
        else:
            logger.warning(
                "Interview %s stopped: %s: %s",
                interview_id,
                type(error).__name__,
                error,
            )

        try:
            await host.emit(Error(code=code, detail=f"{type(error).__name__}: {error}"))
        except Exception:
            logger.debug(
                "Interview %s failure could not be reported",
                interview_id,
                exc_info=True,
            )
            return

    await websocket.close()
