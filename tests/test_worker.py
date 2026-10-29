"""The runtime server, over HTTP and a real websocket."""

import asyncio
import json
import uuid

import pytest
from any_llm.exceptions import ModelNotFoundError, RateLimitError
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ainterviewer import __version__
from ainterviewer.agents.config import AgentConfigs
from ainterviewer.compat import SCHEMA_VERSION
from ainterviewer.config import InterviewConfig
from ainterviewer.embedding import chunks_for_records
from ainterviewer.events import (
    ErrorCode,
    Init,
    InputKind,
    InputRequested,
    Progress,
    SecurityOverrideAnswer,
    UserMessage,
)
from ainterviewer.exceptions import UnexpectedInput
from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.history import HistoryRecord
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.types import SecurityOverride
from ainterviewer.types import MessageRole
from ainterviewer.worker.host import ProtocolError, WebsocketHost
from ainterviewer.worker.server import RUNTIME_VERSION, app, classify, runtime_version
from tests.guides import make_guide, two_sections
from tests.test_shuffle import _gated

PROJECT_ID = uuid.uuid4()
INTERVIEW_ID = uuid.uuid4()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skips the pauses the interviewer takes between messages"""
    sleep = asyncio.sleep

    async def no_sleep(*args, **kwargs):
        await sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_sleep)


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def init(**kwargs) -> Init:
    return Init(
        project_id=PROJECT_ID,
        interview_id=INTERVIEW_ID,
        guide=make_guide(),
        config=InterviewConfig.model_validate({}),
        agent_configs=AgentConfigs.model_validate({}),
        **kwargs,
    )


# ── metadata ─────────────────────────────────────────────────────────────────


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_version(client):
    version = client.get("/version").json()

    assert version["lib_version"] == __version__
    assert version["runtime_version"] == RUNTIME_VERSION
    assert __version__.startswith(f"{RUNTIME_VERSION}.")
    assert version["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize(
    ("lib_version", "name", "expected"),
    [
        ("0.9.0", None, "0.9"),
        ("0.9.3", "0.9", "0.9"),
        ("0.9.0rc2", "0.9-rc", "0.9-rc"),
        ("0.9.0", "0.9-feat-probing", "0.9-feat-probing"),
    ],
)
def test_runtime_version_is_named_from_the_library(lib_version, name, expected):
    assert runtime_version(lib_version, name) == expected


@pytest.mark.parametrize("name", ["0.8", "0.8-rc", "0.9-", "0.9-Feat", "0.9.1", "rc"])
def test_a_preview_cannot_pass_for_another_version(name):
    with pytest.raises(ValueError, match="must be '0.9'"):
        runtime_version("0.9.1", name)


def test_schema_describes_what_a_host_serialises(client):
    schema = client.get("/schema").json()

    assert schema.keys() == {"interview_guide", "interview_config", "agent_configs"}
    assert "question_sections" in schema["interview_guide"]["properties"]


# ── /validate ────────────────────────────────────────────────────────────────


def full_payload() -> dict:
    """Everything a host serialises, as this version dumps it."""
    return {
        "guide": two_sections("skip_section").model_dump(mode="json"),
        "config": InterviewConfig.model_validate({}).model_dump(mode="json"),
        "agent_configs": AgentConfigs.model_validate(
            {"security": {"include": True}}
        ).model_dump(mode="json"),
        "prompt_overrides": {"probing_agent/system_prompt.jinja": "{{ 1 }}"},
    }


def problems(client, payload: dict) -> dict[str, str]:
    response = client.post("/validate", json=payload).json()
    assert response["ok"] is not response["problems"]
    return {problem["path"]: problem["message"] for problem in response["problems"]}


def test_what_this_version_dumps_it_accepts(client):
    # A host serialising for its own runtime version must get exactly this.
    assert problems(client, full_payload()) == {}


def test_survey_items_round_trip(client):
    guide = make_guide().model_dump(mode="json")

    assert problems(client, {"guide": guide}) == {}


def test_a_field_a_model_would_ignore_is_named(client):
    payload = full_payload()
    payload["guide"]["question_sections"][0]["questions"][1]["future"] = True

    assert problems(client, payload) == {
        "guide.question_sections.0.questions.1.future": "Not supported by this version"
    }


def test_a_field_a_model_forbids_is_named_by_its_path(client):
    guide = make_guide().model_dump(mode="json")
    guide["question_sections"][0]["questions"][1]["survey_item"]["colour"] = "red"

    # No union member name in the path, only keys of the payload.
    assert list(problems(client, {"guide": guide})) == [
        "guide.question_sections.0.questions.1.survey_item.colour"
    ]


def test_a_missing_field_is_named(client):
    guide = make_guide().model_dump(mode="json")
    del guide["question_sections"][0]["questions"][0]["main_question"]

    assert list(problems(client, {"guide": guide})) == [
        "guide.question_sections.0.questions.0.main_question"
    ]


def test_config_and_agent_configs_are_checked(client):
    payload = full_payload()
    payload["config"]["future"] = 1
    payload["agent_configs"]["probing"]["future"] = 1

    assert set(problems(client, payload)) == {
        "config.future",
        "agent_configs.probing.future",
    }


def test_prompt_overrides_must_replace_compilable_templates(client):
    payload = {
        "prompt_overrides": {
            "probing_agent/system_prompt.jinja": "{% if %}",
            "future_agent/system_prompt.jinja": "x",
        }
    }

    assert set(problems(client, payload)) == {
        "prompt_overrides.probing_agent/system_prompt.jinja",
        "prompt_overrides.future_agent/system_prompt.jinja",
    }


def test_an_unknown_part_is_named(client):
    assert list(problems(client, {"future": {}})) == ["future"]


# ── /prepare and /chunks ─────────────────────────────────────────────────────


def test_prepare_shuffles_within_the_guide_constraints(client):
    read = Question(main_question="read", shuffle=True)
    others = [Question(main_question=f"q{i}", shuffle=True) for i in range(3)]
    guide = InterviewGuide(
        question_sections=[
            QuestionSection(
                description="s", questions=[read, *others, _gated("gated", read)]
            )
        ]
    ).model_dump(mode="json")

    orders = set()
    for _ in range(30):
        prepared = client.post("/prepare", json={"guide": guide}).json()["guide"]
        order = [
            q["main_question"] for q in prepared["question_sections"][0]["questions"]
        ]
        assert order.index("read") < order.index("gated")
        orders.add(tuple(order))

    assert len(orders) > 1


def test_prepare_refuses_what_validate_would(client):
    guide = make_guide().model_dump(mode="json")
    guide["question_sections"][0]["future"] = True

    response = client.post("/prepare", json={"guide": guide})

    assert response.status_code == 422
    assert response.json()["detail"][0]["path"] == "guide.question_sections.0.future"


def test_chunks_are_derived_as_the_engine_derives_them(client):
    guide = make_guide()
    history = [
        HistoryRecord(
            message_id=1,
            role=MessageRole.ASSISTANT,
            content="Free?",
            section=0,
            main_question=0,
            sub_question=0,
        ),
        HistoryRecord(
            message_id=2,
            role=MessageRole.USER,
            content="Busy but good.",
            section=0,
            main_question=0,
            sub_question=0,
        ),
    ]

    response = client.post(
        "/chunks",
        json={
            "guide": guide.model_dump(mode="json"),
            "history": [record.model_dump(mode="json") for record in history],
            "project_id": str(PROJECT_ID),
            "interview_id": str(INTERVIEW_ID),
            "language": "DA",
        },
    )

    expected = chunks_for_records(
        guide, history, project_id=PROJECT_ID, interview_id=INTERVIEW_ID, language="DA"
    )
    assert response.json() == [chunk.model_dump(mode="json") for chunk in expected]
    assert expected


# ── /run ─────────────────────────────────────────────────────────────────────


def test_an_interview_runs_over_the_socket(client):
    answers = ["My answer", "Yes"]
    seen = []

    with client.websocket_connect("/run") as ws:
        ws.send_text(init().model_dump_json())

        while (event := ws.receive_json())["type"] != "interview_ended":
            seen.append(event["type"])
            if event["type"] == "input_requested":
                ws.send_text(
                    UserMessage(
                        message_id=event["message_id"], content=answers.pop(0)
                    ).model_dump_json()
                )

        # Nothing follows the end but the close.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert not answers
    assert seen.count("input_requested") == 2
    assert seen.count("message_emitted") == 4
    assert "user_message_recorded" in seen


def error_after(client, *frames: str) -> dict:
    """The error the server reports after being sent `frames`."""
    with client.websocket_connect("/run") as ws:
        for frame in frames:
            ws.send_text(frame)
        while (event := ws.receive_json())["type"] != "error":
            pass
        return event


def test_a_session_must_start_with_init(client):
    error = error_after(client, json.dumps({"type": "user_message", "message_id": 1}))

    assert error["code"] == ErrorCode.PROTOCOL


def test_an_init_from_another_schema_version_is_refused(client):
    stale = init().model_copy(update={"schema_version": SCHEMA_VERSION + 1})

    error = error_after(client, stale.model_dump_json())

    assert error["code"] == ErrorCode.PROTOCOL
    assert "schema version" in error["detail"]


def test_the_wrong_kind_of_answer_is_a_protocol_error(client):
    error = error_after(
        client,
        init().model_dump_json(),
        SecurityOverrideAnswer(
            message_id=2, answer=SecurityOverride.ACCEPT
        ).model_dump_json(),
    )

    assert error["code"] == ErrorCode.PROTOCOL


# ── classification ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RateLimitError("slow down"), ErrorCode.INFERENCE_UNAVAILABLE),
        (ModelNotFoundError("gone"), ErrorCode.INFERENCE_ERROR),
        (UnexpectedInput("?"), ErrorCode.PROTOCOL),
        (ProtocolError("?"), ErrorCode.PROTOCOL),
        (KeyError("bug"), ErrorCode.INTERNAL),
    ],
)
def test_failures_are_classified(error, code):
    assert classify(error) == code


# ── the host ─────────────────────────────────────────────────────────────────


class SlowSocket:
    """A websocket whose sends take a while, and log when they start and end."""

    def __init__(self):
        self.log: list[str] = []
        self.release = asyncio.Event()

    async def send_text(self, text: str) -> None:
        self.log.append(f"start {text}")
        await self.release.wait()
        self.log.append(f"end {text}")


def frame(progress: float) -> str:
    return Progress(progress=progress).model_dump_json()


@pytest.mark.anyio
async def test_concurrent_emits_never_interleave():
    socket = SlowSocket()
    host = WebsocketHost(socket)  # ty:ignore[invalid-argument-type]

    sends = asyncio.gather(
        host.emit(Progress(progress=1)), host.emit(Progress(progress=2))
    )
    await asyncio.sleep(0)
    socket.release.set()
    await sends

    assert socket.log == [
        f"start {frame(1)}",
        f"end {frame(1)}",
        f"start {frame(2)}",
        f"end {frame(2)}",
    ]


@pytest.mark.anyio
async def test_a_cancelled_emit_still_sends_its_frame_whole():
    socket = SlowSocket()
    host = WebsocketHost(socket)  # ty:ignore[invalid-argument-type]

    first = asyncio.create_task(host.emit(Progress(progress=1)))
    await asyncio.sleep(0)
    first.cancel()
    second = asyncio.create_task(host.emit(Progress(progress=2)))
    await asyncio.sleep(0)
    socket.release.set()

    with pytest.raises(asyncio.CancelledError):
        await first
    await second

    assert socket.log == [
        f"start {frame(1)}",
        f"end {frame(1)}",
        f"start {frame(2)}",
        f"end {frame(2)}",
    ]


@pytest.mark.anyio
async def test_next_input_puts_the_request_and_reads_the_answer():
    class Socket:
        def __init__(self):
            self.sent = []

        async def send_text(self, text):
            self.sent.append(text)

        async def receive_text(self):
            return UserMessage(message_id=3, content="hi").model_dump_json()

    socket = Socket()
    request = InputRequested(message_id=3, expects=InputKind.FREE_TEXT)

    answer = await WebsocketHost(socket).next_input(request)  # ty:ignore[invalid-argument-type]

    assert socket.sent == [request.model_dump_json()]
    assert answer == UserMessage(message_id=3, content="hi")
