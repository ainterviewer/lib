# Versioned interview runtime: event-emitting workers pinned per project

## Context

Today the lib runs **in-process** inside the backend, so there is exactly one
lib version for the whole platform. Releasing a lib change alters behaviour for
every running project mid-fieldwork. Research projects need a fixed interviewer
(prompts, agents, flow) for their whole data-collection period, with hotfixes
still allowed. Goal: many lib versions run side by side, each project is pinned
to one at creation, and the lib can keep moving.

Decisions so far:

- **Pin at creation.** A project gets a runtime line (`X.Y`) when it is
  created. Patch hotfixes reach it through the moving `:X.Y` image tag.
- **The worker emits events; the backend persists them.**
- **The backend relays the participant websocket.**
- **No backwards compatibility.** The one project still on lib 0.3.5 ends its
  fieldwork this week. Until then, prod stays on backend v0.4.38 and only gets
  hotfixes. The new architecture is built and tested on staging, then switched
  over once, after that project has finished. There is no in-process legacy
  path, and the lib does not need to support 0.3.
- **The client wire format moves out of the lib.** The format the browser
  receives (`OutgoingMessage`/`OutgoingData`) becomes backend-owned, so old
  workers never dictate what the frontend receives.

## Findings (verified)

- **One process means one lib version.** The backend pins `ainterviewer` by git
  tag (`backend/pyproject.toml`, `v0.3.5`; the lib is at `v0.4.4`, which was
  never deployed). `ai.py:149` runs `AInterviewer` in the backend's event loop.
- **The lib's models are the DB schema.**
  `backend/app/db/tables.py:333,407,418,657` store `InterviewConfig`,
  `InterviewGuide` and `AgentConfigs` as `PydanticJSONB(<lib model>)`.
- **Persistence is already effectively write-only.**
  `InterviewRepository.insert_message`
  (`backend/app/db/repositories/interview.py:639-705`) returns the `message_id`
  the lib passed in, and duplicates are skipped on `IntegrityError`. The lib owns
  message ids (`interview_history.current_message_id`). Nothing the lib calls on
  `PersistenceProtocol` returns information it lacks. **So persistence can become
  a stream of emitted events with no request/response.**
- **The lib mixes three concerns today.** `send_data` (`interview.py:241`) both
  writes the DB row (with section, question and probe indices) and sends the
  client payload. `receive_data` (`:207`) reads the client and writes the DB.
  Status, guide updates and tasks are separate DB calls.
- **The client wire format is muddled.** `OutgoingData` (`interfaces.py:66`)
  mixes control tokens passed in `content` (`end_of_interview`,
  `restart_interview`), progress and errors in one model. `send_data` picks the
  model by checking `text in CustomToken` (`interview.py:288`). Progress is sent
  two ways: inside `OutgoingMessage` and standalone in `OutgoingData`.
- **Interviews resume from stored messages** (`process_history`,
  `interview.py:399`), so workers can be killed and restarted.
- **Leaks the worker must cover:** `image.encode(project_id)`
  (`interview.py:661`) reads from project storage, and LLM settings come from
  `lib_settings`/config.toml.

## Architecture

### Before (today)

```
 Browser ──ws──▶ ┌──────────────── backend container ────────────────┐
                 │ /ws/ai ─▶ AInterviewer (lib vX, single version)   │──▶ LLM proxy
                 │              │ io.send / io.receive                │
                 │              │ db.insert_message / insert_task …   │
                 └──────────────┼─────────────────────────────────────┘
                                ▼
                             Postgres   (guide/config stored as lib-vX pydantic JSON)
```

### After

```
                 ┌──────────────────────────── backend ─────────────────────────────┐
 Browser ──ws──▶ │  /ws/ai: auth · session_manager · history replay                 │
    ▲            │        │                                                         │
    │            │        ▼                                                         │
    │            │  runtime router: project.runtime_version "0.5" ─▶ worker-0-5     │
    │            │        │                                                         │
    │            │        │   events from worker ──────────────▶ ┌──────────────────┐│
    │            │        │                                      │ EventSink        ││──▶ Postgres
    └────────────┼────────┼──── client wire format (backend) ◀───│ 1. persist       ││
                 │        │                                      │ 2. forward       ││
                 │        │                                      └──────────────────┘│
                 └────────┼────────────────────────────▲─────────────────────────────┘
                          │ Init, UserMessage           │ MessageEmitted, UserMessageRecorded,
                          ▼                             │ TaskRecorded, StatusChanged,
                 ┌─────────────────── internal network only ──────────────────┐ GuideUpdated…
                 │  ┌─ worker-0-5 ────────────┐   ┌─ worker-0-6 ────────────┐  │
                 │  │ ws /run   (transport)   │   │ ws /run                 │ …│──▶ LLM proxy
                 │  │ lib 0.5.x (engine)      │   │ lib 0.6.x               │  │
                 │  └─────────────────────────┘   └─────────────────────────┘  │
                 │  mounts: storage (read-only) · config.toml                  │
                 │  services generated from the DB by `just runtimes-sync`     │
                 └─────────────────────────────────────────────────────────────┘

  Project A (created on 0.5) ─▶ worker-0-5   ← hotfix 0.5.3→0.5.4 moves the :0.5 tag
  Project B (created on 0.6) ─▶ worker-0-6
```

### One turn

```
Browser          backend (/ws/ai + EventSink)              worker-0-5 (lib engine)
   │  connect  ──▶ auth, load raw JSON, replay history
   │                 ── Init{guide,config,agent_configs,history,…} ──▶ build AInterviewer
   │                                                     ◀── MessageEmitted{id=7, question…}
   │                 insert_message(id=7) → DB
   │ ◀── message(id=7)
   │ answer ──▶     ── UserMessage{text} ──────────────────────▶
   │                                                     ◀── UserMessageRecorded{id=8}
   │                 insert_message(id=8) → DB
   │                                                     ◀── TaskRecorded{has_question_been_answered}
   │                 insert_task → DB                        (LLM calls happen in worker)
   │                                                     ◀── MessageEmitted{id=9, probe…}
   │ ◀── message(id=9)                    …
```

## Boundaries

| Layer                                                         | Owns                                                                                                                                              | Does not own                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **lib (engine)**                                              | interview flow, agents, prompts, message ids, history reconstruction; typed **input events** in, **output events** out                            | DB, websockets, auth, client wire format, which version runs |
| **worker** (lives in the lib repo, one image per lib version) | transport: ws server, (de)serialising events, Init handshake                                                                                      | DB, auth                                                    |
| **backend**                                                   | auth, `session_manager`, DB, project→runtime pinning, mapping events → DB rows, **client wire format**, relaying the client ws, running workers    | interview logic                                             |

The **event schema is the contract** between versions. It must stay
additive-only (new optional fields and new event types are fine; renames and
removals are not), and each event carries `schema_version`. The backend must be
able to consume events from every line still in use.

## Implementation

### 0. Now, until the 0.3.5 project ends (this week)

- Prod stays on backend v0.4.38 with lib 0.3.5. Hotfixes only.
- All work below goes to `main` and is deployed to staging.

### 1. Lib: events instead of IO + persistence (`v0.5.0`, breaking)

- New `src/ainterviewer/events.py` (pydantic, discriminated union on `type`):
  - **Inbound:** `Init` (guide, config, agent_configs, prompt_overrides,
    history, referable_values, language, previous_time_spent, project/interview
    ids), `UserMessage` (text, message_type, audio_file).
  - **Outbound:** `MessageEmitted` (all current `insert_message` fields plus
    progress, is_probe, user_image), `UserMessageRecorded` (the processed answer
    with structure indices), `TaskRecorded`, `StatusChanged`, `GuideUpdated`,
    `Progress`, `InterviewEnded`, `Error`.
  - Control flow becomes explicit event types. No more `CustomToken`s passed as
    message text on the way out. Inbound special answers (`skip_question`)
    become a field on `UserMessage`.
- `interfaces.py`: replace `IOProtocol`, `PersistenceProtocol` and the
  `Outgoing*`/`ReceivedData` models with one `EngineHost` protocol:
  `async emit(event)` and `async next_input() -> UserMessage`. No legacy
  adapter.
- `interview.py`: `send_data`, `receive_data`, `send_progress`,
  `handle_skip_question_exception`, the `insert_task` call sites,
  `__aenter__`/`__aexit__`/`interview()` status updates and
  `update_interview_guide` all become `emit(...)`. `send_data` no longer
  branches on `text in CustomToken`.
- New `src/ainterviewer/worker/server.py` behind the `ainterviewer[worker]`
  extra: one websocket `/run`. It reads `Init`, builds `AInterviewer` with a
  `WebsocketHost`, forwards outbound events and feeds inbound `UserMessage`s. It
  also answers `/health` and `/version` (lib version and event `schema_version`).
- `Dockerfile` in the lib repo. The release CI publishes
  `ghcr.io/ainterviewer/worker:X.Y.Z` and moves `:X.Y`.

### 2. Backend

- Alembic migration: `ProjectTable.runtime_version: str NOT NULL`. Every
  existing project is backfilled to the first line (`"0.5"`). Finished projects
  don't run interviews, so nothing changes for them. New projects get
  `settings.runtime_default`. An admin-only "upgrade runtime" action exists but
  is not exposed to researchers yet.
- Worker address by naming convention, not a per-version URL map:
  line `X.Y` → `ws://ainterviewer-worker-X-Y:8700`.
- **Client wire format moves here**, in two steps:
  1. Move `OutgoingMessage`, `OutgoingHistoryMessage`, `OutgoingData` and
     `ReceivedData` from the lib into `app/api/websockets/schemas.py` unchanged,
     so the frontend doesn't change at cut-over.
  2. In a separate PR together with the frontend, redesign them as one
     discriminated union on `type`: `message` | `history` | `progress` |
     `control{action: end | restart}` | `error{code}`.
- New `app/api/websockets/interviews/events.py`: `EventSink` persists an event
  (if it is a persisted kind) through `InterviewDataBase`, then translates it
  into the client wire format and sends it.
- `ai.py` becomes a relay only. Auth, `session_manager`, history replay and DB
  loading stay. Then it opens a ws to the project's worker, sends `Init` with
  the **raw stored JSON** (not re-validated through the backend's lib models),
  and pumps both directions: worker events go to `EventSink`, and client
  messages go through `WebsocketMessageHandler` to `UserMessage`. If the worker
  can't be reached, send an `error` to the client instead of crashing.
- Remove the backend's runtime use of the lib: there is no `AInterviewer` import,
  and the local LLM error handling in `ai.py` moves into the worker's `Error`
  events.
- Snapshot `config` and `agent_configs` onto the interview at creation, as the
  guide already is.
- Guard rails: at startup, health-check every line referenced by a non-archived
  project and log any that are missing. Refuse to create a project if the
  `runtime_default` worker is unreachable.

### 3. Platform: no hand-maintained version list

- `just runtimes-sync <env>` (backed by a backend CLI command):
  1. Query `SELECT DISTINCT runtime_version FROM project WHERE status <> 'archived'`,
     plus `runtime_default`.
  2. Render `platform/compose/runtimes.<env>.yml` from one service template:
     image `worker:X.Y`, internal network only, storage mounted read-only,
     config.toml, proxy reachable.
  3. `docker compose pull` + `up -d --remove-orphans`.
- Run it on every deploy. Lines nobody uses get removed automatically. A
  hotfix is a lib patch release followed by a sync. The only manual step left
  is bumping `runtime_default`, which is deliberate.
- Later, if the number of lines grows: spawn workers on demand through the
  Docker API, or move to k8s. Not needed now.

### 4. Cut-over (after the 0.3.5 project ends)

1. Confirm that project has no active interviews.
2. Release lib `v0.5.0` and publish `worker:0.5`.
3. Deploy the backend (migration backfills `"0.5"`), then `runtimes-sync`.
4. Smoke-test one interview end to end on prod.

### 5. Deferred

Decoupling the backend's DB/editor from lib models (an `ainterviewer-schema`
package with schema versions and migrations). Until then, a rule: lib model
changes must be additive, so that guides edited by a newer backend still parse
in older workers.

## Critical files

- lib: `src/ainterviewer/interfaces.py`, `src/ainterviewer/interview.py`, new
  `src/ainterviewer/events.py` and `src/ainterviewer/worker/`, `pyproject.toml`,
  new `Dockerfile` and release workflow
- backend: `app/api/websockets/interviews/ai.py`,
  `app/api/websockets/handler.py`, new `app/api/websockets/schemas.py` and
  `app/api/websockets/interviews/events.py`, `app/db/tables.py` and a
  migration, `app/settings.py`, project creation in
  `app/db/repositories/project.py`, `app/cli.py` (runtimes command)
- platform: `platform/compose/{base,dev,staging,prod}.yml`, the template for
  `runtimes.<env>.yml`, `platform/justfile`

## Verification

- lib: tests that drive `AInterviewer` with a fake `EngineHost` (scripted
  `UserMessage`s) and assert the emitted event sequence. The existing suite,
  ported to `EngineHost`, must pass.
- worker: in-process uvicorn test, `Init` → events over a real websocket.
- backend: an end-to-end `/ws/ai` test against a worker container. Run a
  scripted interview and assert the DB rows and the client messages, which must
  match today's wire format until the redesign PR.
- Isolation: two projects pinned to `worker:0.5` and a local `worker:0.6` with
  a changed prompt each get their own behaviour.
- `runtimes-sync`: archiving the last project on a line removes its service.
- Kill a worker mid-interview, reconnect, and confirm the resume works via
  `process_history` and that duplicate inserts are skipped.
- Locust (`backend/scripts/locust_file.py`): compare latency against today's
  in-process setup.
