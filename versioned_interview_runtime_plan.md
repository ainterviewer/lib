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
- **No backwards compatibility.** The architecture is built and tested on
  staging, then switched over once. There is no in-process legacy path. The
  first runtime line is the release that introduces the event engine
  (`v0.8.0`, line `"0.8"`); nothing older needs to run as a worker.
- **The client wire format moves out of the lib.** The format the browser
  receives (`OutgoingMessage`/`OutgoingData`) becomes backend-owned, so old
  workers never dictate what the frontend receives.
- **Workers run the rendered guide.** The multilingual source
  (`LocalizedInterviewGuide`), rendering, translation and guide generation stay
  editor-side in the backend. A worker only ever gets the per-interview
  `InterviewGuide` snapshot in one language.

## What changed since the first draft (lib v0.4.4 → v0.7.1)

The first draft was written against lib 0.4.4. Since then:

- **The backend moved on.** It is at v0.10.9 and pins lib `v0.7.1`
  (`backend/pyproject.toml:76`), so the "prod on v0.4.38 / lib 0.3.5" freeze
  is out of date. Confirm whether the 0.3.5 project has finished (the old plan
  expected it to end the week of 2026-10-12) and what prod runs now.
- **Guide models broke three times.** 0.5.0 added stable section/question ids
  and made conditions and references name questions by `question_id`. 0.6.0
  added the multilingual source (`interview_guides/localized.py`) with option
  ids. The backend migrates stored guides instead of the lib upgrading on load.
  This was fine with one lib version, but it is exactly the kind of change the
  "additive-only" rule (see Deferred) has to stop once several lines run.
- **The engine talks to its host in more ways:**
  - **Embedding** (0.4.x, now wired up): `AInterviewer(embedder=...)` calls
    `EmbeddingProtocol.embed_chunk` (`interview.py:260`) for message, QA-pair
    and interview chunks. The backend queues them (`app/embed/queue.py`), and
    `app/embed/backfill.py` re-derives chunks from stored messages.
  - **Security agent** (0.7.0): an assessment runs as a **background task**
    next to the next generation (`start_security_check`), and it writes
    `assess_security` tasks from inside that task (`interview.py:1131,1144`).
    If something triggers, it sends a message carrying a `SecurityIntervention`
    (shown in a modal) and may wait for a **second kind of input**, a
    `SecurityOverride` answer (`receive_security_override`, `:1195`).
  - **The engine declares the expected input.** `receive_data` (`:297`) now
    works out from `interview_history.current_turn` whether a survey answer or
    free text is expected, and passes that to `io.receive_message`.
  - **Resume reads more fields.** `process_history` (`:536`) and
    `InterviewHistory` now read `security_intervention`, `message_type` and
    `survey_item` off the stored messages, and resume can act on a pending
    security intervention.
- **The client wire format grew.** `OutgoingMessage` gained `question_id`
  (0.6.1) and `security_intervention` (0.7.0). `ReceivedData` stopped
  HTML-escaping (0.4.1). Escaping now happens in the frontend.
- **The lib got lighter.** fastapi, uvicorn, websockets, openai and the
  local/torch extra are gone from the dependencies. The worker extra adds back
  only what the server needs.

## Findings (verified against lib v0.7.1 / backend v0.10.9)

- **One process means one lib version.** `ai.py:276` runs `AInterviewer` in
  the backend's event loop and passes `embedder=`.
- **The lib's models are the DB schema.** `backend/app/db/tables.py` stores
  `InterviewConfig` (`:426`), `LocalizedInterviewGuide` / `LocalizedConsent` /
  `LocalizedWelcome` (`:505-512`), `AgentConfigs` (`:522`, now including the
  security policy) on the project setup, and the rendered `InterviewGuide`
  (`:967`) plus `language` on the interview. About 66 backend modules import
  the lib.
- **Persistence is still effectively write-only.**
  `InterviewRepository.insert_message` (`interview.py:868-922`) returns the
  `message_id` the lib passed in, and duplicates are skipped on
  `IntegrityError` (`insert_task` the same, `:1165-1193`). The lib owns message
  ids. Embedding is fire-and-forget by contract ("must not raise"). **So every
  host call can become an emitted event with no request/response.** The only
  real requests are the two inputs: an answer and a security override.
- **The lib still mixes three concerns.** `send_data` (`interview.py:356`)
  writes the DB row and sends the client payload, and still picks the model by
  `text in CustomToken` (`:406`). `receive_data` reads the client and writes
  the DB. Status, guide updates, tasks and embedding chunks are separate calls.
- **Calls into the host are concurrent now.** The background security task
  calls `insert_task` while the main loop may be sending. Over a websocket,
  `emit` must serialise writes.
- **Embedding assumes one producer version.** The live path and the backfill
  must produce byte-identical chunk text (`EmbeddingTable` docstring,
  `tables.py:1685`). A row's identity (`EmbeddingRepository.chunk_key`,
  `embedding.py:368`) is kind + interview + coordinates + task + model, with
  **no format version**. `needs_embedding` (`:413`) calls a row stale when
  `(content_hash, format_version)` differs from what the backfill derives, and
  `store` (`:440`) deletes and replaces it. The backfill only ever visits
  chunks its own policy derives. It never deletes rows the policy no longer
  produces.
- **Interviews resume from stored messages** (`process_history`), so workers
  can be killed and restarted. That now includes a respondent sitting on a
  security modal.
- **Leaks the worker must cover:** `image.encode(project_id)`
  (`interview.py:899`) reads from project storage, and LLM settings come from
  `lib_settings`/config.toml.

## Architecture

### Before (today)

```txt
 Browser ──ws──▶ ┌──────────────── backend container ────────────────┐
                 │ /ws/ai ─▶ AInterviewer (lib vX, single version)   │──▶ LLM proxy
                 │              │ io.send / io.receive                │
                 │              │ db.insert_message / insert_task …   │
                 │              │ embedder.embed_chunk ─▶ ChunkQueue  │
                 └──────────────┼─────────────────────────────────────┘
                                ▼
                             Postgres   (guide/config stored as lib-vX pydantic JSON)
```

### After

```txt
                 ┌──────────────────────────── backend ─────────────────────────────┐
 Browser ──ws──▶ │  /ws/ai: auth · session_manager · history replay                 │
    ▲            │        │                                                         │
    │            │        ▼                                                         │
    │            │  runtime router: project.runtime_version "0.8" ─▶ worker-0-8     │
    │            │        │                                                         │
    │            │        │   events from worker ──────────────▶ ┌──────────────────┐│
    │            │        │                                      │ EventSink        ││──▶ Postgres
    └────────────┼────────┼──── client wire format (backend) ◀───│ 1. persist       ││
                 │        │                                      │ 2. forward       ││──▶ ChunkQueue
                 │        │                                      └──────────────────┘│
                 └────────┼────────────────────────────▲─────────────────────────────┘
                          │ Init, UserMessage,          │ MessageEmitted, InputRequested,
                          │ SecurityOverrideAnswer      │ UserMessageRecorded, TaskRecorded,
                          ▼                             │ QuestionGroupCompleted, StatusChanged …
                 ┌─────────────────── internal network only ──────────────────┐
                 │  ┌─ worker-0-8 ────────────┐   ┌─ worker-0-9 ────────────┐  │
                 │  │ ws /run   (transport)   │   │ ws /run                 │ …│──▶ LLM proxy
                 │  │ lib 0.8.x (engine)      │   │ lib 0.9.x               │  │
                 │  └─────────────────────────┘   └─────────────────────────┘  │
                 │  mounts: storage (read-only) · config.toml                  │
                 │  services generated from the DB by `just runtimes-sync`     │
                 └─────────────────────────────────────────────────────────────┘

  Project A (created on 0.8) ─▶ worker-0-8   ← hotfix 0.8.3→0.8.4 moves the :0.8 tag
  Project B (created on 0.9) ─▶ worker-0-9
```

### One turn

```txt
Browser          backend (/ws/ai + EventSink)              worker-0-8 (lib engine)
   │  connect  ──▶ auth, load raw JSON, replay history
   │                 ── Init{guide,config,agent_configs,history,…} ──▶ build AInterviewer
   │                                                     ◀── MessageEmitted{id=7, question_id…}
   │                 insert_message(id=7) → DB
   │ ◀── message(id=7)
   │                                                     ◀── InputRequested{id=8, expects=free_text}
   │ answer ──▶     validate against `expects`
   │                ── UserMessage{text} ──────────────────────▶
   │                                                     ◀── UserMessageRecorded{id=8}
   │                 insert_message(id=8) → DB                (security check starts in background)
   │                 message_chunk(row) → ChunkQueue
   │                                                     ◀── TaskRecorded{has_question_been_answered}
   │                                                     ◀── TaskRecorded{assess_security}
   │                                                     ◀── MessageEmitted{id=9, probe…}
   │ ◀── message(id=9)                    …
   │                                                     ◀── QuestionGroupCompleted{section=1, main_question=2}
   │                 reload messages → chunks_for_interview(group) → ChunkQueue
```

A triggered security check adds `MessageEmitted{security_intervention}` and,
if the respondent may override it, `InputRequested{expects=security_override}`.
The backend shows the modal and answers with `SecurityOverrideAnswer`.

## Boundaries

| Layer                                                         | Owns                                                                                                                                                       | Does not own                                                 |
| ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **lib (engine)**                                              | interview flow, agents (incl. security), prompts, message ids, history reconstruction; typed **input events** in, **output events** out                  | DB, websockets, auth, client wire format, which version runs, embedding |
| **lib (schema/editor)**, imported by the backend              | guide/config models, localized source, rendering, `translate_missing`, guide generation, chunk policy (live and backfill)                                 | running interviews                                           |
| **worker** (lives in the lib repo, one image per lib version) | transport: ws server, (de)serialising events, Init handshake, serialised `emit`                                                                            | DB, auth                                                     |
| **backend**                                                   | auth, `session_manager`, DB, project→runtime pinning, mapping events → DB rows, **all embedding chunks** (live and backfill), **client wire format**, relaying the client ws, running workers | interview logic                                              |

The **event schema is the contract** between versions. It must stay
additive-only (new optional fields and new event types are fine; renames and
removals are not), and each event carries `schema_version`. The backend must be
able to consume events from every line still in use. The contract includes the
`HistoryRecord` shape in `Init`. The backend's `InterviewHistory` must also
reconstruct messages written by every live line, since it derives the
embedding chunks.

## Implementation

### 0. Now

- Confirm prod's state (see "What changed"). If the 0.3.5 project is done,
  drop the hotfix-only branch.
- All work below goes to `main` and is deployed to staging.

### 1. Lib: events instead of IO + persistence (`v0.8.0`, breaking)

- New `src/ainterviewer/events.py` (pydantic, discriminated union on `type`):
  - **Inbound:**
    - `Init`: the rendered `InterviewGuide`, config, agent_configs,
      prompt_overrides, history, referable_values, language,
      previous_time_spent, project/interview ids, chunk policy version.
    - `UserMessage`: text, message_type, audio_file, plus a `skip` flag in
      place of the `skip_question` token.
    - `SecurityOverrideAnswer`: `accept | override`.
  - **`HistoryRecord`**: a typed model for what `process_history` and
    `InterviewHistory` read off stored messages today, untyped (role, content,
    message_type, section/question/probe indices, flags, `survey_item`,
    `security_intervention`, image primer). The backend maps its rows into it.
  - **Outbound:**
    - `MessageEmitted`: all current `insert_message` fields plus progress,
      is_probe, user_image, `question_id`, `security_intervention`.
    - `InputRequested`: message_id and
      `expects: free_text | survey_item | security_override`. This replaces
      the `message_type` argument to `io.receive_message`.
    - `UserMessageRecorded`: the processed answer with structure indices. The
      security-override answer is recorded the same way.
    - `TaskRecorded`, including `assess_security`.
    - `QuestionGroupCompleted`: section and main-question index, emitted
      where `_emit_qa_pair` runs today (end of `handle_question`). It is a
      structural signal, not a chunk (see "Embedding" in step 2).
    - `StatusChanged`, `GuideUpdated`, `Progress`, `InterviewEnded`, `Error`.
  - Control flow becomes explicit event types. No more `CustomToken`s passed as
    message text on the way out.
- `interfaces.py`: replace `IOProtocol`, `PersistenceProtocol`,
  `EmbeddingProtocol` and the `Outgoing*`/`ReceivedData` models with one
  `EngineHost` protocol: `async emit(event)` and
  `async next_input(request: InputRequested) -> UserMessage | SecurityOverrideAnswer`.
  `SecurityIntervention` and `SecurityOverride` stay in the lib as event
  payloads. The `embedder`/`chunk_policy` constructor arguments and
  `_emit_chunk` go away. `EmbeddingChunk`, `ChunkPolicy` and the chunk
  builders stay in the lib for the backend (they could move to
  `embedding.py`). No legacy adapter.
- `interview.py`: `send_data`, `receive_data`, `receive_security_override`,
  `send_progress`, `handle_skip_question_exception`, every `insert_task` call
  site (including the ones inside `_assess_security`), `_emit_qa_pair`,
  `__aenter__`/`__aexit__`/`interview()` status updates and
  `update_interview_guide` all become `emit(...)`. `send_data` no longer
  branches on `text in CustomToken`. `process_history` takes
  `list[HistoryRecord]`.
- Concurrency: the background security task emits while the main loop runs.
  `emit` is safe to call concurrently, and events keep the order each caller
  emits them in. The backend must not assume that a `TaskRecorded` arrives
  next to its message.
- New `src/ainterviewer/worker/server.py` behind the `ainterviewer[worker]`
  extra (re-adds a small ASGI server and websockets): one websocket `/run`. It
  reads `Init`, builds `AInterviewer` with a `WebsocketHost`, forwards
  outbound events and feeds inbound inputs. It also answers `/health` and
  `/version` (lib version and event `schema_version`).
- `Dockerfile` in the lib repo. The release workflow's tag-only jobs publish
  `ghcr.io/ainterviewer/worker:X.Y.Z` and move `:X.Y`.

### 2. Backend

- Alembic migration: `ProjectTable.runtime_version: str NOT NULL`. Every
  existing project is backfilled to the first line (`"0.8"`). Finished projects
  don't run interviews, so nothing changes for them. New projects get
  `settings.runtime_default`. An admin-only "upgrade runtime" action exists but
  is not exposed to researchers yet.
- Worker address by naming convention, not a per-version URL map:
  line `X.Y` → `ws://ainterviewer-worker-X-Y:8700`.
- **Client wire format moves here**, in two steps:
  1. Move `OutgoingMessage` (with `question_id` and `security_intervention`),
     `OutgoingHistoryMessage`, `OutgoingData`, `ReceivedData` and the client
     tokens `ai.py` uses from `CustomToken` into
     `app/api/websockets/schemas.py` unchanged, so the frontend doesn't change
     at cut-over.
  2. In a separate PR together with the frontend, redesign them as one
     discriminated union on `type`: `message` | `history` | `progress` |
     `security{action, respondent_override}` | `control{action: end | restart}` |
     `error{code}`.
- New `app/api/websockets/interviews/events.py`: `EventSink` persists an event
  (if it is a persisted kind) through `InterviewDataBase`, derives embedding
  chunks from it (see "Embedding" below), then translates the event into the
  client wire format and sends it.
- `ai.py` becomes a relay only. Auth, `session_manager`, history replay and DB
  loading stay. Then it opens a ws to the project's worker and sends `Init`
  with the interview's **raw stored JSON**: the rendered `interview_guide`
  snapshot, not re-validated through the backend's lib models. It pumps both
  directions: worker events go to `EventSink`. Client messages are validated
  against the last `InputRequested` by `WebsocketMessageHandler` and sent on
  as `UserMessage` or `SecurityOverrideAnswer`. If the worker can't be
  reached, send an `error` to the client instead of crashing.
- Remove the backend's **runtime** use of the lib: no `AInterviewer` import,
  and the local LLM error handling in `ai.py` moves into the worker's `Error`
  events. The backend keeps importing the lib for schema/editor work at its
  own pinned version.
- Snapshot `config` and `agent_configs` (security policy included) onto the
  interview at creation, as the rendered guide already is.
- **Embedding: the backend produces every chunk.** If workers emitted chunks,
  the corpus would have two producers on different lib versions. The live
  chunks would use the project's pinned line, while the backfill uses the
  backend's lib. Chunk text depends on both the `ChunkPolicy` and
  `InterviewHistory`/`transcribe`, and those can drift between lines without
  anyone bumping `CHUNK_FORMAT_VERSION`. Because the row key has no format
  version, the next scheduled backfill would then see every live QA-pair and
  interview chunk as stale and re-embed it. That is exactly the
  re-embed-everything the shared-policy design exists to prevent. With an
  older backend lib, it would also downgrade chunks a newer worker wrote.
  Rows that only one policy produces would never be cleaned up. Comparing
  `format_version` doesn't fix this: it misses rendering drift within one
  version, and it can't tell "newer" from "older" for opaque strings.
  Embedding is analysis infrastructure, not interviewer behaviour, so it
  doesn't need pinning. So:
  - Message chunks: on `UserMessageRecorded`, `EventSink` calls
    `message_chunk(...)` with the backend's policy on the row it just wrote,
    exactly as `chunks_for_interview` does, and puts it on `ChunkQueue`.
  - QA-pair chunks: on `QuestionGroupCompleted`, reload the interview's
    messages and run the backend's `chunks_for_interview`, keeping only that
    group's `QA_PAIR` chunk. One indexed read per question group is cheap.
    Run it off the relay's hot path, like the queue consumer.
  - Interview (and section) chunks: on `InterviewEnded`, the same, keeping the
    `INTERVIEW` and `SECTION` chunks.
  - The backfill is unchanged and stays the single source of truth. Live
    output now matches it byte for byte by construction, whatever line the
    project runs on. Bumping `CHUNK_FORMAT_VERSION` in the backend's lib
    re-embeds the corpus once, as it does today.
  - Cost: the backend's `InterviewHistory.process_history` must handle
    messages written by every live line (see the contract above). The backfill
    already needs that.
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

### 4. Cut-over

1. Confirm no project has active interviews mid-question, or accept that they
   resume on `worker:0.8` through `process_history`.
2. Release lib `v0.8.0` and publish `worker:0.8`.
3. Deploy the backend (migration backfills `"0.8"`), then `runtimes-sync`.
4. Smoke-test one interview end to end on prod, including one security
   intervention and one survey item.

### 5. Deferred, and now more urgent

Decoupling the backend's DB/editor from lib models (an `ainterviewer-schema`
package with schema versions and migrations). 0.5–0.7 showed how often these
models break. Until the split, a rule from `v0.8.0` on: the models a worker
parses (`InterviewGuide`, `InterviewConfig`, `AgentConfigs` including
`SecurityPolicy`, `HistoryRecord`) only change additively,
so data written by a newer backend still parses in older workers. The
localized source is editor-only and may keep changing, as long as it renders
to an `InterviewGuide` every live line accepts. Breaking a worker-facing model
means a new line plus a backend migration that only touches projects on that
line.

## Critical files

- lib: `src/ainterviewer/interfaces.py`, `src/ainterviewer/interview.py`,
  `src/ainterviewer/interview_guides/history.py` (`HistoryRecord`),
  `src/ainterviewer/embedding.py`, new `src/ainterviewer/events.py` and
  `src/ainterviewer/worker/`, `pyproject.toml`, new `Dockerfile`,
  `.github/workflows/release.yml`
- backend: `app/api/websockets/interviews/ai.py`,
  `app/api/websockets/handler.py`, new `app/api/websockets/schemas.py` and
  `app/api/websockets/interviews/events.py`, `app/embed/queue.py`,
  `app/embed/backfill.py`, `app/db/tables.py` and a migration,
  `app/settings.py`, project creation in `app/db/repositories/project.py`,
  `app/cli.py` (runtimes command)
- platform: `platform/compose/{base,dev,staging,prod}.yml`, the template for
  `runtimes.<env>.yml`, `platform/justfile`

## Verification

- lib: tests that drive `AInterviewer` with a fake `EngineHost` (scripted
  inputs answering each `InputRequested`) and assert the emitted event
  sequence. The existing suite (`test_security.py`, `test_embedding.py`,
  `test_history.py` …), ported to `EngineHost`, must pass.
- lib: a security check that triggers while a probe is being generated emits
  its `TaskRecorded` and intervention in a consistent order. Two concurrent
  `emit` calls never interleave a frame.
- worker: in-process server test, `Init` → events over a real websocket.
- backend: an end-to-end `/ws/ai` test against a worker container. Run a
  scripted interview (free text, survey item, overridable security
  intervention) and assert the DB rows, the queued chunks and the client
  messages. Then run `backfill(dry_run=False)` on the same interview and
  assert that `needs_embedding` finds nothing to do. The client messages must match today's wire format until the
  redesign PR.
- Isolation: two projects pinned to `worker:0.8` and a local `worker:0.9` with
  a changed prompt each get their own behaviour.
- `runtimes-sync`: archiving the last project on a line removes its service.
- Kill a worker mid-interview, including while a security modal is open.
  Reconnect and confirm that the resume works via `process_history` and that
  duplicate inserts are skipped.
- Locust (`backend/scripts/locust_file.py`): compare latency against today's
  in-process setup, with the security agent on.
