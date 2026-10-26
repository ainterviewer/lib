# Versioned interview runtime: event-emitting workers pinned per content version

## Context

Today the lib runs **in-process** inside the backend, so there is exactly one
lib version for the whole platform. Releasing a lib change alters behaviour for
every running project mid-fieldwork. Research projects need a fixed interviewer
(prompts, agents, flow) for their whole data-collection period, with hotfixes
still allowed. Goal: many lib versions run side by side, each study runs on a
fixed one, and the lib can keep moving.

Decisions so far:

- **Two axes.** The _content version_ is what is asked: researcher-controlled
  and already semver-classified by `app/db/versioning.py`. The _runtime version_
  (`X.Y`) is who asks it: the lib's prompts, agents and flow. Most edits move
  the first axis only.
- **The runtime version is pinned on the published content version** and is
  sticky from the project's **first real (non-test) interview**. Before that,
  every publish uses the default runtime version. After it, a new publish
  inherits the runtime version of the version it replaces, as long as the
  runtime version's worker accepts the content. Moving to a newer runtime
  version is an explicit upgrade. It is admin-only at first, and researchers get
  it once the feature has been fully tested.
- **One schema for the backend and frontend, frozen copies for the workers.**
  The backend, editor, frontend, analysis and exports only ever see the
  backend's lib version (vB). At publish, the backend freezes a
  `runtime_payload` in exactly the shape the target runtime version accepts, and
  that is all a worker ever sees. Migrations update the vB copies, never the
  frozen ones.
- **Runtime versions have a lifecycle**: supported → deprecated → retired, with
  3 months from deprecated to retired by default. On the retirement date,
  projects still on the runtime version are force-upgraded and in-progress
  interviews resume on the new runtime version. An admin can postpone retirement
  for one project.
- **Patch releases fix bugs, never prompts or flow.** The moving `:X.Y` tag
  reaches every project on the runtime version, and each interview records the
  exact lib version it ran on.
- **The worker emits events; the backend persists them.**
- **The backend relays the participant websocket.**
- **No backwards compatibility.** The architecture is built and tested on
  staging, then switched over once. There is no in-process legacy path. The
  first runtime version is the release that introduces the event engine
  (`v0.8.0`, runtime version `"0.8"`).
- **The client wire format moves out of the lib.** The format the browser
  receives becomes backend-owned, so old workers never dictate what the frontend
  receives.

## What changed since the first draft (lib v0.4.4 → v0.7.1)

The first draft was written against lib 0.4.4. Since then:

- **The backend moved on.** It is at v0.10.9 and pins lib `v0.7.1`
  (`backend/pyproject.toml:76`). The 0.3.5 project has finished, so nothing
  needs the in-process path any more.
- **Guide models broke three times.** 0.5.0 added stable section/question ids
  and made conditions and references name questions by `question_id`. 0.6.0
  added the multilingual source (`interview_guides/localized.py`) with option
  ids. The backend migrates stored guides instead of the lib upgrading on load.
  That only works with one lib version. See "Schema evolution rules".
- **The engine talks to its host in more ways:**
  - **Embedding** (0.4.x, now wired up): `AInterviewer(embedder=...)` calls
    `EmbeddingProtocol.embed_chunk` (`interview.py:260`) for message, QA-pair
    and interview chunks. The backend queues them (`app/embed/queue.py`), and
    `app/embed/backfill.py` re-derives chunks from stored messages.
  - **Security agent** (0.7.0): an assessment runs as a **background task** next
    to the next generation (`start_security_check`), and it writes
    `assess_security` tasks from inside that task (`interview.py:1131,1144`). If
    something triggers, it sends a message carrying a `SecurityIntervention`
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

### How backend, frontend and lib are tied together today

The backend imports one lib version (vB), and every layer below uses it:

| Coupling             | Where                                                                                                                                                                                                                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DB column types      | `tables.py`: draft `projectcontent` and immutable `projectcontentversion` (`LocalizedInterviewGuide`/`Consent`/`Welcome`, `InterviewConfig`, `AgentConfigs`); the interview's rendered, shuffled `InterviewGuide` (`:966`); message rows' `image`, `survey_item`, `security_intervention` (`:1158-1163`), all `PydanticJSONB(<vB model>)` |
| Editor               | live co-editing validates docs with `Localized*` (`app/collab/parts.py`); `translate_missing`; guide generation; publish-time semver classification (`app/db/versioning.py`)                                                                                                                                                              |
| Rendering, shuffling | `content.guide.render(language, fallback)` (`app/db/repositories/project.py:642`); `interview_guide.shuffle()` at interview creation (`app/api/interview.py:217`). Both run in the backend, so neither is pinned today                                                                                                                    |
| Frontend             | backend OpenAPI → `frontend/src/lib/api/types.gen.ts` + `zod.gen.ts` (runtime-validated), used by the guide editor, the interview player (`SurveyItem.svelte`, `MessageBubble.svelte`), analysis (`conditions.ts`), version preview and exports                                                                                           |
| Runtime              | `ai.py:276` runs `AInterviewer` in the backend's event loop                                                                                                                                                                                                                                                                               |
| Embedding            | `app/embed/backfill.py` is the backend's only use of `InterviewHistory`                                                                                                                                                                                                                                                                   |

Interviews already point at an immutable content version
(`InterviewTable.content_version_id`, `:1002`): the published one for real
interviews, a draft snapshot for tests.

### Why "keep worker-facing models additive" is not enough

- **The models disagree about unknown fields.** `InterviewConfig`, every
  `AgentConfig` and `SurveyItemBase` use `extra="forbid"`, so a field added in
  vB makes an older worker reject `Init`. `InterviewGuide`, `QuestionSection`
  and `Question` use pydantic's default, `ignore`, so a newer feature a
  researcher turns on is silently dropped by an older worker.
- **Breaking changes can't be migrated per runtime version** while every column
  is typed with vB models: the backend couldn't load the rows it left alone.
- **The frontend is one build.** It can't offer a different editor per runtime
  version.
- **History round-trips through vB.** Message rows store `survey_item` as a vB
  model, and resume reads it back.

### Engine facts the event design builds on

- **Persistence is still effectively write-only.**
  `InterviewRepository.insert_message` (`interview.py:868-922`) returns the
  `message_id` the lib passed in, and duplicates are skipped on `IntegrityError`
  (`insert_task` the same, `:1165-1193`). The lib owns message ids. Embedding is
  fire-and-forget by contract ("must not raise"). **So every host call can
  become an emitted event with no request/response.** The only real requests are
  the two inputs: an answer and a security override.
- **The lib still mixes three concerns.** `send_data` (`interview.py:356`)
  writes the DB row and sends the client payload, and still picks the model by
  `text in CustomToken` (`:406`). `receive_data` reads the client and writes the
  DB. Status, guide updates, tasks and embedding chunks are separate calls.
- **Calls into the host are concurrent now.** The background security task calls
  `insert_task` while the main loop may be sending. Over a websocket, `emit`
  must serialise writes.
- **Embedding assumes one producer version.** The live path and the backfill
  must produce byte-identical chunk text (`EmbeddingTable` docstring,
  `tables.py:1685`). A row's identity (`EmbeddingRepository.chunk_key`,
  `embedding.py:368`) is kind + interview + coordinates + task + model, with
  **no format version**. `needs_embedding` (`:413`) calls a row stale when
  `(content_hash, format_version)` differs from what the backfill derives, and
  `store` (`:440`) deletes and replaces it.
- **Interviews resume from stored messages** (`process_history`), so workers can
  be killed and restarted. That now includes a respondent sitting on a security
  modal.
- **Message rows carry no `question_id`**, only section/question/probe indices
  into the interview's own shuffled guide.
- **Leaks the worker must cover:** `image.encode(project_id)`
  (`interview.py:899`) reads from project storage, and LLM settings come from
  `lib_settings`/config.toml.

## Architecture

### Before (today)

```txt
 Browser ──ws──▶ ┌──────────────── backend container ────────────────┐
                 │ /ws/ai ─▶ AInterviewer (lib vB, single version)   │──▶ LLM proxy
                 │              │ io.send / io.receive                │
                 │              │ db.insert_message / insert_task …   │
                 │              │ embedder.embed_chunk ─▶ ChunkQueue  │
                 └──────────────┼─────────────────────────────────────┘
                                ▼
                             Postgres   (guide/config stored as lib-vB pydantic JSON)
```

### After

```txt
                 ┌──────────────────────────── backend (lib vB) ────────────────────┐
 Browser ──ws──▶ │  /ws/ai: auth · session_manager · history replay                 │
    ▲            │        │                                                         │
    │            │        ▼                                                         │
    │            │  runtime router: interview.runtime_version "0.8" ─▶ worker-0-8   │
    │            │        │                                                         │
    │            │        │   events from worker ──────────────▶ ┌──────────────────┐│
    │            │        │                                      │ EventSink        ││──▶ Postgres
    └────────────┼────────┼──── client wire format (vB shape) ◀──│ 1. persist (vB)  ││
                 │        │                                      │ 2. forward       ││──▶ ChunkQueue
                 │        │                                      └──────────────────┘│
                 └────────┼────────────────────────────▲─────────────────────────────┘
                          │ Init (frozen payload),      │ MessageEmitted, InputRequested,
                          │ UserMessage,                │ UserMessageRecorded, TaskRecorded,
                          │ SecurityOverrideAnswer      │ ChunkEmitted, StatusChanged …
                          ▼                             │
                 ┌─────────────────── internal network only ──────────────────┐
                 │  ┌─ worker-0-8 ────────────┐   ┌─ worker-0-9 ────────────┐  │
                 │  │ ws /run   (transport)   │   │ ws /run                 │ …│──▶ LLM proxy
                 │  │ POST /validate          │   │ POST /validate          │  │
                 │  │ POST /prepare           │   │ POST /prepare           │  │
                 │  │ POST /chunks            │   │ POST /chunks            │  │
                 │  │ GET /schema             │   │ GET /schema             │  │
                 │  │ lib 0.8.x (engine)      │   │ lib 0.9.x               │  │
                 │  └─────────────────────────┘   └─────────────────────────┘  │
                 │  mounts: storage (read-only) · config.toml                  │
                 │  services generated from the DB by `just runtimes-sync`     │
                 └─────────────────────────────────────────────────────────────┘
```

### Where each copy of the content lives

```txt
 draft (vB, editable) ──publish──▶ content version
                                     ├─ vB copy (migrated with the backend; editor, preview, analysis)
                                     ├─ runtime_version "0.8"
                                     └─ runtime_payload (frozen, 0.8 shape) ◀─ checked by worker-0-8 /validate

 content version ──new interview──▶ interview
                                     ├─ interview_guide (vB, shuffled; display, analysis, exports)
                                     ├─ runtime_version "0.8", runtime_lib_versions ["0.8.3", …]
                                     └─ runtime_guide (frozen, 0.8 shape, shuffled by worker-0-8 /prepare)
```

### One turn

```txt
Browser          backend (/ws/ai + EventSink)              worker-0-8 (lib engine)
   │  connect  ──▶ auth, load frozen payload, replay history
   │                 ── Init{runtime_guide,config,agent_configs,history,…} ──▶ build AInterviewer
   │                                                     ◀── MessageEmitted{id=7, question_id…}
   │                 insert_message(id=7) → DB
   │ ◀── message(id=7)
   │                                                     ◀── InputRequested{id=8, expects=free_text}
   │ answer ──▶     validate against `expects`
   │                ── UserMessage{text} ──────────────────────▶
   │                                                     ◀── UserMessageRecorded{id=8}
   │                 insert_message(id=8) → DB                (security check starts in background)
   │                                                     ◀── ChunkEmitted{message, id=8}
   │                 → ChunkQueue
   │                                                     ◀── TaskRecorded{has_question_been_answered}
   │                                                     ◀── TaskRecorded{assess_security}
   │                                                     ◀── MessageEmitted{id=9, probe…}
   │ ◀── message(id=9)                    …
   │                                                     ◀── ChunkEmitted{qa_pair, section=1, main_question=2}
   │                 → ChunkQueue
```

A triggered security check adds `MessageEmitted{security_intervention}` and, if
the respondent may override it, `InputRequested{expects=security_override}`. The
backend shows the modal and answers with `SecurityOverrideAnswer`.

## Boundaries

| Layer                                                         | Owns                                                                                                                                                                                                                                              | Does not own                                                          |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **lib (engine)**                                              | interview flow, agents (incl. security), prompts, shuffling, message ids, history reconstruction, **embedding chunks** (policy and rendering, live and backfill); typed **input events** in, **output events** out                                | DB, websockets, auth, client wire format, which version runs, vectors |
| **lib (schema/editor)**, imported by the backend              | guide/config models, localized source, rendering, `translate_missing`, guide generation, the `EmbeddingChunk` model, upgrade functions for worker-facing models (`ainterviewer.compat`)                                                           | running interviews, building chunks                                   |
| **worker** (lives in the lib repo, one image per lib version) | transport: ws server, (de)serialising events, Init handshake, serialised `emit`; `/validate`, `/prepare`, `/chunks`                                                                                                                               | DB, auth                                                              |
| **backend**                                                   | auth, `session_manager`, DB, runtime version pinning and lifecycle, freezing payloads, mapping events → vB rows, chunk queue, vectors and their storage, scheduling the backfill, **client wire format**, relaying the client ws, running workers | interview logic, chunk text                                           |
| **frontend**                                                  | one build on vB types; shows the runtime version, marks features the runtime version doesn't support                                                                                                                                              | anything per runtime version                                          |

The **contract between backend and workers** has three parts:

1. **Frozen payloads.** The backend sends a worker only JSON that the same
   runtime version already accepted (`/validate`) or produced (`/prepare`).
2. **Events and `HistoryRecord`.** They stay additive-only (new optional fields
   and new event types are fine; renames and removals are not), and each event
   carries `schema_version`. The backend consumes events from every live runtime
   version, upgrading older payload shapes with `ainterviewer.compat` before
   storing or forwarding them.
3. **`EmbeddingChunk`**, returned by `ChunkEmitted` and `/chunks`, is
   additive-only too.

Only workers reconstruct `InterviewHistory`, and `HistoryRecord` carries no
guide data. So the backend never parses guide JSON in another runtime version's
shape.

## Implementation

### 0. Now

- Retire the 0.3.5 hotfix setup (backend v0.4.38). That project has finished.
- All work below goes to `main` and is deployed to staging.

### 1. Lib: events instead of IO + persistence (`v0.8.0`, breaking)

- New `src/ainterviewer/events.py` (pydantic, discriminated union on `type`):
  - **Inbound:**
    - `Init`: the interview's `runtime_guide`, config, agent_configs,
      prompt_overrides, history, referable_values, language,
      previous_time_spent, project/interview ids.
    - `UserMessage`: text, message_type, plus a `skip` flag in place of the
      `skip_question` token.
    - `SecurityOverrideAnswer`: `accept | override`.
  - **`HistoryRecord`**: a typed model for what `process_history` and
    `InterviewHistory` read off stored messages (role, content, message_type,
    section/question/probe indices, flags, `security_intervention`). It carries
    **no guide data**. `survey_item` and the image primer are looked up by those
    indices in the worker's own `runtime_guide`, so a change to a vB guide model
    never reaches resume. The backend maps its rows into it.
  - **Outbound:**
    - `MessageEmitted`: all current `insert_message` fields plus progress,
      is_probe, user_image, `question_id`, `security_intervention`.
    - `InputRequested`: message_id and
      `expects: free_text | survey_item | security_override`. This replaces the
      `message_type` argument to `io.receive_message`.
    - `UserMessageRecorded`: the processed answer with structure indices. The
      security-override answer is recorded the same way.
    - `TaskRecorded`, including `assess_security`.
    - `ChunkEmitted`: carries an `EmbeddingChunk`, emitted wherever
      `_emit_chunk` runs today (message, QA pair, interview). A message chunk is
      emitted after that message's `UserMessageRecorded`, so its row already
      exists when `EmbeddingRepository.store` looks it up.
    - `StatusChanged`, `GuideUpdated`, `Progress`, `InterviewEnded`, `Error`.
  - Control flow becomes explicit event types. No more `CustomToken`s passed as
    message text on the way out.
- `interfaces.py`: replace `IOProtocol`, `PersistenceProtocol`,
  `EmbeddingProtocol` and the `Outgoing*`/`ReceivedData` models with one
  `EngineHost` protocol: `async emit(event)` and
  `async next_input(request: InputRequested) -> UserMessage | SecurityOverrideAnswer`.
  `SecurityIntervention` and `SecurityOverride` stay in the lib as event
  payloads. The `embedder` constructor argument goes away; `_emit_chunk` emits
  `ChunkEmitted` instead, still never raising. The chunk policy is the runtime
  version's own `DEFAULT_POLICY`. No legacy adapter.
- `embedding.py`: move the backend's `chunks_for_interview`
  (`app/embed/backfill.py`) into the lib as
  `chunks_for_records(guide, records: list[HistoryRecord], …)`. It builds
  message chunks per record and QA-pair, section and interview chunks from the
  reconstructed history, exactly as today. Live and backfill then share one
  implementation per runtime version.
- `interview.py`: `send_data`, `receive_data`, `receive_security_override`,
  `send_progress`, `handle_skip_question_exception`, every `insert_task` call
  site (including the ones inside `_assess_security`), `_emit_chunk`,
  `__aenter__`/`__aexit__`/`interview()` status updates and
  `update_interview_guide` all become `emit(...)`. `send_data` no longer
  branches on `text in CustomToken`. `process_history` takes
  `list[HistoryRecord]`.
- Concurrency: the background security task emits while the main loop runs.
  `emit` is safe to call concurrently, and events keep the order each caller
  emits them in. The backend must not assume that a `TaskRecorded` arrives next
  to its message.
- New `src/ainterviewer/compat.py`: one upgrade function per breaking change to
  a worker-facing model or event payload, keyed by `schema_version`
  (`upgrade(kind, json, from_version) -> json`). The backend's DB migrations and
  `EventSink` both use it (see "Schema evolution rules").
- New `src/ainterviewer/worker/server.py` behind the `ainterviewer[worker]`
  extra (re-adds a small ASGI server and websockets). Endpoints:
  - `ws /run`: reads `Init`, builds `AInterviewer` with a `WebsocketHost`,
    forwards outbound events and feeds inbound inputs.
  - `POST /validate`: `{guide, config, agent_configs, prompt_overrides}` in. It
    parses them **strictly** with this runtime version's models
    (`extra="forbid"` everywhere, for this check only) and returns ok, or a list
    of the paths it doesn't understand or can't accept.
  - `POST /prepare`: a validated guide in, this interview's shuffled guide out,
    using this runtime version's `shuffle()`.
  - `POST /chunks`:
    `{guide, history: list[HistoryRecord], project/interview ids, language}` in,
    `list[EmbeddingChunk]` out, via `chunks_for_records`.
  - `GET /schema`: the JSON Schema of this runtime version's worker-facing
    models (`InterviewGuide`, `InterviewConfig`, `AgentConfigs`), from
    pydantic's `model_json_schema()`. The backend serialises payloads against
    it.
  - `/health` and `/version` (lib version, event `schema_version`,
    `CHUNK_FORMAT_VERSION`).

  `/validate`, `/prepare`, `/chunks` and `/schema` keep no state and make no LLM
  calls.

- `Dockerfile` in the lib repo. The release workflow's tag-only jobs publish
  `ghcr.io/ainterviewer/worker:X.Y.Z` and move `:X.Y`.

### 2. Backend

#### Pinning and frozen payloads

- **`runtime_version` table**: version (`"0.8"`), status
  (`supported | deprecated | retired`), `deprecated_at`, `retire_at` (default
  `deprecated_at + 3 months`). Exactly one supported runtime version is the
  default. This replaces a `runtime_default` setting.
- **On `projectcontentversion`**: `runtime_version` and `runtime_payload` (plain
  JSONB, _not_ `PydanticJSONB`). The payload holds per offered language the
  rendered guide, plus config, agent_configs and prompt_overrides, serialised
  for the target runtime version. The existing vB columns stay as they are and
  keep being migrated.
- **On `interview`**: `runtime_version`, `runtime_guide` (plain JSONB: the
  runtime version's `/prepare` output) and `runtime_lib_versions` (each lib
  version reported by `/version` at `Init`, appended per session). The existing
  `interview_guide` (vB) stays for display, analysis and exports. It is
  reordered to match the worker's shuffle by section/question ids, which are
  stable across versions.
- **Serialising for a runtime version: what the editor shows is what runs.** The
  backend walks the full vB dump next to the target's `GET /schema` (cached per
  lib version), following `$defs` and the `type` discriminator of survey items:
  - **A field the target knows** is always written out, with the stored value.
    The worker's own defaults never decide anything, so defaults may change
    between runtime versions. Fields whose default comes from settings, such as
    `AgentConfig.model`, need no special case.
  - **A field the target doesn't know, at vB's default,** is dropped. The
    feature isn't used.
  - **A field the target doesn't know, not at default,** is kept, so `/validate`
    refuses it and names it.

  Stored content keeps the values it was saved with. After a default changes,
  existing questions keep their old value and new questions get the new one, and
  the editor shows each.

- **Which runtime version a publish gets**:
  - No real interview yet → the default runtime version.
  - Otherwise → the runtime version of the currently published content version,
    if it isn't retired.
  - Explicit upgrade → the chosen newer runtime version (admin-only for now).
  - The payload is then sent to that runtime version's `/validate`. On failure,
    the publish is refused with the paths returned, and the researcher chooses
    between "remove these features" and "upgrade to X.Y".
- **The editor checks on save, not just at publish.** The editor calls a new
  backend endpoint, which runs `/validate` for the runtime version the draft
  would publish to. The editor shows it as the "interviewer version"
  ("Interviewer version 0.8"), the user-facing name for a runtime version, and
  marks the features it doesn't support. The frontend stays one build on vB
  types.
- **Test interviews** run on the runtime version the draft would publish to,
  with an option to "test on X.Y" to preview an upgrade. Their snapshot versions
  get a payload like any publish.
- **New interview** (`app/api/interview.py`): the content version's payload for
  the respondent's language goes through `/prepare` on that content version's
  runtime version → `runtime_guide`. The vB `interview_guide` is reordered to
  match. This replaces the backend's `shuffle()` call.
- **An upgrade is provenance, not a content change.** It writes a new version
  with the same content (and semver) and a new runtime version. Analysis gets a
  filter by runtime version. Nothing more for now.

#### Runtime version lifecycle

- **Deprecating** a runtime version: no new projects start on it. Publishes may
  stay on it, with an "upgrade recommended" banner that shows `retire_at`. Only
  security and critical fixes get backported.
- **Retiring** (a scheduled job on `retire_at`):
  1. For each project whose published version is on the runtime version and has
     no admin postponement: re-render the **published** version's vB copy (not
     the draft), `/validate` it on the default runtime version, and publish it
     as an upgrade. A validation failure stops that project's upgrade, alerts an
     admin, and keeps the runtime version running for it.
  2. For each unfinished interview on the runtime version: serialise its vB
     `interview_guide` for the new runtime version, `/validate` it, and replace
     `runtime_guide` and `runtime_version`. The next connect resumes it on the
     new runtime version through `process_history`. `HistoryRecord` is
     independent of the runtime version, which is what makes this possible.
  3. Mark the runtime version retired. The next `runtimes-sync` removes its
     worker.
- **Admin override**: a per-project postponement (`retire_postponed_until`)
  keeps that project, and so the runtime version's worker, on the retired
  runtime version until the date.

#### Relay, events and wire format

- Worker address by naming convention, not a per-version URL map: runtime
  version `X.Y` → `ws://ainterviewer-worker-X-Y:8700`.
- **Client wire format moves here**, in two steps:
  1. Move `OutgoingMessage` (with `question_id` and `security_intervention`),
     `OutgoingHistoryMessage`, `OutgoingData`, `ReceivedData` and the client
     tokens `ai.py` uses from `CustomToken` into `app/api/websockets/schemas.py`
     unchanged, so the frontend doesn't change at cut-over.
  2. In a separate PR together with the frontend, redesign them as one
     discriminated union on `type`: `message` | `history` | `progress` |
     `security{action, respondent_override}` | `control{action: end | restart}`
     | `error{code}`.
- New `app/api/websockets/interviews/events.py`: `EventSink` takes an event in
  its runtime version's shape and runs `ainterviewer.compat` on it, so that
  `survey_item` and `image` are in vB shape. It persists the event (if it is a
  persisted kind) into the vB message columns, and puts `ChunkEmitted` on the
  existing `ChunkQueue`. Then it translates the event into the client wire
  format and sends it. The frontend therefore only ever sees vB shapes, from
  every runtime version.
- `ai.py` becomes a relay only. Auth, `session_manager`, history replay and DB
  loading stay. Then it opens a ws to the interview's worker and sends `Init`
  built from the **frozen** `runtime_guide` and the version's `runtime_payload`,
  sent as raw JSON. It pumps both directions: worker events go to `EventSink`.
  Client messages are validated against the last `InputRequested` by
  `WebsocketMessageHandler` and sent on as `UserMessage` or
  `SecurityOverrideAnswer`. An audio answer's filename stays in the relay, keyed
  by that `InputRequested.message_id`, and is written onto the row when
  `UserMessageRecorded` for that id arrives. If the worker can't be reached,
  send an `error` to the client instead of crashing.
- Remove the backend's **runtime** use of the lib: no `AInterviewer` import, no
  `shuffle()`, and the local LLM error handling in `ai.py` moves into the
  worker's `Error` events. The backend keeps importing the lib for schema/editor
  work at its own pinned version.

#### Embedding: one chunk producer per project, its pinned runtime version

Chunk text depends on the `ChunkPolicy` and on `InterviewHistory`/`transcribe`,
and those can drift between runtime versions without anyone bumping
`CHUNK_FORMAT_VERSION`. The row key has no format version. So if live chunks and
backfilled chunks came from different lib versions, every scheduled backfill
would see the live chunks as stale and re-embed them, or downgrade them. So the
worker that ran the interview builds both its live chunks and its backfilled
ones, and the backend no longer imports any history or chunking code.

- Live: `EventSink` puts each `ChunkEmitted` on `ChunkQueue`. The queue,
  `QueueEmbedder`'s drop-don't-block behaviour, `EmbeddingService` and
  `EmbeddingRepository` are unchanged.
- Backfill (`app/embed/backfill.py`): `_interviews` and `_messages` stay. Group
  interviews by `interview.runtime_version`. Map each interview's message rows
  to `HistoryRecord`, using the same mapping as the `Init` history replay. Call
  `POST /chunks` on that runtime version's worker, with the interview's
  `runtime_guide`, in place of `chunks_for_interview`, which moves to the lib.
  `needs_embedding`, `embed_and_store` and the report are unchanged.
  `pending_chunks` and `rehydrate_text` get their chunks the same way.
- A runtime version that is down or retired: the backfill skips its interviews,
  logs them in `BackfillReport.failed`, and goes on. Interviews completed on a
  retired runtime version keep the corpus they have. That's acceptable for now.
  If it ever isn't, the backfill can bring a runtime version up temporarily.
- Each project's corpus can mix runtime versions only across an upgrade. Every
  search is scoped to one project (`EmbeddingTable` indexes and docstring), and
  the runtime version filter in analysis applies to search too.
- A chunk-policy change becomes a patch release on each runtime version that
  should get it, followed by `runtimes-sync`. The next backfill then re-embeds
  those interviews once, through the existing hash/`format_version` check.

#### Guard rails

- At startup, health-check every runtime version that `runtimes-sync` would run,
  and log any that are missing.
- Refuse to create a project, publish or start an interview when the target
  runtime version's worker is unreachable. The error names the runtime version.

### 3. Platform: no hand-maintained version list

- `just runtimes-sync <env>` (backed by a backend CLI command):
  1. Select the runtime versions to run: the default runtime version, plus every
     non-retired runtime version (or retired runtime version with an active
     postponement) that is referenced by a published or test version of a
     non-archived project, or by an unfinished interview.
  2. Render `platform/compose/runtimes.<env>.yml` from one service template:
     image `worker:X.Y`, internal network only, storage mounted read-only,
     config.toml, proxy reachable.
  3. `docker compose pull` + `up -d --remove-orphans`.
- Run it on every deploy and after the retirement job. Runtime versions nobody
  uses get removed automatically. A hotfix is a lib patch release followed by a
  sync. The manual steps left are deliberate ones: adding a runtime version to
  the table, making it the default, and deprecating old ones.
- Later, if the number of runtime versions grows: spawn workers on demand
  through the Docker API, or move to k8s. Not needed now.

### 4. Cut-over

1. No project is live, so nothing needs to resume across the switch.
2. Release lib `v0.8.0` and publish `worker:0.8`.
3. Deploy the backend. The migration:
   - creates `runtime_version` with `"0.8"` as the default runtime version;
   - sets `runtime_version = "0.8"` and builds `runtime_payload` for every
     existing content version from its vB columns;
   - sets `runtime_version = "0.8"` and `runtime_guide = interview_guide` for
     every existing interview.

   No project is live, so `v0.8.0` may still break guide models. In that case,
   the backend migration upgrades the vB columns first, as it does today, and
   builds the frozen copies from the upgraded data.

4. Run `runtimes-sync`. Run `/validate` over every migrated payload and fix
   anything it refuses before opening the runtime version to respondents.
5. Smoke-test one interview end to end on prod, including one security
   intervention and one survey item.

### 5. Schema evolution rules

These replace the old "keep it additive" note. Splitting the models into an
`ainterviewer-schema` package stays deferred.

- **Worker-facing models** (`InterviewGuide` and everything in it,
  `InterviewConfig`, `AgentConfigs` including `SecurityPolicy`) and the contract
  (`HistoryRecord`, events, `EmbeddingChunk`):
  - A new field's default must mean the old behaviour, so that an older runtime
    version can run content that doesn't use it.
  - Defaults of existing fields may change. Payloads write every field the
    target knows explicitly (see "Serialising for a runtime version").
  - A breaking change is allowed. It ships with an `ainterviewer.compat` upgrade
    function and a `schema_version` bump. The backend's DB migration applies it
    to the vB copies, and `EventSink` applies it to events from older runtime
    versions. Frozen payloads are never rewritten. They only change through an
    upgrade or a retirement, which builds a new payload from the vB copy.
- **Newer runtime versions accept everything older runtime versions did.** An
  upgrade or retirement must always validate. A runtime version that drops a
  feature needs an explicit migration of affected content before older runtime
  versions can retire onto it.
- **Editor-only models** (`Localized*`, consent, welcome) change freely with
  backend migrations, as today.

## Critical files

- lib: `src/ainterviewer/interfaces.py`, `src/ainterviewer/interview.py`,
  `src/ainterviewer/interview_guides/history.py` (`HistoryRecord`),
  `src/ainterviewer/embedding.py`, new `src/ainterviewer/events.py`,
  `src/ainterviewer/compat.py` and `src/ainterviewer/worker/`, `pyproject.toml`,
  new `Dockerfile`, `.github/workflows/release.yml`
- backend: `app/api/websockets/interviews/ai.py`,
  `app/api/websockets/handler.py`, new `app/api/websockets/schemas.py` and
  `app/api/websockets/interviews/events.py`, `app/api/interview.py` (interview
  creation, shuffle), `app/db/repositories/version.py` (publish, payloads,
  runtime version choice), `app/db/repositories/project.py`,
  `app/db/versioning.py`, `app/embed/queue.py`, `app/embed/backfill.py`,
  `app/db/tables.py` and a migration, `app/settings.py`, `app/cli.py` (runtimes
  command, retirement job)
- frontend: the guide editor and publish dialog (interviewer-version badge,
  unsupported features, upgrade choice), the deprecation banner, the runtime
  version filter in analysis
- platform: `platform/compose/{base,dev,staging,prod}.yml`, the template for
  `runtimes.<env>.yml`, `platform/justfile`

## Verification

- lib: tests that drive `AInterviewer` with a fake `EngineHost` (scripted inputs
  answering each `InputRequested`) and assert the emitted event sequence. The
  existing suite (`test_security.py`, `test_embedding.py`, `test_history.py` …),
  ported to `EngineHost`, must pass.
- lib: a security check that triggers while a probe is being generated emits its
  `TaskRecorded` and intervention in a consistent order. Two concurrent `emit`
  calls never interleave a frame.
- lib: resume from `HistoryRecord`s alone (no guide data in them) rebuilds the
  same `InterviewHistory` as today's message-row resume, survey items included.
- worker: in-process server test, `Init` → events over a real websocket.
  `/validate` refuses a guide that uses a field the runtime version lacks, and
  names that field's path. `/prepare` respects the shuffle constraints.
- Compatibility, against two real images (`worker:0.8` and a local `worker:0.9`
  that adds a question setting):
  - A guide built in a 0.9 backend that doesn't use the setting validates on
    0.8. One that uses it is refused.
  - A 0.8 project's interview runs on 0.8. The client receives vB shapes.
  - 0.9 changes an existing default, e.g. `max_probes_n` from 3 to 5. A 0.8
    payload of a question saved at 5 carries `max_probes_n: 5` and runs with 5.
  - Serialising a guide for the backend's own runtime version reproduces it
    exactly (round-trip), survey-item variants included.
- Pinning: before the first real interview, a publish takes the default runtime
  version. After it, the runtime version sticks. An explicit upgrade moves it.
  Test interviews run on the draft's target runtime version.
- Retirement: deprecate 0.8 and move `retire_at` to now. The job upgrades the
  published version, and an unfinished interview resumes on 0.9 from its
  history. A postponed project stays on 0.8. `runtimes-sync` keeps or drops the
  0.8 worker accordingly.
- backend: an end-to-end `/ws/ai` test against a worker container. Run a
  scripted interview (free text, survey item, overridable security intervention)
  and assert the DB rows, the queued chunks and the client messages. The client
  messages must match today's wire format until the redesign PR.
- Embedding: after that interview, run the backfill (it calls the worker's
  `/chunks`) and assert that `needs_embedding` finds nothing to do. That proves
  the live and backfill chunks are byte-identical. With the worker stopped, the
  backfill reports the interview in `failed` and carries on.
- Kill a worker mid-interview, including while a security modal is open.
  Reconnect and confirm that the resume works via `process_history` and that
  duplicate inserts are skipped.
- Cut-over migration: every migrated payload passes `/validate` on 0.8.
- Locust (`backend/scripts/locust_file.py`): compare latency against today's
  in-process setup, with the security agent on.
