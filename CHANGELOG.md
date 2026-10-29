# Changelog

Generated from [Conventional Commits](https://www.conventionalcommits.org) by
[git-cliff](https://git-cliff.org). Releases older than the earliest entry below
predate this changelog — see `git log` for their history.

## [0.8.0rc1] - 2026-10-29

### Breaking Changes

- (engine) IOProtocol, PersistenceProtocol, EmbeddingProtocol,
the Outgoing* models and ReceivedData are removed, as is the embedder
argument. process_history takes list[HistoryRecord].

### Features

- (events) Add engine event contract and HistoryRecord
- (engine) Run AInterviewer on an EngineHost [**breaking**]
- (embedding) Derive chunks from HistoryRecords and emit sections live
- (compat) Convert contract JSON between schema versions
- (worker) Add the runtime server
- (loggers) Choose the agent log destination with AINTERVIEWER_AGENTS_LOG

### Internal

- Add plan
- Update plan
- Pin runtime versions per project in plan
- (worker) Build, smoke-test and publish worker images
- Merge main (v0.7.2) into the versioned runtime
- (deps) Move polars and pyarrow to the analysis extra

## [0.7.2] - 2026-10-27

### Features

- (interview) Honour a question's can_skip

### Bug Fixes

- (conditions) Treat an unanswered question as unknown

### Internal

- Update any-llm-sdk

## [0.7.1] - 2026-10-22

### Features

- Discriminate survey item unions on `type`

### Internal

- Drop base attributes that subclasses narrow
- (deps) Upgrade dependencies

## [0.7.0] - 2026-10-22

### Features

- First pass at new safety agent implementation
- Second pass at security agent
- Update security agent system prompt
- Handle security override action

### Bug Fixes

- Keep a failed security check from ending the interview

### Internal

- Add raise_condition helper function
- Create InterviewControlCondition union type and reorder their creation

## [0.6.1] - 2026-10-15

### Features

- Tell clients which question a message asks

### Internal

- Format comments

## [0.6.0] - 2026-10-14

### Breaking Changes

- `translate_interview_guide`, `translate_consent` and
`translate_welcome` are replaced by `translate_missing`, which fills the
leaves a language lacks in a localized source.

### Features

- Multilingual guide source with text per language [**breaking**]

### Bug Fixes

- (deps) Depend on email-validator for EmailStr

## [0.5.0] - 2026-10-14

### Breaking Changes

- Fix typo in evaluate_condition task name
- `QuestionContext` serialises `question_index` instead of
`section` and `question`. A condition or reference whose `question_id` names
no question in the guide, or a guide with duplicate section or question ids,
now fails validation.
- A condition or reference without `question_id`, or a
`QuestionContext` in the `{section, question}` shape, fails validation.

### Features

- Stable ids on guide sections and questions; conditions and references by id [**breaking**]

### Bug Fixes

- Fix typo in evaluate_condition task name [**breaking**]
- Shuffle keeps questions a condition reads before the question it gates

### Internal

- Route preview and graduation trailers in the changelog
- Require question_id; migrate stored guides instead of upgrading on load [**breaking**]

## [0.4.4] - 2026-09-21

### Features

- Update models and implement per model configuration for alex endpoint
- Remove default extra traits from interviewees

### Internal

- Run test job on pull requests

## [0.4.3] - 2026-09-14

### Features

- Store condition_carrier message in task context

## [0.4.2] - 2026-09-11

### Breaking Changes

- Make interview_history the source of truth as to what message type is received

### Bug Fixes

- Make interview_history the source of truth as to what message type is received [**breaking**]

## [0.4.1] - 2026-09-10

### Features

- Move html escaping to frontend rendering, so raw text is stored. This makes downstream text analysis tasks easier.
- Add sections as embedding chunks

### Bug Fixes

- (build) Commit only release files in publish recipe

## [0.4.0] - 2026-09-01

### Breaking Changes

- Move vLLM model configuration to proxy repository

### Features

- Move vLLM model configuration to proxy repository [**breaking**]
- Add message embedding components and fix linting

### Internal

- Add citation information
- Fix broken url in readme
- Fix imports
- Update dev deps

## [0.3.5] - 2026-08-21

### Bug Fixes

- Inactive interview status is now set correctly

## [0.3.4] - 2026-08-20

### Internal

- Rename script

## [0.3.3] - 2026-08-19

### Internal

- Remove old unused safety module

## [0.3.2] - 2026-08-19

### Internal

- Remove default_language from InterviewConfig (now lives in backend table instead)

## [0.3.1] - 2026-08-14

### Bug Fixes

- When translating an interview guide, the condition trigger values are now also translated

## [0.3.0] - 2026-08-14

### Breaking Changes

- Move translation to BasePrompts.translation

### Features

- Improve json rendering in jinja templates
- Better traceback for language errors

### Bug Fixes

- Improve language localization prompts and answering agent
- Add explicit current_section=None to GuideAgentPrompts.generate_section_prompt template render
- Move translation to BasePrompts.translation [**breaking**]
- Improve whitespace in jinja templates

### Internal

- Document the dead language templates in the codebase

## [0.2.17] - 2026-08-12

### Internal

- Implement cliff and release note strategy
