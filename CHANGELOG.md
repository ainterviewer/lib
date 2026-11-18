# Changelog

Generated from [Conventional Commits](https://www.conventionalcommits.org) by
[git-cliff](https://git-cliff.org). Releases older than the earliest entry below
predate this changelog — see `git log` for their history.

## [0.8.3] - 2026-11-18

### Features

- (welcome) Make the contact email optional while the contact box is hidden

## [0.8.2] - 2026-11-18

### Bug Fixes

- (guide) Report a question with no text as a problem in the guide

## [0.8.1] - 2026-11-18

### Breaking Changes

- (interview) `PersistenceProtocol.insert_message` takes
`timed_message_id` in place of `timed`, and a stored message is read as timed
by its `timed_message_id`.

### Features

- (interview) Send each timed message once, and as a modal when asked [**breaking**]

## [0.8.0] - 2026-11-18

### Breaking Changes

- (conditions) `Question.conditions` is `list[Condition]` instead of
`Conditions | None`. `Conditions` is renamed `Condition`, and its
`conditions` field `blocks`; the old `Condition` is `ConditionBlock`.
The same goes for the localized models. `ConditionEvaluator`'s
`evaluate_conditions` is `evaluate_condition` and its
`evaluate_condition` is `evaluate_block`. Stored guides must be
migrated.

### Features

- (conditions) Let a question carry several conditions [**breaking**]

## [0.7.15] - 2026-11-17

### Bug Fixes

- (settings) Print no values in settings validation errors

## [0.7.14] - 2026-11-17

### Breaking Changes

- (settings) `settings.secrets` is gone; the provider keys are
`ainterviewer.settings.api_keys` (class `ApiKeys`), under the same
environment variables. A config file holding a secret, or a table key
nothing reads, now fails startup.

### Internal

- Work on each branch in its own worktree
- (settings) Read secrets from the environment only [**breaking**]

## [0.7.13] - 2026-11-17

### Breaking Changes

- (settings) A config that still sets `available_models` fails to load,
rather than being ignored and falling back to defaults under which nothing
could be published. Move its entries into `models` and `test_only_models`.

### Features

- (settings) Split the model list into models and test_only_models [**breaking**]

### Bug Fixes

- (lpm) Apply the per-model settings of alex: models

## [0.7.12] - 2026-11-17

### Features

- (interview) Tell the probing agent how many follow-ups remain
- (interview) Show the probing agent its earlier follow-ups to avoid repeats
- (survey) Shuffle a Likert grid's rows
- (interview) Leave suggested probes the respondent has already answered out of the probing prompt
- (survey) Add list and ranking items

### Internal

- (probing) Add a benchmark for probes that repeat what was said
- (probing) Classify answered suggestions with the production prompt
- (survey) Let each item describe its own answer

## [0.7.11] - 2026-11-14

### Bug Fixes

- (llm) Raise when a structured response has nothing parsed
- (interview) Clean up the prompt for questions generated mid-interview

## [0.7.10] - 2026-11-14

### Features

- (guide) Give generated guides, sections and questions the interviewer's question-writing guidance

### Bug Fixes

- (guide) Keep a generated question's probe time limit as a time limit
- (guide) Keep generated questions from repeating what earlier sections asked
- (guide) Tell the generator that the introduction and outro take no questions

### Internal

- (guide) Show the guide agent the guide as its text, as JSON

## [0.7.9] - 2026-11-13

### Features

- (guide) Shuffle the options of single and multiple choice items

## [0.7.8] - 2026-11-13

### Features

- (guide) Say which question a guide error is in

### Bug Fixes

- (guide) Name questions in condition order errors as the editor numbers them
- (interviews) Fill question references with each referenced answer
- (interviews) Check a resumed question's rules on its own answer
- (interviews) Still ask a question whose probes a rule skips
- (interviews) Check if the second question of a section was answered
- (interviews) Resume past messages that need no answer
- (interviews) Time timed messages from the start of the interview
- (interviews) Judge exhaustion with the section's description
- (interviews) Keep a skipped question's probes skipped on resume
- (interviews) Lose no probes when a reconnect cuts a timed message short

### Internal

- (interviews) Share what follows a main answer across paths
- (interviews) Type the stored message a resume settles on

## [0.7.7] - 2026-11-12

### Features

- (conditions) Compare Likert answers by their place on the scale

## [0.7.6] - 2026-11-11

### Features

- (interviews) Record why a completed interview ended

### Internal

- (deps) Bump any-llm-sdk to 1.50.0
- (deps) Refresh lockfile

## [0.7.5] - 2026-11-11

### Bug Fixes

- (llm) Prefer DeepInfra on OpenRouter only for gpt-oss

### Internal

- (llm) Default to gpt-oss through OpenRouter
- Remove unused agent config templates

## [0.7.4] - 2026-11-10

### Features

- (survey) Add a Likert grid item

### Bug Fixes

- (survey) Accept the ends of a number, slider or date range
- (survey) Keep a checkbox option that contains a comma whole

## [0.7.3] - 2026-11-04

### Features

- (llm) Allow a full base URL for self-hosted vLLM servers

### Internal

- Document commit types and the Preview: trailer

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
