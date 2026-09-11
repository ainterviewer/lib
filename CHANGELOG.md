# Changelog

Generated from [Conventional Commits](https://www.conventionalcommits.org) by
[git-cliff](https://git-cliff.org). Releases older than the earliest entry below
predate this changelog — see `git log` for their history.

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
