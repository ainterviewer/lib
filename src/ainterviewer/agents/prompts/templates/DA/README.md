# Danish templates (currently unused)

**Editing the templates in this directory has no effect.** The agents always render the
`EN/` templates, regardless of the interview language: `BasePrompts.__init__` hardcodes
the `PackageLoader` to `EN`, and the language the agent must actually speak is injected
into those English templates through the `translation` variable (see
`BasePrompts.translation`).

This means every code in `ainterviewer.constants.LANGUAGE_CODES` is supported, not just
the ones with a directory here.

The directory is kept because per-language templates may come back — in which case the
lookup must fall back to `EN` for any language without its own directory (see
`PROMPT_LANGS` in `get_prompts.py`). Until then, treat `EN/` as the only live templates,
and note that anything in here has drifted from it.
