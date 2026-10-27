"""Converting contract JSON between schema versions.

No breaking change has been made yet, so the registry is empty; most of these
run the machinery against a pretend schema history instead, written the way a
real change would be.
"""

import pytest

from ainterviewer import compat
from ainterviewer.compat import (
    MIN_SCHEMA_VERSION,
    SCHEMA_VERSION,
    Json,
    Kind,
    Unsupported,
    downgrade,
    downgrade_event,
    downgrades,
    upgrade,
    upgrade_event,
    upgrades,
)
from ainterviewer.events import SCHEMA_VERSION as EVENT_SCHEMA_VERSION
from ainterviewer.events import InputKind, InputRequested, outbound_event_adapter


def questions(guide: Json):
    """Every question in a guide, with its path."""
    for s, section in enumerate(guide["question_sections"]):
        for q, question in enumerate(section["questions"]):
            yield f"question_sections.{s}.questions.{q}", question


@pytest.fixture
def history(monkeypatch):
    """A library at schema version 3.

    1 → 2 renamed a likert item's `options` to `choices`: lossless.
    2 → 3 let an image have several primers: lossless down to one primer, not
    beyond.
    """
    monkeypatch.setattr(compat, "SCHEMA_VERSION", 3)
    monkeypatch.setattr(compat, "UPGRADES", {})
    monkeypatch.setattr(compat, "DOWNGRADES", {})

    # ── 1 → 2 ──

    @upgrades(Kind.SURVEY_ITEM, 1)
    def likert_choices(item: Json) -> Json:
        if item["type"] == "likert":
            item["choices"] = item.pop("options")
        return item

    @downgrades(Kind.SURVEY_ITEM, 1)
    def likert_options(item: Json) -> Json:
        if item["type"] == "likert":
            item["options"] = item.pop("choices")
        return item

    def in_guide(step):
        def guide_step(guide: Json) -> Json:
            for _, question in questions(guide):
                if question.get("survey_item"):
                    question["survey_item"] = step(question["survey_item"])
            return guide

        return guide_step

    def in_event(step):
        def event_step(event: Json) -> Json:
            if event["type"] == "message_emitted" and event.get("survey_item"):
                event["survey_item"] = step(event["survey_item"])
            return event

        return event_step

    upgrades(Kind.INTERVIEW_GUIDE, 1)(in_guide(likert_choices))
    downgrades(Kind.INTERVIEW_GUIDE, 1)(in_guide(likert_options))
    upgrades(Kind.EVENT, 1)(in_event(likert_choices))
    downgrades(Kind.EVENT, 1)(in_event(likert_options))

    # ── 2 → 3 ──

    @upgrades(Kind.IMAGE, 2)
    def primers(image: Json) -> Json:
        primer = image.pop("primer")
        image["primers"] = [primer] if primer else []
        return image

    @downgrades(Kind.IMAGE, 2)
    def primer(image: Json) -> Json:
        match image.pop("primers"):
            case []:
                image["primer"] = None
            case [one]:
                image["primer"] = one
            case _:
                raise Unsupported(["primers"])
        return image

    def images_in_guide(step):
        def guide_step(guide: Json) -> Json:
            refused: list[Unsupported] = []
            for path, question in questions(guide):
                if question.get("image"):
                    try:
                        question["image"] = step(question["image"])
                    except Unsupported as error:
                        refused.append(error.at(f"{path}.image"))
            if refused:
                raise Unsupported(p for error in refused for p in error.paths)
            return guide

        return guide_step

    upgrades(Kind.INTERVIEW_GUIDE, 2)(images_in_guide(primers))
    downgrades(Kind.INTERVIEW_GUIDE, 2)(images_in_guide(primer))


def guide_v1(*primers: str | None) -> Json:
    return {
        "question_sections": [
            {
                "description": "s",
                "questions": [
                    {
                        "main_question": "Agree?",
                        "survey_item": {"type": "likert", "options": ["No", "Yes"]},
                    },
                    *(
                        {
                            "main_question": "Look",
                            "image": {"name": "a.png", "primer": primer},
                        }
                        for primer in primers
                    ),
                ],
            }
        ]
    }


# ── upgrading ────────────────────────────────────────────────────────────────


def test_upgrades_chain_from_the_given_version(history):
    guide = upgrade(Kind.INTERVIEW_GUIDE, guide_v1("Hi"), from_version=1)

    agree, look = guide["question_sections"][0]["questions"]
    assert agree["survey_item"] == {"type": "likert", "choices": ["No", "Yes"]}
    assert look["image"] == {"name": "a.png", "primers": ["Hi"]}


def test_a_kind_no_version_touched_comes_back_as_it_was(history):
    config = {"with_consent": True}

    assert upgrade(Kind.INTERVIEW_CONFIG, config, from_version=1) == config
    assert downgrade(Kind.INTERVIEW_CONFIG, config, to_version=1) == config


def test_a_part_stored_on_its_own_converts_by_itself(history):
    # The survey item on a stored message, outside any guide.
    item = {"type": "likert", "options": ["No", "Yes"]}

    assert upgrade(Kind.SURVEY_ITEM, item, from_version=1) == {
        "type": "likert",
        "choices": ["No", "Yes"],
    }


def test_events_are_upgraded_from_the_version_they_carry(history):
    event = {
        "type": "message_emitted",
        "schema_version": 1,
        "survey_item": {"type": "likert", "options": ["No"]},
    }

    assert upgrade_event(event) == {
        "type": "message_emitted",
        "schema_version": 3,
        "survey_item": {"type": "likert", "choices": ["No"]},
    }


# ── downgrading ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("primers", [(), ("Hi",), (None,)])
def test_a_lossless_change_round_trips(history, primers):
    guide = guide_v1(*primers)

    current = upgrade(Kind.INTERVIEW_GUIDE, guide, from_version=1)

    assert downgrade(Kind.INTERVIEW_GUIDE, current, to_version=1) == guide


def test_downgrades_stop_at_the_given_version(history):
    current = upgrade(Kind.INTERVIEW_GUIDE, guide_v1("Hi"), from_version=1)

    agree, look = downgrade(Kind.INTERVIEW_GUIDE, current, to_version=2)[
        "question_sections"
    ][0]["questions"]

    assert agree["survey_item"]["choices"] == ["No", "Yes"]
    assert look["image"]["primer"] == "Hi"


def test_what_an_older_version_cannot_express_is_refused_by_path(history):
    current = upgrade(Kind.INTERVIEW_GUIDE, guide_v1("Hi", None), from_version=1)
    current["question_sections"][0]["questions"][1]["image"]["primers"] = ["a", "b"]
    current["question_sections"][0]["questions"][2]["image"]["primers"] = ["c", "d"]

    with pytest.raises(Unsupported) as refused:
        downgrade(Kind.INTERVIEW_GUIDE, current, to_version=1)

    assert refused.value.paths == [
        "question_sections.0.questions.1.image.primers",
        "question_sections.0.questions.2.image.primers",
    ]


def test_events_are_downgraded_to_the_version_asked_for(history):
    event = {
        "type": "message_emitted",
        "schema_version": 3,
        "survey_item": {"type": "likert", "choices": ["No"]},
    }

    assert downgrade_event(event, to_version=1) == {
        "type": "message_emitted",
        "schema_version": 1,
        "survey_item": {"type": "likert", "options": ["No"]},
    }


# ── the machinery ────────────────────────────────────────────────────────────


def test_the_input_is_left_alone(history):
    guide = guide_v1("Hi")
    current = upgrade(Kind.INTERVIEW_GUIDE, guide, from_version=1)
    frozen = repr(current)

    downgrade(Kind.INTERVIEW_GUIDE, current, to_version=1)

    assert guide == guide_v1("Hi")
    assert repr(current) == frozen


@pytest.mark.parametrize("version", [0, 4])
def test_unknown_versions_are_refused(history, version):
    with pytest.raises(ValueError, match="schema version"):
        upgrade(Kind.EVENT, {}, from_version=version)
    with pytest.raises(ValueError, match="schema version"):
        downgrade(Kind.EVENT, {}, to_version=version)


def test_pruned_versions_are_refused(history, monkeypatch):
    monkeypatch.setattr(compat, "MIN_SCHEMA_VERSION", 2)

    with pytest.raises(ValueError, match="converts 2 to 3"):
        upgrade(Kind.INTERVIEW_GUIDE, guide_v1(), from_version=1)


def test_a_step_is_registered_once(history):
    with pytest.raises(ValueError, match="exists"):
        upgrades(Kind.SURVEY_ITEM, 1)(lambda item: item)


def test_a_refusal_found_inside_a_value_is_placed_in_it():
    assert Unsupported(["primers", ""]).at("questions.0.image").paths == [
        "questions.0.image.primers",
        "questions.0.image",
    ]


# ── the real registry ────────────────────────────────────────────────────────


def test_every_change_goes_both_ways():
    assert compat.UPGRADES.keys() == compat.DOWNGRADES.keys()


def test_every_step_is_within_the_versions_converted():
    assert all(
        MIN_SCHEMA_VERSION <= version < SCHEMA_VERSION for _, version in compat.UPGRADES
    )


def test_events_carry_the_contract_version():
    assert EVENT_SCHEMA_VERSION == SCHEMA_VERSION


def test_a_current_event_converts_to_itself():
    event = InputRequested(message_id=1, expects=InputKind.FREE_TEXT)
    data = event.model_dump(mode="json")

    assert outbound_event_adapter.validate_python(upgrade_event(data)) == event
    assert downgrade_event(data, to_version=SCHEMA_VERSION) == data
