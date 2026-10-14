"""The multilingual source and its renderings.

The structure is stored once and only text per language, so these pin that a
rendering is exactly the guide that language would have had on its own, that
building the source from renderings never invents text, and that a condition
on a choice holds in every language because it names the option, not its
wording.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from ainterviewer.interview_guides import (
    InterviewGuide,
    Question,
    QuestionSection,
    RadioItem,
    TimedMessage,
    translate,
)
from ainterviewer.interview_guides import localized as loc
from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    ConditionEvaluator,
    Conditions,
    QuestionContext,
)
from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.localized import (
    LocalizedCondition,
    LocalizedConditionEvaluation,
    LocalizedConditions,
    LocalizedInterviewGuide,
    LocalizedOption,
    LocalizedQuestion,
    LocalizedQuestionSection,
    LocalizedRadioItem,
    LocalizedWelcome,
)
from ainterviewer.interview_guides.media import Image
from ainterviewer.interview_guides.references import Reference
from ainterviewer.interview_guides.survey_items import (
    CheckboxItem,
    LikertItem,
    SliderItem,
)
from ainterviewer.interview_guides.types import ConditionAction, ConditionTrigger


def _guide(
    language: str, *, often: str, never: str, probe: str, ask: str
) -> InterviewGuide:
    """One language's rendering: a radio question, and one gated on it."""
    radio = Question(
        main_question=ask,
        survey_item=RadioItem(options=[often, never]),
        id=_ids["radio"],
    )
    gated = Question(
        main_question=f"{language} why?",
        probes=[probe],
        id=_ids["gated"],
        conditions=Conditions(
            action=ConditionAction.SKIP_QUESTION,
            conditions=[
                Condition(
                    question_context=QuestionContext(
                        question_index=(0, 0), question_id=radio.id
                    ),
                    evaluation=[ConditionEvaluation(trigger_value=never)],
                )
            ],
        ),
    )
    return InterviewGuide(
        framing=f"{language} framing",
        introduction=f"{language} hello",
        outro=f"{language} bye",
        question_sections=[
            QuestionSection[Question](
                id=_ids["section"],
                description=f"{language} section",
                questions=[radio, gated],
            )
        ],
        timed_messages=[TimedMessage(message=f"{language} hurry", time=60)],
    )


_ids = {"radio": uuid4(), "gated": uuid4(), "section": uuid4()}
EN = _guide("EN", often="Often", never="Never", probe="Tell me more", ask="How often?")
DA = _guide("DA", often="Ofte", never="Aldrig", probe="Fortæl mere", ask="Hvor ofte?")


def _dump(guide: InterviewGuide) -> dict:
    return guide.model_dump(mode="json")


class TestFieldsMirrorTheRenderedModels:
    """A field added to a rendered model must be added to its source twin, or
    `render` silently drops it."""

    @pytest.mark.parametrize(
        ("source", "rendered", "derived"),
        [
            (LocalizedQuestion, Question, {"index"}),
            (LocalizedQuestionSection, QuestionSection, set()),
            (LocalizedInterviewGuide, InterviewGuide, set()),
            (loc.LocalizedTimedMessage, TimedMessage, set()),
            (loc.LocalizedImage, Image, set()),
            (LocalizedRadioItem, RadioItem, set()),
            (loc.LocalizedCheckboxItem, CheckboxItem, set()),
            (loc.LocalizedLikertItem, LikertItem, set()),
            (loc.LocalizedSliderItem, SliderItem, set()),
            (LocalizedCondition, Condition, set()),
            (loc.LocalizedConsent, Consent, set()),
            (LocalizedWelcome, Welcome, set()),
        ],
    )
    def test_same_fields(self, source, rendered, derived):
        assert set(source.model_fields) == set(rendered.model_fields) - derived

    def test_evaluation_adds_only_the_option(self):
        assert set(LocalizedConditionEvaluation.model_fields) == set(
            ConditionEvaluation.model_fields
        ) | {"option_id"}


class TestRoundTrip:
    def test_each_language_renders_back_to_itself(self):
        report: list[str] = []
        source = LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": DA}, structure="EN", report=report
        )

        assert report == []
        assert _dump(source.render("EN")) == _dump(EN)
        assert _dump(source.render("DA")) == _dump(DA)

    def test_survives_json(self):
        source = LocalizedInterviewGuide.from_rendered({"EN": EN}, structure="EN")
        again = LocalizedInterviewGuide.model_validate_json(source.model_dump_json())
        assert again == source

    def test_choice_trigger_names_the_option(self):
        source = LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": DA}, structure="EN"
        )
        radio, gated = source.question_sections[0].questions
        assert isinstance(radio.survey_item, LocalizedRadioItem)
        never = radio.survey_item.options[1]
        assert never.text == {"EN": "Never", "DA": "Aldrig"}

        assert gated.conditions is not None
        (evaluation,) = gated.conditions.conditions[0].evaluation
        assert evaluation.option_id == never.id
        assert evaluation.trigger_value is None


class TestFromRenderedNeverInvents:
    def test_missing_question_is_reported_and_left_untranslated(self):
        da = DA.model_copy(deep=True)
        da.question_sections[0].questions.pop(1)
        report: list[str] = []

        source = LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": da}, structure="EN", report=report
        )

        assert any("questions[1]" in line and line.startswith("DA") for line in report)
        assert "question_sections[0].questions[1].main_question" in source.missing("DA")
        # The structure is the default's: DA still asks the question, in EN.
        rendered = source.render("DA", fallback=["EN"])
        assert rendered.question_sections[0].questions[1].main_question == "EN why?"

    def test_probe_count_mismatch_keeps_the_aligned_ones(self):
        da = DA.model_copy(deep=True)
        da.question_sections[0].questions[1].probes = []
        en = EN.model_copy(deep=True)
        en.question_sections[0].questions[1].probes = ["One", "Two"]
        report: list[str] = []

        source = LocalizedInterviewGuide.from_rendered(
            {"EN": en, "DA": da}, structure="EN", report=report
        )

        probes = source.question_sections[0].questions[1].probes
        assert probes == [{"EN": "One"}, {"EN": "Two"}]
        assert any("probes" in line for line in report)

    def test_trigger_that_is_no_option_stays_a_pattern(self):
        en = EN.model_copy(deep=True)
        gated = en.question_sections[0].questions[1]
        assert gated.conditions is not None
        gated.conditions.conditions[0].evaluation[0].trigger_value = "Sometimes"
        report: list[str] = []

        source = LocalizedInterviewGuide.from_rendered(
            {"EN": en}, structure="EN", report=report
        )

        conditions = source.question_sections[0].questions[1].conditions
        assert conditions is not None
        evaluation = conditions.conditions[0].evaluation[0]
        assert (evaluation.option_id, evaluation.trigger_value) == (None, "Sometimes")
        assert any("Sometimes" in line for line in report)

    def test_optional_text_left_unset_stays_unset(self):
        source = LocalizedInterviewGuide.from_rendered({"EN": EN}, structure="EN")
        assert source.question_sections[0].questions[0].description is None
        assert source.alt_outro is None


class TestRender:
    @pytest.fixture
    def source(self) -> LocalizedInterviewGuide:
        return LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": DA}, structure="EN"
        )

    def test_missing_leaf_falls_back(self, source):
        source.question_sections[0].questions[1].main_question.pop("DA")

        assert source.missing("DA") == [
            "question_sections[0].questions[1].main_question"
        ]
        assert source.missing("EN") == []
        assert (
            source.render("DA", fallback=["EN"])
            .question_sections[0]
            .questions[1]
            .main_question
            == "EN why?"
        )
        assert source.render("DA").question_sections[0].questions[1].main_question == ""

    @pytest.mark.anyio
    async def test_condition_holds_in_every_language(self, source):
        evaluator = ConditionEvaluator()
        for language, answer in (("EN", "Never"), ("DA", "Aldrig")):
            gated = source.render(language).question_sections[0].questions[1]
            assert gated.conditions is not None
            assert await evaluator.evaluate_conditions([answer], gated.conditions)

    @pytest.mark.anyio
    async def test_rewording_an_option_moves_the_trigger_with_it(self, source):
        radio = source.question_sections[0].questions[0]
        assert isinstance(radio.survey_item, LocalizedRadioItem)
        radio.survey_item.options[1].text["DA"] = "Aldrig nogensinde"

        gated = source.render("DA").question_sections[0].questions[1]
        assert gated.conditions is not None
        assert await ConditionEvaluator().evaluate_conditions(
            ["Aldrig nogensinde"], gated.conditions
        )


class TestValidation:
    def _with_condition(self, option_id) -> dict:
        radio = LocalizedQuestion(
            main_question={"EN": "How often?"},
            survey_item=LocalizedRadioItem(
                options=[LocalizedOption(text={"EN": "Never"})]
            ),
        )
        other = LocalizedQuestion(
            main_question={"EN": "Other"},
            survey_item=LocalizedRadioItem(
                options=[LocalizedOption(text={"EN": "Elsewhere"})]
            ),
        )
        gated = LocalizedQuestion(
            main_question={"EN": "Why?"},
            conditions=LocalizedConditions(
                action=ConditionAction.SKIP_QUESTION,
                conditions=[
                    LocalizedCondition(
                        question_context=QuestionContext(
                            question_index=(0, 0), question_id=radio.id
                        ),
                        evaluation=[
                            LocalizedConditionEvaluation(
                                option_id=option_id(radio, other)
                            )
                        ],
                    )
                ],
            ),
        )
        return {
            "question_sections": [
                {"questions": [q.model_dump() for q in (radio, other, gated)]}
            ]
        }

    def test_option_of_the_read_question(self):
        data = self._with_condition(lambda radio, _: radio.survey_item.options[0].id)
        LocalizedInterviewGuide.model_validate(data)

    def test_option_of_another_question_is_rejected(self):
        data = self._with_condition(lambda _, other: other.survey_item.options[0].id)
        with pytest.raises(ValidationError, match="not one of the question it reads"):
            LocalizedInterviewGuide.model_validate(data)

    def test_one_trigger_exactly(self):
        with pytest.raises(ValidationError, match="exactly one"):
            LocalizedConditionEvaluation(option_id=uuid4(), trigger_value="x")
        with pytest.raises(ValidationError, match="exactly one"):
            LocalizedConditionEvaluation()

    def test_classification_cannot_name_an_option(self):
        with pytest.raises(ValidationError, match="classification"):
            LocalizedCondition(
                question_context=QuestionContext(
                    question_index=(0, 0), question_id=uuid4()
                ),
                trigger_type=ConditionTrigger.CLASSIFICATION,
                evaluation=[LocalizedConditionEvaluation(option_id=uuid4())],
            )

    def test_every_language_is_validated(self):
        target = LocalizedQuestion(
            main_question={"EN": "Pick", "DA": "Vælg"},
            survey_item=LocalizedRadioItem(
                options=[LocalizedOption(text={"EN": "A", "DA": "A"})]
            ),
        )
        referring = LocalizedQuestion(
            main_question={"EN": "You said {}", "DA": "Du sagde"},
            references=[Reference(question_index=(0, 0), question_id=target.id)],
        )
        with pytest.raises(ValidationError, match="DA"):
            LocalizedInterviewGuide(
                question_sections=[
                    LocalizedQuestionSection(questions=[target, referring])
                ]
            )

    def test_untranslated_language_still_validates(self):
        # DA has no main question for the referring question yet: validation
        # falls back to EN's text rather than rejecting a draft in progress.
        target = LocalizedQuestion(main_question={"EN": "Pick", "DA": "Vælg"})
        referring = LocalizedQuestion(
            main_question={"EN": "You said {}"},
            references=[Reference(question_index=(0, 0), question_id=target.id)],
        )
        LocalizedInterviewGuide(
            question_sections=[LocalizedQuestionSection(questions=[target, referring])]
        )

    def test_survey_item_changes_the_defaults_like_question(self):
        data = {
            "main_question": {"EN": "Pick"},
            "survey_item": {"type": "radio", "options": [{"text": {"EN": "A"}}]},
        }
        source = LocalizedQuestion.model_validate(data)
        rendered = Question.model_validate(
            {
                "main_question": "Pick",
                "survey_item": {"type": "radio", "options": ["A"]},
            }
        )
        assert source.max_probes_n == rendered.max_probes_n == 0


class TestWelcome:
    def test_shared_fields_come_from_the_structure(self):
        en = Welcome(title="Hi", text="Welcome", email="a@example.org", with_id=False)
        da = Welcome(title="Hej", text="Velkommen", email="b@example.org")

        source = LocalizedWelcome.from_rendered({"EN": en, "DA": da}, structure="EN")

        assert source.render("DA") == Welcome(
            title="Hej", text="Velkommen", email="a@example.org", with_id=False
        )


class TestTranslateMissing:
    @pytest.mark.anyio
    async def test_translates_only_the_missing_leaves(self, monkeypatch):
        sent: list[dict[str, str]] = []

        async def fake(strings, *, target_language, model, context):
            sent.append(strings)
            return {
                path: f"[{target_language}] {text}" for path, text in strings.items()
            }

        monkeypatch.setattr(translate, "_translate_strings", fake)
        source = LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": DA}, structure="EN"
        )
        source.question_sections[0].questions[1].main_question.pop("DA")
        source.outro = {"EN": "bye"}

        result = await translate.translate_missing(
            source, target="DA", source="EN", model="test"
        )

        assert sorted(path for batch in sent for path in batch) == [
            "outro",
            "question_sections[0].questions[1].main_question",
        ]
        assert result.missing("DA") == []
        assert result.outro == {"EN": "bye", "DA": "[Danish] bye"}
        # The input is not touched, and what DA had is kept.
        assert "DA" not in source.question_sections[0].questions[1].main_question
        assert result.introduction["DA"] == "DA hello"

    @pytest.mark.anyio
    async def test_nothing_missing_calls_nothing(self, monkeypatch):
        async def fail(*args, **kwargs):
            raise AssertionError("no translation expected")

        monkeypatch.setattr(translate, "_translate_strings", fail)
        source = LocalizedInterviewGuide.from_rendered(
            {"EN": EN, "DA": DA}, structure="EN"
        )
        assert (
            await translate.translate_missing(
                source, target="DA", source="EN", model="test"
            )
            == source
        )
