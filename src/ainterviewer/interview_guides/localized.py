"""A study's multilingual source: one shared structure, text per language.

A project in several languages asks the same questions in the same order with
the same survey items, conditions and settings everywhere; only the wording
differs. So the structure is stored once, and every piece of respondent- or
model-facing text is a `LocalizedText` -- a `{language: text}` map -- in place
of a plain string. A missing key is an untranslated leaf.

`InterviewGuide`, `Consent` and `Welcome` stay the *rendered* form: the
structure plus one language, which is what the interview loop reads.
`render(language, fallback)` produces them, and `from_rendered` builds the
source back from one rendering per language (used to import guides, merge
generated content and migrate the old per-language storage).

Conditions are shared like the rest of the structure. One that matches a
choice answer names the option by id (`LocalizedConditionEvaluation.option_id`)
rather than by its text, so it holds in every language and survives the option
being reworded or reordered; rendering writes that option's text into
`trigger_value`, which is what the evaluator compares answers against.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, EmailStr, Field, ValidationError, model_validator

from ainterviewer.interview_guides.conditions import (
    Condition,
    ConditionEvaluation,
    Conditions,
    QuestionContext,
)
from ainterviewer.interview_guides.extra import Consent, Welcome
from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewMessage,
    TimedMessage,
)
from ainterviewer.interview_guides.media import Image, MediaType
from ainterviewer.interview_guides.questions import Question
from ainterviewer.interview_guides.references import Reference
from ainterviewer.interview_guides.sections import GeneratedQuestions, QuestionSection
from ainterviewer.interview_guides.survey_items import (
    CheckboxItem,
    DateItem,
    DatetimeItem,
    LikertItem,
    NumberItem,
    RadioItem,
    SliderItem,
    SurveyItem,
    SurveyItemBase,
    TimeItem,
)
from ainterviewer.interview_guides.types import (
    ConditionAction,
    ConditionTrigger,
    ProbingContext,
)
from ainterviewer.types import LanguageCode

type LocalizedText = dict[LanguageCode, str]
"""One piece of text in every language it has been written or translated into."""


def _pick_localized_text(text: LocalizedText, languages: Sequence[str]) -> str:
    """The text in the first of `languages` that has any, else empty."""
    for language in languages:
        if value := text.get(language):
            return value
    return ""


def _pick_optional(text: LocalizedText | None, languages: Sequence[str]) -> str | None:
    """`_pick` for an optional field: unset stays unset."""
    return None if text is None else _pick_localized_text(text, languages)


def _missing(text: LocalizedText, language: str) -> bool:
    """Untranslated: written in some language, but not in `language`."""
    return not text.get(language) and any(text.values())


class _Merge:
    """State shared while building a source from renderings: the problems
    found, and the options conditions may name."""

    def __init__(self, structure: str) -> None:
        self.structure = structure
        self.report: list[str] = []
        # Question id -> its choice item's options, for condition triggers.
        self.options: dict[UUID, list[LocalizedOption]] = {}

    def text(self, values: Mapping[str, str | None]) -> LocalizedText | None:
        """Gather one leaf across languages. None if the structure language
        has no value at all (an optional field left unset)."""
        if values.get(self.structure) is None:
            return None
        return {language: value for language, value in values.items() if value}

    def texts(
        self, values: Mapping[str, list[str] | None], where: str
    ) -> list[LocalizedText] | None:
        """Gather a list leaf (probes, wordings) by position."""
        base = values.get(self.structure)
        if base is None:
            return None
        for language, items in values.items():
            if language != self.structure and len(items or []) != len(base):
                self.report.append(
                    f"{language}: {where} has {len(items or [])} entries against "
                    f"{len(base)}; the rest are left untranslated"
                )
        return [
            {
                language: items[n]
                for language, items in values.items()
                if items and n < len(items) and items[n]
            }
            for n in range(len(base))
        ]


# -- Survey items --------------------------------------------------------------


class LocalizedOption(BaseModel):
    """One answer option of a choice item."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Stable identity of the option, shared by every translation of it; conditions name it.",
    )
    text: LocalizedText = Field(default_factory=dict)


def _merge_options(
    items: Mapping[str, SurveyItem | None], merge: _Merge, where: str
) -> list[LocalizedOption]:
    options = merge.texts(
        {
            language: getattr(item, "options", None)
            for language, item in items.items()
            if item is not None
        },
        f"{where}.survey_item.options",
    )
    return [LocalizedOption(text=text) for text in options or []]


class LocalizedRadioItem(SurveyItemBase):
    type: Literal["radio"] = "radio"
    options: list[LocalizedOption]
    with_other: bool = False


class LocalizedCheckboxItem(SurveyItemBase):
    type: Literal["checkbox"] = "checkbox"
    options: list[LocalizedOption]
    with_other: bool = False


class LocalizedLikertItem(SurveyItemBase):
    type: Literal["likert"] = "likert"
    options: list[LocalizedOption]
    ui: Literal["radio", "slider"] = "radio"


class LocalizedSliderItem(SurveyItemBase):
    type: Literal["slider"] = "slider"
    min: int | float
    max: int | float
    min_label: LocalizedText | None = None
    max_label: LocalizedText | None = None
    step: int | float | None = 1


LocalizedSurveyItem = Annotated[
    LocalizedRadioItem
    | LocalizedCheckboxItem
    | LocalizedLikertItem
    | LocalizedSliderItem
    | NumberItem
    | DateItem
    | DatetimeItem
    | TimeItem,
    Field(discriminator="type"),
]

type _LocalizedChoice = LocalizedRadioItem | LocalizedCheckboxItem | LocalizedLikertItem
type _Choice = RadioItem | CheckboxItem | LikertItem

_CHOICE_ITEMS: dict[type[_LocalizedChoice], type[_Choice]] = {
    LocalizedRadioItem: RadioItem,
    LocalizedCheckboxItem: CheckboxItem,
    LocalizedLikertItem: LikertItem,
}
_LOCALIZED_CHOICE_ITEMS: dict[type[_Choice], type[_LocalizedChoice]] = {
    rendered: local for local, rendered in _CHOICE_ITEMS.items()
}


def _render_survey_item(
    item: LocalizedSurveyItem, languages: Sequence[str]
) -> SurveyItem:
    match item:
        case LocalizedRadioItem() | LocalizedCheckboxItem() | LocalizedLikertItem():
            return _CHOICE_ITEMS[type(item)].model_validate(
                {
                    **item.model_dump(exclude={"options"}),
                    "options": [
                        _pick_localized_text(option.text, languages)
                        for option in item.options
                    ],
                }
            )
        case LocalizedSliderItem():
            return SliderItem.model_validate(
                {
                    **item.model_dump(exclude={"min_label", "max_label"}),
                    "min_label": _pick_optional(item.min_label, languages),
                    "max_label": _pick_optional(item.max_label, languages),
                }
            )
        case _:
            return item


def _merge_survey_item(
    items: Mapping[str, SurveyItem | None], merge: _Merge, where: str
) -> LocalizedSurveyItem | None:
    base = items.get(merge.structure)
    if base is None:
        return None
    # A language whose item is of another type has no text to give this one.
    same = {
        language: item
        for language, item in items.items()
        if item is not None and item.type == base.type
    }
    for language in items.keys() - same.keys():
        merge.report.append(
            f"{language}: {where} has no {base.type} survey item like "
            f"{merge.structure}'s; its text is left untranslated"
        )

    match base:
        case RadioItem() | CheckboxItem() | LikertItem():
            return _LOCALIZED_CHOICE_ITEMS[type(base)].model_validate(
                {
                    **base.model_dump(exclude={"options"}),
                    "options": _merge_options(same, merge, where),
                }
            )
        case SliderItem():
            return LocalizedSliderItem.model_validate(
                {
                    **base.model_dump(exclude={"min_label", "max_label"}),
                    "min_label": merge.text(
                        {
                            language: getattr(item, "min_label", None)
                            for language, item in same.items()
                        }
                    ),
                    "max_label": merge.text(
                        {
                            language: getattr(item, "max_label", None)
                            for language, item in same.items()
                        }
                    ),
                }
            )
        case _:
            return base


# -- Media ---------------------------------------------------------------------


class LocalizedImage(BaseModel):
    """An image to show the interviewee. The file is shared; what is said
    about it is per language."""

    type: MediaType = "image"
    name: str = Field(description="The filename")
    data: str | None = Field(None, repr=False)
    primer: LocalizedText | None = None
    description: LocalizedText = Field(default_factory=dict)
    alt: LocalizedText = Field(default_factory=dict)

    def render(self, languages: Sequence[str]) -> Image:
        return Image(
            name=self.name,
            data=self.data,
            primer=_pick_optional(self.primer, languages),
            description=_pick_localized_text(self.description, languages),
            alt=_pick_localized_text(self.alt, languages),
        )


def _merge_image(
    images: Mapping[str, Image | None], merge: _Merge, where: str
) -> LocalizedImage | None:
    base = images.get(merge.structure)
    if base is None:
        return None
    present = {language: image for language, image in images.items() if image}
    return LocalizedImage(
        name=base.name,
        data=base.data,
        primer=merge.text(
            {language: image.primer for language, image in present.items()}
        ),
        description=merge.text(
            {language: image.description for language, image in present.items()},
        )
        or {},
        alt=merge.text({language: image.alt for language, image in present.items()})
        or {},
    )


# -- Conditions ----------------------------------------------------------------


class LocalizedConditionEvaluation(BaseModel):
    """`ConditionEvaluation` with the trigger named by option where it is one.

    Exactly one of `option_id` and `trigger_value` is set. A classification
    instruction, a pattern or a number to compare against is a `trigger_value`
    shared by every language: the classifier reads answers in any language,
    and a number is a number.
    """

    option_id: UUID | None = Field(
        None,
        description="The option of the read question's choice item that this evaluation matches. Rendered into `trigger_value` as the option's text.",
    )
    trigger_value: str | int | float | None = Field(
        None,
        description="The pattern to match, value to compare against or classification instruction, when the evaluation does not name an option.",
    )
    comparison_operator: Literal["==", "<", "<=", ">", ">="] = "=="
    combine_next: Literal["AND", "OR"] | None = None

    @model_validator(mode="after")
    def _one_trigger(self):
        if (self.option_id is None) == (self.trigger_value is None):
            raise ValueError("Set exactly one of option_id and trigger_value")
        if self.option_id is not None and self.comparison_operator != "==":
            raise ValueError("An option can only be matched with '=='")
        return self


class LocalizedCondition(BaseModel):
    question_context: QuestionContext
    evaluation: list[LocalizedConditionEvaluation]
    negated: bool = False
    trigger_type: ConditionTrigger = ConditionTrigger.MATCH
    combine_next: Literal["AND", "OR"] | None = None

    @model_validator(mode="after")
    def _options_only_match(self):
        if self.trigger_type == ConditionTrigger.CLASSIFICATION and any(
            evaluation.option_id for evaluation in self.evaluation
        ):
            raise ValueError("A classification condition cannot name an option")
        return self


class LocalizedConditions(BaseModel):
    conditions: list[LocalizedCondition]
    action: ConditionAction

    def render(
        self, options: Mapping[UUID, LocalizedText], languages: Sequence[str]
    ) -> Conditions:
        def trigger(evaluation: LocalizedConditionEvaluation) -> str | int | float:
            if evaluation.option_id is None:
                assert evaluation.trigger_value is not None
                return evaluation.trigger_value
            if evaluation.option_id not in options:
                raise ValueError(
                    f"A condition names option {evaluation.option_id}, which does not exist"
                )
            return _pick_localized_text(options[evaluation.option_id], languages)

        return Conditions(
            action=self.action,
            conditions=[
                Condition(
                    **condition.model_dump(exclude={"evaluation"}),
                    evaluation=[
                        ConditionEvaluation(
                            **evaluation.model_dump(
                                exclude={"option_id", "trigger_value"}
                            ),
                            trigger_value=trigger(evaluation),
                        )
                        for evaluation in condition.evaluation
                    ],
                )
                for condition in self.conditions
            ],
        )

    @classmethod
    def from_rendered(
        cls, conditions: Conditions, merge: _Merge, where: str
    ) -> LocalizedConditions:
        """Take the structure language's conditions, naming an option wherever
        a match trigger is one of the read question's options. Other
        languages' triggers are translations of the same option and are not
        read."""

        def evaluation(
            condition: Condition, evaluation: ConditionEvaluation
        ) -> LocalizedConditionEvaluation:
            options = merge.options.get(condition.question_context.question_id)
            value = evaluation.trigger_value
            if (
                options
                and condition.trigger_type == ConditionTrigger.MATCH
                and evaluation.comparison_operator == "=="
            ):
                key = str(value).strip().casefold()
                for option in options:
                    text = option.text.get(merge.structure, "")
                    if text.strip().casefold() == key:
                        return LocalizedConditionEvaluation(
                            option_id=option.id,
                            combine_next=evaluation.combine_next,
                        )
                merge.report.append(
                    f"{where}: trigger {value!r} is none of the read question's "
                    "options; kept as a pattern shared by every language"
                )
            return LocalizedConditionEvaluation(**evaluation.model_dump())

        return cls(
            action=conditions.action,
            conditions=[
                LocalizedCondition(
                    **condition.model_dump(exclude={"evaluation"}),
                    evaluation=[evaluation(condition, e) for e in condition.evaluation],
                )
                for condition in conditions.conditions
            ],
        )


# -- Questions and sections ----------------------------------------------------

# Fields rendered by hand; every other field of `Question` is shared and copied.
_QUESTION_TEXT: set[str] = {
    "main_question",
    "description",
    "probes",
    "alternative_main_questions",
    "survey_item",
    "image",
    "conditions",
}
# Derived on the rendered guide (`InterviewGuide.index_questions`).
_QUESTION_DERIVED: set[str] = {"index"}


class LocalizedQuestion(BaseModel):
    """`Question` with its text per language. See `Question` for the fields."""

    id: UUID = Field(default_factory=uuid4)
    main_question: LocalizedText = Field(default_factory=dict)
    description: LocalizedText | None = None
    probes: list[LocalizedText] | None = None
    alternative_main_questions: list[LocalizedText] | None = None
    max_probes_n: int | None = Field(default=3, ge=0)
    max_probes_time: float | None = Field(None, gt=0)
    survey_item: LocalizedSurveyItem | None = None
    can_answer: bool = True
    check_if_answered: bool = False
    check_if_exhausted: bool = False
    can_skip: bool = True
    shuffle: bool = False
    create_segue: bool = False
    exclude_from_history: bool = False
    references: list[Reference] | None = None
    image: LocalizedImage | None = None
    user_image: bool = False
    conditions: LocalizedConditions | None = None
    probing_context: ProbingContext | None = None

    @model_validator(mode="before")
    @classmethod
    def _survey_defaults(cls, data):
        # Mirrors `Question.__init__`: a survey item changes these defaults.
        if isinstance(data, dict) and data.get("survey_item") is not None:
            data = {
                "max_probes_n": 0,
                "create_segue": False,
                "check_if_answered": False,
                **data,
            }
        return data

    def render(
        self, options: Mapping[UUID, LocalizedText], languages: Sequence[str]
    ) -> Question:
        return Question.model_validate(
            {
                **self.model_dump(exclude=_QUESTION_TEXT),
                "main_question": _pick_localized_text(self.main_question, languages),
                "description": _pick_optional(self.description, languages),
                "probes": None
                if self.probes is None
                else [_pick_localized_text(probe, languages) for probe in self.probes],
                "alternative_main_questions": None
                if self.alternative_main_questions is None
                else [
                    _pick_localized_text(wording, languages)
                    for wording in self.alternative_main_questions
                ],
                "survey_item": None
                if self.survey_item is None
                else _render_survey_item(self.survey_item, languages),
                "image": None if self.image is None else self.image.render(languages),
                "conditions": None
                if self.conditions is None
                else self.conditions.render(options, languages),
            }
        )

    def texts(self, where: str) -> Iterator[tuple[str, LocalizedText]]:
        yield f"{where}.main_question", self.main_question
        if self.description is not None:
            yield f"{where}.description", self.description
        for n, probe in enumerate(self.probes or []):
            yield f"{where}.probes[{n}]", probe
        for n, wording in enumerate(self.alternative_main_questions or []):
            yield f"{where}.alternative_main_questions[{n}]", wording
        match self.survey_item:
            case LocalizedRadioItem() | LocalizedCheckboxItem() | LocalizedLikertItem():
                for n, option in enumerate(self.survey_item.options):
                    yield f"{where}.survey_item.options[{n}]", option.text
            case LocalizedSliderItem():
                if self.survey_item.min_label is not None:
                    yield f"{where}.survey_item.min_label", self.survey_item.min_label
                if self.survey_item.max_label is not None:
                    yield f"{where}.survey_item.max_label", self.survey_item.max_label
        if self.image is not None:
            if self.image.primer is not None:
                yield f"{where}.image.primer", self.image.primer
            yield f"{where}.image.description", self.image.description
            yield f"{where}.image.alt", self.image.alt

    @classmethod
    def from_rendered(
        cls, questions: Mapping[str, Question], merge: _Merge, where: str
    ) -> LocalizedQuestion:
        """Build from one rendering per language. Conditions are converted
        afterwards (`_merge_conditions`), once every option is known."""
        base = questions[merge.structure]
        question = cls.model_validate(
            {
                **base.model_dump(exclude=_QUESTION_TEXT | _QUESTION_DERIVED),
                "main_question": merge.text(
                    {lang: q.main_question for lang, q in questions.items()}
                ),
                "description": merge.text(
                    {lang: q.description for lang, q in questions.items()}
                ),
                "probes": merge.texts(
                    {lang: q.probes for lang, q in questions.items()},
                    f"{where}.probes",
                ),
                "alternative_main_questions": merge.texts(
                    {
                        lang: q.alternative_main_questions
                        for lang, q in questions.items()
                    },
                    f"{where}.alternative_main_questions",
                ),
                "survey_item": _merge_survey_item(
                    {lang: q.survey_item for lang, q in questions.items()}, merge, where
                ),
                "image": _merge_image(
                    {lang: q.image for lang, q in questions.items()}, merge, where
                ),
            }
        )
        if (options := getattr(question.survey_item, "options", None)) is not None:
            merge.options[question.id] = options
        return question


class LocalizedQuestionSection(BaseModel):
    """`QuestionSection` with its text per language."""

    id: UUID = Field(default_factory=uuid4)
    description: LocalizedText = Field(default_factory=dict)
    questions: list[LocalizedQuestion] = Field(default_factory=list)
    shuffle: bool = False
    ai_generated_questions: GeneratedQuestions = GeneratedQuestions()

    def render(
        self, options: Mapping[UUID, LocalizedText], languages: Sequence[str]
    ) -> QuestionSection[Question]:
        return QuestionSection[Question](
            id=self.id,
            description=_pick_localized_text(self.description, languages),
            questions=[q.render(options, languages) for q in self.questions],
            shuffle=self.shuffle,
            ai_generated_questions=self.ai_generated_questions,
        )

    def texts(self, where: str) -> Iterator[tuple[str, LocalizedText]]:
        yield f"{where}.description", self.description
        for n, question in enumerate(self.questions):
            yield from question.texts(f"{where}.questions[{n}]")

    @classmethod
    def from_rendered(
        cls,
        sections: Mapping[str, QuestionSection[Question]],
        structure: str,
        report: list[str] | None = None,
    ) -> LocalizedQuestionSection:
        """Build a section from one rendering per language, e.g. a generated
        section in the language it was generated in."""
        merge = _Merge(structure)
        section = _merge_section(sections, merge, "section")
        _merge_conditions(
            [section], {q.id: q for q in sections[structure].questions}, merge
        )
        if report is not None:
            report.extend(merge.report)
        return section


def _merge_section(
    sections: Mapping[str, QuestionSection[Question]], merge: _Merge, where: str
) -> LocalizedQuestionSection:
    base = sections[merge.structure]
    by_id = {
        language: {question.id: question for question in section.questions}
        for language, section in sections.items()
    }
    questions = []
    for n, question in enumerate(base.questions):
        renderings = {
            language: found[question.id]
            for language, found in by_id.items()
            if question.id in found
        }
        for language in sections.keys() - renderings.keys():
            merge.report.append(
                f"{language}: {where}.questions[{n}] ({question.id}) is missing; "
                "its text is left untranslated"
            )
        questions.append(
            LocalizedQuestion.from_rendered(
                renderings, merge, f"{where}.questions[{n}]"
            )
        )
    return LocalizedQuestionSection(
        id=base.id,
        description=merge.text(
            {language: section.description for language, section in sections.items()},
        )
        or {},
        questions=questions,
        shuffle=base.shuffle,
        ai_generated_questions=base.ai_generated_questions,
    )


def _merge_conditions(
    sections: Sequence[LocalizedQuestionSection],
    rendered: Mapping[UUID, Question],
    merge: _Merge,
) -> None:
    """Convert the structure language's conditions, now that every option has
    an id. `rendered` holds the structure language's questions by id."""
    for n_section, section in enumerate(sections):
        for n_question, question in enumerate(section.questions):
            source = rendered[question.id]
            if source.conditions is not None:
                question.conditions = LocalizedConditions.from_rendered(
                    source.conditions,
                    merge,
                    f"question_sections[{n_section}].questions[{n_question}].conditions",
                )


# -- The guide -----------------------------------------------------------------


class LocalizedTimedMessage(BaseModel):
    """`TimedMessage` with its text per language."""

    message: LocalizedText = Field(default_factory=dict)
    time: int
    include_in_history: bool = False
    as_modal: bool = False

    def render(self, languages: Sequence[str]) -> TimedMessage:
        return TimedMessage(
            **self.model_dump(exclude={"message"}),
            message=_pick_localized_text(self.message, languages),
        )


def _message_text(value: str | InterviewMessage | None) -> str | None:
    return value.message if isinstance(value, InterviewMessage) else value


class LocalizedInterviewGuide(BaseModel):
    """`InterviewGuide`'s single source across a project's languages.

    Validating it renders every language it has text in, so a source that
    loads is one every language can be interviewed from. A leaf missing in one
    language falls back to the others for that check; at interview time the
    caller chooses the fallback (the project's default language).
    """

    model_config = {"title": "LocalizedInterviewGuide"}

    framing: LocalizedText = Field(default_factory=dict)
    introduction: LocalizedText = Field(default_factory=dict)
    question_sections: list[LocalizedQuestionSection] = Field(default_factory=list)
    outro: LocalizedText | None = None
    alt_outro: LocalizedText | None = None
    extra_variables: list[str] = Field(default_factory=list)
    timed_messages: list[LocalizedTimedMessage] | None = None
    ai_generated_sections: int = 0

    @model_validator(mode="after")
    def _validate_renders(self):
        option_ids = [
            option.id
            for options in self._options_by_question().values()
            for option in options
        ]
        if duplicates := sorted(
            {str(i) for i in option_ids if option_ids.count(i) > 1}
        ):
            raise ValueError(f"Duplicate option id(s): {duplicates}")

        owned = {
            option.id: question_id
            for question_id, options in self._options_by_question().items()
            for option in options
        }
        for n_section, section in enumerate(self.question_sections):
            for n_question, question in enumerate(section.questions):
                for condition in (
                    question.conditions.conditions if question.conditions else []
                ):
                    read = condition.question_context.question_id
                    for evaluation in condition.evaluation:
                        if (
                            evaluation.option_id is not None
                            and owned.get(evaluation.option_id) != read
                        ):
                            raise ValueError(
                                f"question_sections[{n_section}].questions[{n_question}] "
                                "has a condition naming an option that is not one of "
                                "the question it reads"
                            )

        languages = sorted(self.languages())
        for language in languages or [""]:
            try:
                self.render(language, fallback=languages)
            except ValidationError as error:
                raise ValueError(f"{language or 'The guide'}: {error}") from None
        return self

    def _options_by_question(self) -> dict[UUID, list[LocalizedOption]]:
        return {
            question.id: question.survey_item.options
            for section in self.question_sections
            for question in section.questions
            if isinstance(
                question.survey_item,
                LocalizedRadioItem | LocalizedCheckboxItem | LocalizedLikertItem,
            )
        }

    def texts(self) -> Iterator[tuple[str, LocalizedText]]:
        """Every text leaf with its path, in guide order. The leaves are the
        model's own dicts, so writing to one edits the guide."""
        yield "framing", self.framing
        yield "introduction", self.introduction
        for n, section in enumerate(self.question_sections):
            yield from section.texts(f"question_sections[{n}]")
        if self.outro is not None:
            yield "outro", self.outro
        if self.alt_outro is not None:
            yield "alt_outro", self.alt_outro
        for n, message in enumerate(self.timed_messages or []):
            yield f"timed_messages[{n}].message", message.message

    def languages(self) -> set[str]:
        """The languages any text is written in."""
        return {
            language
            for _, text in self.texts()
            for language, value in text.items()
            if value
        }

    def missing(self, language: str) -> list[str]:
        """Paths of the leaves written in another language but not in this one."""
        return [path for path, text in self.texts() if _missing(text, language)]

    def render(self, language: str, fallback: Sequence[str] = ()) -> InterviewGuide:
        """The guide in `language`. A leaf it has no text for takes the first
        of `fallback` that has, else is empty."""
        languages = [language, *fallback]
        options = {
            option.id: option.text
            for options in self._options_by_question().values()
            for option in options
        }
        return InterviewGuide(
            framing=_pick_localized_text(self.framing, languages),
            introduction=_pick_localized_text(self.introduction, languages),
            question_sections=[
                section.render(options, languages) for section in self.question_sections
            ],
            outro=_pick_optional(self.outro, languages),
            alt_outro=_pick_optional(self.alt_outro, languages),
            extra_variables=self.extra_variables,
            timed_messages=None
            if self.timed_messages is None
            else [message.render(languages) for message in self.timed_messages],
            ai_generated_sections=self.ai_generated_sections,
        )

    @classmethod
    def from_rendered(
        cls,
        guides: Mapping[str, InterviewGuide],
        structure: str,
        report: list[str] | None = None,
    ) -> LocalizedInterviewGuide:
        """Build the source from one rendering per language.

        `structure`'s rendering supplies everything shared -- sections,
        questions and their order, settings, survey item types, conditions.
        The others contribute only text: sections and questions are matched
        by id, and probes, wordings, options and timed messages by position
        within them. Nothing is invented where they disagree; the text is left
        missing and the mismatch appended to `report`.
        """
        merge = _Merge(structure)
        base = guides[structure]

        sections_by_id = {
            language: {section.id: section for section in guide.question_sections}
            for language, guide in guides.items()
        }
        sections = []
        for n, section in enumerate(base.question_sections):
            renderings = {
                language: found[section.id]
                for language, found in sections_by_id.items()
                if section.id in found
            }
            for language in guides.keys() - renderings.keys():
                merge.report.append(
                    f"{language}: question_sections[{n}] ({section.id}) is missing; "
                    "its text is left untranslated"
                )
            sections.append(
                _merge_section(renderings, merge, f"question_sections[{n}]")
            )

        _merge_conditions(
            sections,
            {
                question.id: question
                for section in base.question_sections
                for question in section.questions
            },
            merge,
        )

        timed = merge.texts(
            {
                language: [message.message for message in guide.timed_messages]
                if guide.timed_messages is not None
                else None
                for language, guide in guides.items()
            },
            "timed_messages",
        )
        # Variables are a union: a translation may use one the structure
        # language happens not to.
        extra_variables = list(
            dict.fromkeys(
                variable
                for guide in (base, *guides.values())
                for variable in guide.extra_variables
            )
        )

        guide = cls(
            framing=merge.text({language: g.framing for language, g in guides.items()})
            or {},
            introduction=merge.text(
                {
                    language: _message_text(g.introduction)
                    for language, g in guides.items()
                }
            )
            or {},
            question_sections=sections,
            outro=merge.text(
                {language: _message_text(g.outro) for language, g in guides.items()},
            ),
            alt_outro=merge.text(
                {language: g.alt_outro for language, g in guides.items()}
            ),
            extra_variables=extra_variables,
            timed_messages=None
            if base.timed_messages is None or timed is None
            else [
                LocalizedTimedMessage(
                    **message.model_dump(exclude={"message"}), message=text
                )
                for message, text in zip(base.timed_messages, timed)
            ],
            ai_generated_sections=base.ai_generated_sections,
        )
        if report is not None:
            report.extend(merge.report)
        return guide


# -- Consent and welcome -------------------------------------------------------


class LocalizedConsent(BaseModel):
    title: LocalizedText = Field(default_factory=dict)
    text: LocalizedText = Field(default_factory=dict)

    def texts(self) -> Iterator[tuple[str, LocalizedText]]:
        yield "title", self.title
        yield "text", self.text

    def missing(self, language: str) -> list[str]:
        return [path for path, text in self.texts() if _missing(text, language)]

    def render(self, language: str, fallback: Sequence[str] = ()) -> Consent:
        languages = [language, *fallback]
        return Consent(
            title=_pick_localized_text(self.title, languages),
            text=_pick_localized_text(self.text, languages),
        )

    @classmethod
    def from_rendered(cls, consents: Mapping[str, Consent]) -> LocalizedConsent:
        return cls(
            title={lang: c.title for lang, c in consents.items() if c.title},
            text={lang: c.text for lang, c in consents.items() if c.text},
        )


class LocalizedWelcome(BaseModel):
    title: LocalizedText = Field(default_factory=dict)
    text: LocalizedText = Field(default_factory=dict)
    email: EmailStr
    video_file_name: str | None = None
    with_id: bool = True

    def texts(self) -> Iterator[tuple[str, LocalizedText]]:
        yield "title", self.title
        yield "text", self.text

    def missing(self, language: str) -> list[str]:
        return [path for path, text in self.texts() if _missing(text, language)]

    def render(self, language: str, fallback: Sequence[str] = ()) -> Welcome:
        languages = [language, *fallback]
        return Welcome(
            **self.model_dump(exclude={"title", "text"}),
            title=_pick_localized_text(self.title, languages),
            text=_pick_localized_text(self.text, languages),
        )

    @classmethod
    def from_rendered(
        cls, welcomes: Mapping[str, Welcome], structure: str
    ) -> LocalizedWelcome:
        """`structure`'s welcome supplies the shared fields."""
        return cls(
            **welcomes[structure].model_dump(exclude={"title", "text"}),
            title={lang: w.title for lang, w in welcomes.items() if w.title},
            text={lang: w.text for lang, w in welcomes.items() if w.text},
        )
