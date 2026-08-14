"""
This module contains the prompt generators for the agents
    in the interview process.
They are not supposed to be imported directly,
    but through the `ainterviewer.agents.prompts.get_prompts` module.
"""

from abc import ABC, abstractmethod
from typing import Union

from jinja2 import BaseLoader, Environment, PackageLoader, StrictUndefined, Template

from ainterviewer.agents.config import ProbingPromptSlots
from ainterviewer.agents.types import DiceStrategy
from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.interview_guide import QuestionSection
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.interview_guides.sections import QuestionSectionTemplate
from ainterviewer.synthesize.interviewees import (
    INTERVIEWEE_INFORMATION_TEMPLATE,
    InterviewSubject,
)
from ainterviewer.types import LanguageCode
from ainterviewer.utils import get_language_dict


class BasePrompts(ABC):
    system_prompt: str = ""

    def __init__(
        self,
        lang: LanguageCode = "EN",
        template_loader: BaseLoader | None = None,
    ):
        if not template_loader:
            template_loader = PackageLoader(
                "ainterviewer.agents.prompts.templates",
                "EN",  # lang
            )

        self.lang: LanguageCode = lang
        # The templates are written in English, so `translation` is the name of the
        # language the agent must actually speak -- `None` when that is English.
        self.translation: str | None = (
            get_language_dict(language_code=lang)["name"] if lang != "EN" else None
        )

        self.env = Environment(loader=template_loader, undefined=StrictUndefined)
        self.system_prompt = self.generate_system_prompt()

    def get_template(self, template_name: str) -> Template:
        return self.env.get_template(template_name)

    def get_source(self, template: Union[str, Template]) -> str:
        assert self.env.loader

        if isinstance(template, str):
            return self.env.loader.get_source(self.env, template)[0]
        elif isinstance(template, Template):
            if not template.filename:
                raise FileNotFoundError("The template has no filename")

            return self.env.loader.get_source(
                self.env, "/".join(template.filename.split("/")[-2:])
            )[0]

    @abstractmethod
    def generate_system_prompt(self) -> str: ...

    def print_prompt(self):
        if self.system_prompt:
            print("============== SYSTEM PROMPT ==============")
            print(self.system_prompt)
            print("===========================================\n\n")


class AnsweringAgentPrompts(BasePrompts):
    def __init__(self, interview_subject: InterviewSubject | str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interview_subject: InterviewSubject | str = interview_subject

        self.interviewee_information: str

        if isinstance(interview_subject, InterviewSubject):
            self.interviewee_information = INTERVIEWEE_INFORMATION_TEMPLATE.render(
                interview_subject
            )
        else:
            self.interviewee_information = interview_subject

    def generate_system_prompt(self) -> str:
        system_prompt_template = self.get_template(
            "answering_agent/system_prompt.jinja"
        )
        return system_prompt_template.render(translation=self.translation)

    def generate_answering_prompt(
        self,
        transcript,
        question,
        additional_instructions: str | None = None,
        translation: str | None = None,
        response_schema: dict | None = None,
    ) -> str:
        answering_prompt_template = self.get_template(
            "answering_agent/instruction_prompt.jinja"
        )

        return answering_prompt_template.render(
            interviewee_information=self.interviewee_information,
            transcript=transcript,
            question=question,
            additional_instructions=additional_instructions,
            translation=translation,
            response_schema=response_schema,
        )


class ProbingAgentPrompts(BasePrompts):
    def __init__(
        self,
        *args,
        prompt_slots: ProbingPromptSlots | None = None,
        **kwargs,
    ):
        # Resolve before super().__init__, since BasePrompts.__init__ renders the
        # system prompt (which reads these slots) during construction.
        self.prompt_slots = (prompt_slots or ProbingPromptSlots()).resolved()
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("probing_agent/system_prompt.jinja").render(
            persona=self.prompt_slots.persona,
            question_qualities=self.prompt_slots.question_qualities,
            guidelines=self.prompt_slots.guidelines,
        )

    def generate_probing_prompt(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        suggested_probes: str | None,
        translation: str | None,
        few_shot_examples: list[str] | None = None,
    ) -> str:
        return self.get_template("probing_agent/instruction_prompt.jinja").render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            suggested_probes=suggested_probes,
            translation=translation,
            few_shot_examples=few_shot_examples,
            instructions=self.prompt_slots.instructions,
        )

    STRATEGY_TEMPLATE_MAP: dict[DiceStrategy, str] = {
        DiceStrategy.DESCRIPTIVE: "probing_agent/descriptive_detail_prompt.jinja",
        DiceStrategy.IDIOGRAPHIC: "probing_agent/idiographic_memory_prompt.jinja",
        DiceStrategy.CLARIFYING: "probing_agent/clarifying_prompt.jinja",
        DiceStrategy.EXPLANATORY: "probing_agent/explanatory_prompt.jinja",
    }

    def generate_specialized_probe_prompt(
        self,
        strategy_name: DiceStrategy,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        suggested_probes: str | None,
        translation: str | None,
        few_shot_examples: list[str] | None = None,
    ) -> str:
        template_name = self.STRATEGY_TEMPLATE_MAP.get(strategy_name)
        if template_name is None:
            raise ValueError(
                f"Unknown probing strategy '{strategy_name}'. "
                f"Available: {list(self.STRATEGY_TEMPLATE_MAP.keys())}"
            )
        return self.get_template(template_name).render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            suggested_probes=suggested_probes,
            translation=translation,
            few_shot_examples=few_shot_examples,
        )

    def generate_master_to_one_prompt(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        suggested_probes: str | None,
        response_schema: dict,
    ) -> str:
        return self.get_template("probing_agent/master_to_one_prompt.jinja").render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            suggested_probes=suggested_probes,
            response_schema=response_schema,
        )

    def generate_ensemble_to_master_prompt(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        suggested_probes: str | None,
        candidate_probes: list[dict[str, str]],
        translation: str | None,
        few_shot_examples: list[str] | None = None,
    ) -> str:
        return self.get_template(
            "probing_agent/ensemble_to_master_prompt.jinja"
        ).render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            suggested_probes=suggested_probes,
            candidate_probes=candidate_probes,
            translation=translation,
            few_shot_examples=few_shot_examples,
        )

    def generate_master_to_ensemble_prompt(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        suggested_probes: str | None,
        response_schema: dict,
    ) -> str:
        return self.get_template(
            "probing_agent/master_to_ensemble_prompt.jinja"
        ).render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            suggested_probes=suggested_probes,
            response_schema=response_schema,
        )


class GuideAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("guide_agent/system_prompt.jinja").render()

    def generate_question_prompt(
        self,
        interview_transcript: str,
        interview_guide: InterviewGuide,
        section: QuestionSection[Question],
        translation: str | None,
    ) -> str:
        return self.get_template("guide_agent/instruction_prompt.jinja").render(
            interview_guide_component="main question",
            interview_transcript=interview_transcript,
            interview_guide=interview_guide,
            current_section=section,
            translation=translation,
            interview_guide_component_schema=QuestionBase.model_json_schema(),
        )

    def generate_section_prompt(
        self,
        interview_transcript: str,
        interview_guide: InterviewGuide,
        translation: str | None,
    ) -> str:
        return self.get_template("guide_agent/instruction_prompt.jinja").render(
            interview_guide_component="question section",
            interview_transcript=interview_transcript,
            interview_guide=interview_guide,
            translation=translation,
            interview_guide_component_schema=QuestionSectionTemplate.model_json_schema(),
        )


class HistoryAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("history_agent/system_prompt.jinja").render()


class SecurityAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("security_agent/system_prompt.jinja").render()

    def generate_security_prompt(self, question: str, answer: str) -> str:
        return self.get_template("security_agent/instruction_prompt.jinja").render(
            question=question, answer=answer
        )


class ClassificationAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("classification_agent/system_prompt.jinja").render()

    def generate_classification_prompt(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples: str | dict | None = None,
    ) -> str:
        return self.get_template(
            "classification_agent/instruction_prompt.jinja"
        ).render(
            text=text,
            next_question_instruction=next_question_instruction,
            classification_examples=classification_examples,
            interview_history=interview_history,
        )


class VisualAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description_prompt = self.generate_description_prompt()

    def generate_system_prompt(self) -> str:
        return self.get_template("visual_agent/system_prompt.jinja").render()

    def generate_description_prompt(self) -> str:
        return self.get_template("visual_agent/instruction_prompt.jinja").render()


class ReformulationAgentPrompts(BasePrompts):
    def generate_system_prompt(self) -> str:
        return self.get_template("reformulation_agent/system_prompt.jinja").render()

    def generate_reformulation_prompt(
        self,
        interview_transcript: str,
        probing_context: str,
        question: str,
        reason: str,
        additional_guidelines: list[str] | None = None,
        translation: LanguageCode | None = None,
    ) -> str:
        return self.get_template("reformulation_agent/instruction_prompt.jinja").render(
            interview_transcript=interview_transcript,
            probing_context=probing_context,
            question=question,
            reason=reason,
            additional_guidelines=additional_guidelines,
            translation=translation,
        )
