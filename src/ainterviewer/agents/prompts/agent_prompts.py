"""
This module contains the prompt generators for the agents
    in the interview process.
They are not supposed to be imported directly,
    but through the `ainterviewer.agents.prompts.get_prompts` module.
"""

from abc import ABC, abstractmethod
from typing import Union

from jinja2 import BaseLoader, Environment, PackageLoader, StrictUndefined, Template

from ainterviewer.synthesize.interviewees import (
    INTERVIEWEE_INFORMATION_TEMPLATE,
    InterviewSubject,
)
from ainterviewer.types import LanguageCode


class BasePrompts(ABC):
    system_prompt: str = ""

    def __init__(
        self,
        lang: LanguageCode = "EN",
        template_loader: BaseLoader | None = None,
    ):
        if not template_loader:
            template_loader = PackageLoader(
                "ainterviewer.agents.prompts.templates", lang
            )

        self.env = Environment(loader=template_loader, undefined=StrictUndefined)
        self.system_prompt = self.generate_system_prompt()

    def get_template(self, template_name: str) -> Template:
        return self.env.get_template(template_name)

    def get_source(self, template: Union[str, Template]) -> str:
        if isinstance(template, str):
            return self.env.loader.get_source(self.env, template)[0]
        elif isinstance(template, Template):
            if not template.filename:
                raise FileNotFoundError("The template has no filename")

            return self.env.loader.get_source(
                self.env, template.filename.split("/")[-1]
            )[0]

        raise TypeError(
            f"Expected `template` to be of type `str` or `Template`, but got {type(template)}"
        )

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
            "answering_agent_system_prompt.jinja"
        )
        return system_prompt_template.render()

    def generate_answering_prompt(
        self,
        transcript,
        question,
        additional_instructions: str | None = None,
        translation: LanguageCode | None = None,
    ) -> str:
        answering_prompt_template = self.get_template(
            "answering_agent_instruction_prompt.jinja"
        )

        return answering_prompt_template.render(
            interviewee_information=self.interviewee_information,
            transcript=transcript,
            question=question,
            additional_instructions=additional_instructions,
            translation=translation,
        )


class ProbingAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("probing_agent_system_prompt.jinja").render()

    def generate_probing_prompt(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        main_question: str,
        interview_transcript: str,
        probes: str | None,
        translation: str | None,
        few_shot_examples: list[str] | None = None,
    ) -> str:
        return self.get_template("probing_agent_instruction_prompt.jinja").render(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=interview_transcript,
            probes=probes,
            translation=translation,
            few_shot_examples=few_shot_examples,
        )


class HistoryAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("history_agent_system_prompt.jinja").render()


class SecurityAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("security_agent_system_prompt.jinja").render()

    def generate_security_prompt(self, last_question, user_answer) -> str:
        return self.get_template("security_agent_instruction_prompt.jinja").render(
            last_question=last_question, user_answer=user_answer
        )


class ClassificationAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_system_prompt(self) -> str:
        return self.get_template("classification_agent_system_prompt.jinja").render()

    def generate_classification_prompt(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples: str | dict | None = None,
    ) -> str:
        return self.get_template(
            "classification_agent_instruction_prompt.jinja"
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
        return self.get_template("visual_agent_system_prompt.jinja").render()

    def generate_description_prompt(self) -> str:
        return self.get_template("visual_agent_instruction_prompt.jinja").render()


class ReformulationAgentPrompts(BasePrompts):
    def generate_system_prompt(self) -> str:
        return self.get_template("reformulation_agent_system_prompt.jinja").render()

    def generate_reformulation_prompt(
        self,
        interview_transcript: str,
        probing_context: str,
        question: str,
        reason: str,
        additional_guidelines: list[str] | None = None,
        translation: LanguageCode | None = None,
    ) -> str:
        return self.get_template("reformulation_agent_instruction_prompt.jinja").render(
            interview_transcript=interview_transcript,
            probing_context=probing_context,
            question=question,
            reason=reason,
            additional_guidelines=additional_guidelines,
            translation=translation,
        )
