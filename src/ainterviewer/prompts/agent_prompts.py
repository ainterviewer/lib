"""
This module contains the prompt generators for the agents
    in the interview process.
They are not supposed to be imported directly,
    but through the `ainterviewer.prompts.get_prompts` module.
"""

import jinja2

from ainterviewer.lpm.types import CustomTokens
from ainterviewer.prompts.language_map import LANGUAGE_MAP
from ainterviewer.prompts.models import BasePrompts
from ainterviewer.synthesize.interviewees import InterviewSubject
from ainterviewer.types import LanguageCode


class AnsweringAgentPrompts(BasePrompts):
    def __init__(self, interview_subject: InterviewSubject, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.generate_system_prompt()
        self.interview_subject = interview_subject

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
            interviewee_information=str(self.interview_subject),
            transcript=transcript,
            question=question,
            additional_instructions=additional_instructions,
            translation=translation,
        )


class ProbingAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.generate_system_prompt()

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
        self.system_prompt = self.generate_system_prompt()

    def generate_system_prompt(self) -> str:
        return self.get_template("history_agent_system_prompt.jinja").render()


class SecurityAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.generate_system_prompt()

    def generate_system_prompt(self) -> str:
        return self.get_template("security_agent_system_prompt.jinja").render()

    def generate_security_prompt(self, last_question, user_answer) -> str:
        return self.get_template("security_agent_instruction_prompt.jinja").render(
            last_question=last_question, user_answer=user_answer
        )


class ClassificationAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.generate_system_prompt()

    def generate_system_prompt(self) -> str:
        return self.get_template("classification_agent_system_prompt.jinja").render()

    def genererate_classification_prompt(
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


class TranslationAgentPrompts(BasePrompts):
    def __init__(self, target_language, source_language, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_language = LANGUAGE_MAP[target_language]
        self.source_language = LANGUAGE_MAP[source_language]
        self.system_prompt = self.generate_system_prompt()
        self.probing_prompt = self.generate_probing_prompt()

    def generate_system_prompt(self) -> jinja2.Template:
        return self.get_template("translation_agent_system_prompt.jinja")

    def generate_probing_prompt(self) -> str:
        return self.get_template("translation_agent_instruction_prompt.jinja").render(
            target_language=self.target_language,
            source_language=self.source_language,
        )


class VisualAgentPrompts(BasePrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = self.generate_system_prompt()
        self.description_prompt = self.generate_description_prompt()

    def generate_system_prompt(self) -> str:
        return self.get_template("visual_agent_system_prompt.jinja").render()

    def generate_description_prompt(self) -> str:
        return self.get_template("visual_agent_instruction_prompt.jinja").render()
