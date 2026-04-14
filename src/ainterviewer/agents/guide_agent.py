from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import GuideAgentPrompts
from ainterviewer.interview_guides import InterviewGuide, Question
from ainterviewer.interview_guides.interview_guide import QuestionSection
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.interview_guides.sections import QuestionSectionTemplate
from ainterviewer.lpm.types import Message
from ainterviewer.types import LanguageCode, MessageRole
from ainterviewer.utils import get_language_dict


class GuideAgent(BaseAgent[GuideAgentPrompts]):
    def __init__(
        self,
        language: LanguageCode,
        *args,
        **kwargs,
    ):
        """An agent that probes an interviewee based on an interview an interview guide and the answers."""
        super().__init__(language=language, *args, **kwargs)

        self.messages: list[Message] = [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]

    async def generate_main_question(
        self,
        interview_transcript: str,
        interview_guide: InterviewGuide,
    ) -> Question:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_question_prompt(
            interview_transcript=interview_transcript,
            interview_guide=interview_guide,
            translation=translation_lang,
        )

        messages: list[Message] = self.messages + [
            {"role": MessageRole.USER, "content": probing_prompt}
        ]
        self.logger.info(f"Generating main question: {messages}")
        base = await self.chat_api(messages, response_format=QuestionBase)
        self.logger.info(f"Main question generated: {base}")

        question = Question.model_validate(base.model_dump())

        return question

    async def generate_question_section(
        self,
        interview_transcript: str,
        interview_guide: InterviewGuide,
    ) -> QuestionSection[Question]:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_section_prompt(
            interview_transcript=interview_transcript,
            interview_guide=interview_guide,
            translation=translation_lang,
        )

        messages: list[Message] = self.messages + [
            {"role": MessageRole.USER, "content": probing_prompt}
        ]
        self.logger.info(f"Generating main question: {messages}")
        template = await self.chat_api(
            messages, response_format=QuestionSectionTemplate
        )
        self.logger.info(f"Main question generated: {template}")

        question_section = QuestionSection[Question].model_validate(
            template.model_dump()
        )

        return question_section
