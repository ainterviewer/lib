from typing import Literal

from ainterviewer.lpm.types import Message
from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import ReformulationAgentPrompts
from ainterviewer.types import LanguageCode, MessageRole
from ainterviewer.utils import get_language_dict

ReformulationReason = Literal["already_answered", "segue", "skipped"]


class ReformulationAgent(BaseAgent[ReformulationAgentPrompts]):
    def __init__(
        self,
        language: LanguageCode,
        *args,
        **kwargs,
    ):
        """An agent that probes an interviewee based on an interview an interview guide and the answers."""
        super().__init__(language=language, *args, **kwargs)

        self.messages = [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]
        self.reformulation_reasons: dict[ReformulationReason, str] = {
            "already_answered": "That question has already been answered. Please respond with a reformulated version of the question and nothing else, while taking the previous answer into account",
            "segue": "Question should be reforumlated if it improves the conversational flow. Draw on previous answers in a natural and general way if they are relevant",
            "skipped": "The user tried to skip that question. Please respond with a reformulated version of the question and nothing else",
        }

    async def reformulate_question(
        self,
        interview_transcript: str,
        probing_context: str,
        question: str,
        reason: ReformulationReason,
        additional_guidelines: list[str] | None = None,
    ) -> str:
        if additional_guidelines is None:
            additional_guidelines = []

        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        reformulation_prompt = self.prompts.generate_reformulation_prompt(
            interview_transcript=interview_transcript,
            probing_context=probing_context,
            question=question,
            reason=self.reformulation_reasons[reason],
            translation=translation_lang,
            additional_guidelines=additional_guidelines,
        )

        messages: list[Message] = self.messages + [
            {"role": MessageRole.USER, "content": reformulation_prompt}
        ]

        self.logger.info(f"Reformulating question: {reformulation_prompt}")

        reformulated_question = await self.chat_api(messages)

        self.logger.info(f"Reformulated question: {reformulated_question}")

        return reformulated_question
