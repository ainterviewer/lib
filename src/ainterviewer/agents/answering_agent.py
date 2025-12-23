from random import uniform

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import AnsweringAgentPrompts
from ainterviewer.synthesize.interviewees import InterviewSubject
from ainterviewer.types import LanguageCode, MessageRole
from ainterviewer.utils import create_transcript


class AnsweringAgent(BaseAgent[AnsweringAgentPrompts]):
    """Agent that answers questions"""

    def __init__(
        self,
        interview_subject: InterviewSubject,
        language: LanguageCode | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs | {"interview_subject": interview_subject})
        self.interview_subject = interview_subject
        self.language = language

        self.messages += [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]

    async def answer(
        self, question: str, additional_instructions: str | None = None
    ) -> str:
        transcript = create_transcript(self.messages, interviewee=True)

        self.messages.append({"role": MessageRole.USER, "content": question})

        if uniform(0, 1) < self.interview_subject.refusal_rate:
            refusal_instruction = "\nIMPORTANT: You must refuse to answer the question."

            if additional_instructions:
                additional_instructions += refusal_instruction
            else:
                additional_instructions = refusal_instruction

        answering_prompt = self.prompts.generate_answering_prompt(
            transcript=transcript,
            question=question,
            additional_instructions=additional_instructions,
            translation=self.language,
        )

        messages = [
            {"role": "system", "content": self.prompts.system_prompt},
            {"role": "user", "content": answering_prompt},
        ]
        message = await self.chat_api(messages)

        self.messages.append({"role": MessageRole.ASSISTANT, "content": message})

        return message
