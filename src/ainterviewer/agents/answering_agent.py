from random import uniform

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import AnsweringAgentPrompts
from ainterviewer.interview_guides import SurveyItem
from ainterviewer.interview_guides.survey_items import create_survey_answer_model
from ainterviewer.lpm.types import Message
from ainterviewer.synthesize.interviewees import InterviewSubject
from ainterviewer.types import LanguageCode, MessageRole
from ainterviewer.utils import create_transcript


class AnsweringAgent(BaseAgent[AnsweringAgentPrompts]):
    """Agent that answers questions"""

    def __init__(
        self,
        interview_subject: InterviewSubject | str,
        language: LanguageCode,
        *args,
        **kwargs,
    ):
        super().__init__(
            language=language,
            *args,
            **kwargs | {"interview_subject": interview_subject},
        )
        self.interview_subject: InterviewSubject | str = interview_subject

        self.messages += [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]

    async def answer(
        self,
        question: str,
        survey_item: SurveyItem | None = None,
        additional_instructions: str | None = None,
    ) -> str:
        transcript = create_transcript(self.messages, interviewee=True)

        self.messages.append({"role": MessageRole.USER, "content": question})

        if isinstance(self.interview_subject, InterviewSubject):
            if uniform(0, 1) < self.interview_subject.refusal_rate:
                refusal_instruction = (
                    "\nIMPORTANT: You must refuse to answer the question."
                )

                if additional_instructions:
                    additional_instructions += refusal_instruction
                else:
                    additional_instructions = refusal_instruction

        answering_prompt = self.prompts.generate_answering_prompt(
            transcript=transcript,
            question=question,
            additional_instructions=additional_instructions,
            translation=self.language if self.language != "EN" else None,
        )

        messages: list[Message] = [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
            {"role": MessageRole.USER, "content": answering_prompt},
        ]

        if survey_item:
            SurveyAnswerModel = create_survey_answer_model(survey_item)

            # Providing the json schema as a part of the message
            # greatly improves compliance/performance
            messages[-1]["content"] += (
                f"\nIMPORTANT: Follow the following json schema:\n\n```\n{SurveyAnswerModel.model_json_schema()}\n```"
            )

            response = await self.chat_api(messages, response_format=SurveyAnswerModel)

            if isinstance(response.answer, str):
                message = response.answer
            elif isinstance(response.answer, list):
                message = ", ".join(response.answer)
            else:
                message = str(response.answer)
        else:
            message = await self.chat_api(messages)

        self.messages.append({"role": MessageRole.ASSISTANT, "content": message})

        return message
