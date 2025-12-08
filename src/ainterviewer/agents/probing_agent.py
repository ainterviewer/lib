from ainterviewer.agents.base import BaseAgent
from ainterviewer.prompts.agent_prompts import ProbingAgentPrompts
from ainterviewer.types import MessageRole
from ainterviewer.utils import get_language_dict


class ProbingAgent(BaseAgent[ProbingAgentPrompts]):
    def __init__(
        self,
        interview_framing: str | None,
        few_shot_examples: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """An agent that probes an interviewee based on an interview an interview guide and the answers."""
        super().__init__(*args, **kwargs)

        self.interview_framing = interview_framing
        self.few_shot_examples = few_shot_examples
        self.messages = [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]

    async def generate_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        probes: str | None,
        translation: str | None,
    ) -> str:
        translation_lang = (
            get_language_dict(language_code=translation)["name"]
            if translation
            else None
        )

        probing_prompt = self.prompts.generate_probing_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            probes=probes,
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [{"role": "user", "content": probing_prompt}]
        self.logger.info(f"Generating probe: {messages}")
        probe = await self.chat_api(messages)
        self.logger.info(f"Probe generated: {probe}")
        return probe
