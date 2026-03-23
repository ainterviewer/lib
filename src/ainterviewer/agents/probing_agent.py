from typing import Literal
from pydantic import BaseModel, Field

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import ProbingAgentPrompts
from ainterviewer.lpm.types import Message
from ainterviewer.types import LanguageCode, MessageRole
from ainterviewer.utils import get_language_dict


class SpecializedProbeType(BaseModel):
    event: bool = Field()
    emotion: bool = Field()


class DiceProbes(BaseModel):
    """DICE probing strategies to use in qualitative research interview.

    The DICE framework (Robinson, 2023) identifies four theoretically grounded
    probe types for eliciting rich, deep data in semi-structured interviews:

    - **descriptive**: Draws out an in-depth account of a specific episode across
      both its *outer landscape* (who was present, where, when, what happened,
      surrounding context) and *inner landscape* (recalled feelings, thoughts,
      and motivations). Best used early in an interview while rapport is being
      established.

    - **idiographic**: Moves the interviewee from *generic* memories (averaged,
      habitual, lacking temporal detail) into *specific* time-framed memories of
      a single instance. Two sub-types: an *example probe* ("Can you walk me
      through a specific time when that happened?") and a *time-specific probe*
      that anchors the interviewee to a particular moment or period.

    - **clarifying**: Unpacks the implicit meaning behind a word, phrase, or
      statement the interviewee has already used, facilitating deeper
      self-disclosure by making the unstated explicit. Typically laddered
      sequentially to progressively surface concealed layers of meaning.
      Better suited to the middle or later stages of an interview.

    - **explanatory**: Elicits the interviewee's *personal* causal attributions —
      why they believe an event occurred, what they feel led to an outcome, or
      what caused a particular feeling. Questions must emphasise the subjective
      nature of the explanation (e.g. "Why do you personally think X happened?")
      rather than inviting objective analysis. Reserved for later in the
      interview when the interviewee is ready to make complex inferences.
    """


class DiceProbesSingle(DiceProbes):
    probing_type: Literal["descriptive", "idiographic", "clarifying", "explanatory"] = (
        Field(
            ...,
            description=(
                "The DICE probe type that is most relevant. "
                "'descriptive' develops outer and inner landscape detail of a specific episode. "
                "'idiographic' shifts the interviewee from generic to specific autobiographical memories. "
                "'clarifying' unpacks implicit meaning in a word or phrase the interviewee has used. "
                "'explanatory' elicits personal causal attributions for an event, outcome, or feeling."
            ),
        )
    )


class DiceProbesMultiple(DiceProbes):
    probing_types: list[
        Literal["descriptive", "idiographic", "clarifying", "explanatory"]
    ] = Field(
        ...,
        description=(
            "The DICE probes types that could be relevant. "
            "'descriptive' develops outer and inner landscape detail of a specific episode. "
            "'idiographic' shifts the interviewee from generic to specific autobiographical memories. "
            "'clarifying' unpacks implicit meaning in a word or phrase the interviewee has used. "
            "'explanatory' elicits personal causal attributions for an event, outcome, or feeling."
        ),
    )


class ProbingAgent(BaseAgent[ProbingAgentPrompts]):
    def __init__(
        self,
        language: LanguageCode,
        interview_framing: str,
        few_shot_examples: list[str] | None = None,
        *args,
        **kwargs,
    ):
        """An agent that probes an interviewee based on an interview an interview guide and the answers."""
        super().__init__(language=language, *args, **kwargs)

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
        suggested_probes: str | None,
    ) -> str:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_probing_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=probing_prompt)
        ]
        self.logger.info(f"Generating probe: {messages}")
        probe = await self.chat_api(messages)
        self.logger.info(f"Probe generated: {probe}")
        return probe

    async def generate_master_to_one_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> str:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_master_to_one_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=probing_prompt)
        ]
        self.logger.info(f"Generating probe: {messages}")
        probe = await self.chat_api(messages)
        self.logger.info(f"Probe generated: {probe}")
        return probe

    async def generate_ensemble_to_master_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> str:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_ensemble_to_master_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=probing_prompt)
        ]
        self.logger.info(f"Generating probe: {messages}")
        probe = await self.chat_api(messages)
        self.logger.info(f"Probe generated: {probe}")
        return probe

    async def generate_master_to_ensemble_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> str:
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_master_to_ensemble_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=probing_prompt)
        ]
        self.logger.info(f"Generating probe: {messages}")
        probe = await self.chat_api(messages)
        self.logger.info(f"Probe generated: {probe}")
        return probe
