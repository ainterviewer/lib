import asyncio
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


DEFAULT_STRATEGIES = {"descriptive", "idiographic", "clarifying", "explanatory"}


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
    probing_types: set[
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

    # ############### #
    # General Probing #
    # ############### #
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

    # ################### #
    # Specialized Probing #
    # ################### #

    async def _generate_specialized_probe(
        self,
        strategy_name: str,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> dict[str, str]:
        """Generate a probe using a specific specialized probing strategy.

        Returns a dict with 'strategy' and 'question' keys.
        """
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        probing_prompt = self.prompts.generate_specialized_probe_prompt(
            strategy_name=strategy_name,
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
        self.logger.info(f"Generating specialized probe ({strategy_name})")
        question = await self.chat_api(messages)
        self.logger.info(f"Specialized probe ({strategy_name}): {question}")
        return {"strategy": strategy_name, "question": question}

    async def generate_master_to_one_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> str:
        """Master selects the single best strategy, then that strategy generates the probe.

        Steps:
            1. Master picks one strategy via structured output (DiceProbesSingle).
            2. The selected specialized agent generates the probe.
        """
        # Step 1: Master selects the best strategy
        probing_prompt = self.prompts.generate_master_to_one_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            response_schema=DiceProbesSingle.model_json_schema(),
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=probing_prompt)
        ]
        self.logger.info("Master selecting probe strategy")
        selection = await self.chat_api(messages, response_format=DiceProbesSingle)
        self.logger.info(f"Master selected strategy: {selection.probing_type}")

        # Step 2: Selected specialized agent generates the probe
        result = await self._generate_specialized_probe(
            strategy_name=selection.probing_type,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            transcript=transcript,
            suggested_probes=suggested_probes,
        )
        return result["question"]

    async def generate_ensemble_to_master_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
        strategy_names: set[str] = DEFAULT_STRATEGIES,
    ) -> str:
        """All specialized agents generate probes concurrently, master picks the best.

        Steps:
            1. All specialized agents generate candidate probes in parallel.
            2. Master selects the best candidate.
        """
        # Step 1: All specialized agents generate probes concurrently
        self.logger.info(f"Ensemble generating probes for strategies: {strategy_names}")
        candidate_probes = await asyncio.gather(
            *(
                self._generate_specialized_probe(
                    strategy_name=name,
                    section_description=section_description,
                    question_description=question_description,
                    main_question=main_question,
                    transcript=transcript,
                    suggested_probes=suggested_probes,
                )
                for name in strategy_names
            )
        )

        # Step 2: Master selects the best candidate
        translation_lang = (
            get_language_dict(language_code=self.language)["name"]
            if self.language != "EN"
            else None
        )

        master_prompt = self.prompts.generate_ensemble_to_master_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            candidate_probes=list(candidate_probes),
            translation=translation_lang,
            few_shot_examples=self.few_shot_examples,
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=master_prompt)
        ]

        self.logger.info(
            f"Master selecting best from {len(candidate_probes)} candidates"
        )

        probe = await self.chat_api(messages)

        self.logger.info(f"Master selected probe: {probe}")

        return probe

    async def generate_master_to_ensemble_to_one_probe(
        self,
        section_description: str,
        question_description: str,
        main_question: str,
        transcript: str,
        suggested_probes: str | None,
    ) -> str:
        """Master selects relevant strategies, those generate probes concurrently, master picks the best.

        Steps:
            1. Master selects a subset of strategies via structured output (DiceProbesMultiple).
            2. Selected specialized agents generate candidate probes in parallel.
            3. Master selects the best candidate.
        """
        # Step 1: Master selects relevant strategies
        selection_prompt = self.prompts.generate_master_to_ensemble_prompt(
            interview_framing=self.interview_framing,
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            interview_transcript=transcript,
            suggested_probes=suggested_probes,
            response_schema=DiceProbesMultiple.model_json_schema(),
        )

        messages = self.messages + [
            Message(role=MessageRole.USER, content=selection_prompt)
        ]
        self.logger.info("Master selecting ensemble strategies")
        selection = await self.chat_api(messages, response_format=DiceProbesMultiple)
        selected_strategies = selection.probing_types
        self.logger.info(f"Master selected strategies: {selected_strategies}")

        # Steps 2 & 3: Delegate to ensemble_to_master with the selected subset
        return await self.generate_ensemble_to_master_probe(
            section_description=section_description,
            question_description=question_description,
            main_question=main_question,
            transcript=transcript,
            suggested_probes=suggested_probes,
            strategy_names=selected_strategies,  # ty: ignore[invalid-argument-type]
        )
