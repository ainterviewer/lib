import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, NoReturn, Self

from jinja2 import BaseLoader
from pydantic import UUID4

from ainterviewer.agents import (
    ClassificationAgent,
    GuideAgent,
    HistoryAgent,
    ProbingAgent,
    ReformulationAgent,
    SecurityAgent,
    VisualAgent,
)
from ainterviewer.agents.config import AgentConfigs
from ainterviewer.agents.reformulation_agent import ReformulationReason
from ainterviewer.agents.security_policy import TriggeredDecision
from ainterviewer.agents.types import ProbingStrategy
from ainterviewer.config import InterviewConfig
from ainterviewer.embedding import (
    DEFAULT_POLICY,
    ChunkPolicy,
    interview_chunk,
    message_chunk,
    qa_pair_chunk,
)
from ainterviewer.exceptions import (
    EndInterviewCondition,
    SkipProbesCondition,
    SkipQuestionCondition,
    SkipQuestionException,
    SkipSectionCondition,
)
from ainterviewer.interfaces import (
    EmbeddingChunk,
    EmbeddingProtocol,
    IOProtocol,
    OutgoingData,
    OutgoingMessage,
    PersistenceProtocol,
    SecurityIntervention,
    SecurityOverride,
)
from ainterviewer.interview_guides import (
    Condition,
    Image,
    InterviewGuide,
    InterviewMessage,
    Question,
    TimedMessage,
    fill_variables_in_message,
)
from ainterviewer.interview_guides.conditions import (
    ConditionEvaluator,
    Conditions,
    raise_condition,
)
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    ImageHistory,
    InterviewHistory,
    Turn,
)
from ainterviewer.interview_guides.references import QuestionIndex
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.interview_guides.types import (
    ConditionAction,
    ProbingContext,
    SecurityAction,
)
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import InterviewStatus, LanguageCode, MessageRole, MessageType

logger = logging.getLogger(__name__)


class AInterviewer:
    def __init__(
        self,
        io: IOProtocol,
        db: PersistenceProtocol,
        interview_guide: InterviewGuide,
        config: InterviewConfig,
        agent_configs: AgentConfigs,
        project_id: UUID4,
        interview_id: UUID4,
        previous_time_spent: int = 0,
        one_question: bool = False,
        template_loader: BaseLoader | None = None,
        language: LanguageCode = "EN",
        referable_values: dict[str, Any] | None = None,
        embedder: EmbeddingProtocol | None = None,
        chunk_policy: ChunkPolicy = DEFAULT_POLICY,
    ):
        self.interview_started = datetime.now(UTC)

        # NOTE:
        # If previous time spent is larger than timed_message.time, the timed
        # message will be removed. This is to avoid showing the timed message
        # twice.
        # TODO:
        # Should be improved by storing the timed message in the database
        # instead.

        if interview_guide.timed_messages:
            interview_guide.timed_messages = [
                timed_message
                for timed_message in interview_guide.timed_messages
                if previous_time_spent <= timed_message.time
            ]

        self.io: IOProtocol = io
        self.db: PersistenceProtocol = db

        self.config: InterviewConfig = config

        self.one_question: bool = one_question

        self.interview_guide: InterviewGuide = interview_guide

        self.interview_history: InterviewHistory = InterviewHistory()

        self.language: LanguageCode = language
        self.translation = language if language != "EN" else None

        # Optional: when left unset the interview emits no embedding chunks at
        # all, which is how the synthetic test runner opts out.
        self.embedder: EmbeddingProtocol | None = embedder
        # Decides which text is embeddable and how it is rendered. Whatever a
        # consumer passes here must be the same policy its backfill uses, or the
        # two paths produce different text for the same chunk.
        self.chunk_policy: ChunkPolicy = chunk_policy

        self.project_id: UUID4 = project_id
        self.interview_id: UUID4 = interview_id

        self.referable_values = (referable_values or {}) | {
            "project_id": project_id,
            "interview_id": interview_id,
        }

        self.resume_from_history: bool = False

        self._evaluated_conditions: dict[QuestionIndex, str] = {}

        # An assessment of the latest answer, running alongside whatever is
        # generated next.
        self._security_check: asyncio.Task[TriggeredDecision | None] | None = None
        # The intervention's own message takes the place of the outro.
        self.ended_by_security_check: bool = False
        # Set when a resumed interview has to move on to the next section.
        self.skip_resumed_section: bool = False

        self.probing_agent: ProbingAgent = ProbingAgent(
            interview_framing=interview_guide.framing,
            few_shot_examples=agent_configs.probing.few_shot_examples,
            prompt_slots=agent_configs.probing.prompt_slots,
            template_loader=template_loader,
            model=agent_configs.probing.model,
            language=language,
            chat_kwargs=agent_configs.probing.chat_kwargs,
        )

        self.guide_agent: GuideAgent = GuideAgent(
            template_loader=template_loader,
            model=agent_configs.guide.model,
            language=language,
            chat_kwargs=agent_configs.guide.chat_kwargs,
        )

        self.history_agent: HistoryAgent = HistoryAgent(
            template_loader=template_loader,
            model=agent_configs.history.model,
            language=language,
            chat_kwargs=agent_configs.history.chat_kwargs,
        )

        self.classification_agent: ClassificationAgent = ClassificationAgent(
            template_loader=template_loader,
            model=agent_configs.classification.model,
            language=language,
            chat_kwargs=agent_configs.classification.chat_kwargs,
        )

        self.condition_evaluator = ConditionEvaluator(
            classifier=self.classification_agent
        )

        self.reformulation_agent: ReformulationAgent = ReformulationAgent(
            template_loader=template_loader,
            model=agent_configs.reformulation.model,
            language=language,
            chat_kwargs=agent_configs.reformulation.chat_kwargs,
        )

        if agent_configs.security.include:
            self.security_agent: SecurityAgent | None = SecurityAgent(
                policy=agent_configs.security.policy,
                template_loader=template_loader,
                model=agent_configs.security.model,
                language=language,
                chat_kwargs=agent_configs.security.chat_kwargs,
            )
        else:
            self.security_agent = None

        if agent_configs.visual.include:
            self.visual_agent: VisualAgent = VisualAgent(
                template_loader=template_loader,
                model=agent_configs.visual.model,
                language=language,
            )

    async def __aenter__(self) -> Self:
        self.db.update_interview_status(
            self.project_id,
            self.interview_id,
            status=InterviewStatus.ACTIVE,
        )

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if not self.interview_history.is_finished:
            self.db.update_interview_status(
                self.project_id,
                self.interview_id,
                status=InterviewStatus.INACTIVE,
                time_spent=self.time_spent,
            )

        return False

    @property
    def time_spent(self) -> int:
        time_spent = (datetime.now(UTC) - self.interview_started).seconds
        return time_spent

    async def _emit_chunk(self, chunk: EmbeddingChunk | None) -> None:
        """Hand a chunk to the embedder, if there is one and there is a chunk.

        Never raises: this sits on the interview's critical path, and an
        unreachable embedding backend must cost the respondent nothing. What is
        dropped here is recoverable by re-deriving the chunks from the stored
        messages later, which is the reason the chunk rules live in
        `ainterviewer.embedding` rather than in either caller.
        """
        if self.embedder is None or chunk is None:
            return

        try:
            await self.embedder.embed_chunk(chunk)
        except Exception:
            logger.warning(
                "Failed to emit %s embedding chunk for interview %s",
                chunk.kind,
                self.interview_id,
                exc_info=True,
            )

    async def _emit_qa_pair(self) -> None:
        """Emit the question group the interview just finished with."""
        if self.embedder is None:
            return

        section_index = self.interview_history.current_section_index
        question_index = self.interview_history.current_question_index

        if question_index is None:
            return

        try:
            question = self.interview_history.current_question
        except IndexError:
            return

        await self._emit_chunk(
            qa_pair_chunk(
                question,
                project_id=self.project_id,
                interview_id=self.interview_id,
                section=section_index,
                main_question=question_index,
                language=self.language,
                policy=self.chunk_policy,
            )
        )

    async def receive_data(self) -> str:
        # The IO layer cannot work out what kind of answer it is receiving --
        # the respondent's client submits a closed answer in the same frame as
        # free text -- so the type is declared here, from the turn the answer
        # lands on.
        turn = self.interview_history.current_turn

        message_type_to_receive = (
            MessageType.SURVEY_ITEM
            if turn is not None and turn.survey_item is not None
            else None
        )

        text, message_type_received, audio_file = await self.io.receive_message(
            message_id=self.interview_history.current_message_id + 1,
            message_type=message_type_to_receive,
        )

        processed_text = await self.preprocess_answer(text)

        # Read before the answer is added to the history, which advances them.
        message_id = self.interview_history.current_message_id + 1
        section = self.interview_history.current_section_index
        main_question = self.interview_history.current_question_index
        sub_question = self.interview_history.current_probe_index

        self.db.insert_message(
            message_id=message_id,
            content=processed_text,
            message_type=message_type_received,
            audio_file=audio_file,
            role=MessageRole.USER,
            section=section,
            main_question=main_question,
            sub_question=sub_question,
            interview_id=self.interview_id,
            project_id=self.project_id,
        )

        self.interview_history.add_answer(HistoryMessage(message=processed_text))

        await self._emit_chunk(
            message_chunk(
                project_id=self.project_id,
                interview_id=self.interview_id,
                message_id=message_id,
                content=processed_text,
                role=MessageRole.USER,
                message_type=message_type_received,
                language=self.language,
                section=section,
                main_question=main_question,
                sub_question=sub_question,
                policy=self.chunk_policy,
            )
        )

        return processed_text

    async def send_data(
        self,
        text: str,
        can_answer: bool = True,
        include_in_history: bool = True,
        survey_item: SurveyItem | None = None,
        image: Image | list[Image] | None = None,
        with_interview_structure: bool = True,
        user_image: bool = False,
        questions_asked: int | None = None,
        is_introduction: bool = False,
        outro: bool = False,
        timed: bool = False,
        security_intervention: SecurityIntervention | None = None,
    ) -> None:
        message_id = self.db.insert_message(
            message_id=self.interview_history.current_message_id,
            content=text,
            message_type=MessageType.TEXT
            if not survey_item
            else MessageType.SURVEY_ITEM,
            can_answer=can_answer,
            include_in_history=include_in_history,
            role=MessageRole.ASSISTANT,
            survey_item=survey_item,
            image=image,
            section=(
                self.interview_history.current_section_index
                if with_interview_structure
                else None
            ),
            main_question=(
                self.interview_history.current_question_index
                if with_interview_structure
                else None
            ),
            sub_question=(
                self.interview_history.current_probe_index
                if with_interview_structure
                else None
            ),
            is_introduction=is_introduction,
            outro=outro,
            timed=timed,
            security_intervention=security_intervention,
            interview_id=self.interview_id,
            project_id=self.project_id,
        )

        if text in CustomToken:
            data = OutgoingData(content=text)
        else:
            if questions_asked:
                progress = self.calculate_progress(questions_asked)
            else:
                progress = None

            data = OutgoingMessage(
                content=text,
                survey_item=survey_item,
                image=image,
                user_image=user_image,
                message_id=message_id,
                role=MessageRole.ASSISTANT,
                can_answer=can_answer,
                progress=progress,
                is_probe=bool(self.interview_history.current_probe_index),
                security_intervention=security_intervention,
            )

        await self.io.send_data(data)

    async def send_progress(
        self,
        questions_asked: int | None,
        finished: bool = False,
    ):
        if questions_asked is not None:
            progress = self.calculate_progress(questions_asked)
        elif finished:
            progress = 100
        else:
            raise ValueError(
                "Must either provide questions_asked or set finished to True"
            )

        payload = OutgoingData(progress=progress)

        await self.io.send_data(payload)

    def calculate_progress(self, questions_asked: int):
        return questions_asked / self.interview_guide.n_total_questions * 100

    async def interview(self, interview_history: list | None = None):
        """
        Main entry point for the interview process
        """

        if not interview_history and (intro := self.interview_guide.introduction):
            await self.handle_intro(intro)

        try:
            if interview_history:
                # Inside the try, as resuming may act on a security intervention.
                await self.process_history(interview_history)

            await self.handle_sections()
        except EndInterviewCondition:
            # Raised by a condition that ends the interview, i.e. missing
            # consent

            # TODO:
            # - Make more fine-grained an configurable.
            if not self.ended_by_security_check and (
                outro := self.interview_guide.alt_outro
            ):
                message = await self.preprocess_message(outro)
                self.interview_history.outro = HistoryMessage(message=message)
                await self.send_data(
                    message,
                    with_interview_structure=False,
                    can_answer=False,
                    outro=True,
                )

        if (
            self.interview_history.outro is None
            and not self.ended_by_security_check
            and (outro := self.interview_guide.outro)
        ):
            if isinstance(outro, InterviewMessage):
                outro = outro.message

            outro = fill_variables_in_message(
                text=outro,
                referable_values=self.referable_values,
            )

            message = await self.preprocess_message(outro)

            self.interview_history.outro = HistoryMessage(message=message)

            await asyncio.sleep(2)

            await self.send_data(
                message,
                with_interview_structure=False,
                can_answer=False,
                outro=True,
            )

        await self.send_progress(None, finished=True)

        self.db.update_interview_status(
            self.project_id,
            self.interview_id,
            status=InterviewStatus.COMPLETED,
            time_spent=self.time_spent,
        )

        self.interview_history.is_finished = True

        await self.send_data(
            CustomToken.end_of_interview,
            with_interview_structure=False,
            can_answer=False,
        )

        await self._emit_chunk(
            interview_chunk(
                self.interview_history,
                project_id=self.project_id,
                interview_id=self.interview_id,
                language=self.language,
                policy=self.chunk_policy,
            )
        )

    async def process_history(self, interview_history: list):
        # FIXME:
        # - Fix image replay
        # If an image has failed being send, the primer might be the last
        # message and the interview stuck.

        # TODO:
        # - The data class for the stored interviews should be a part of this
        # library, so we can use it in this function

        message = self.interview_history.process_history(
            interview_history, self.interview_guide
        )

        if not message:
            raise ValueError("No messages in interview history")

        if message.is_introduction:
            return

        await self.send_progress(questions_asked=self.interview_history.n_questions - 1)

        intervention = self.last_security_intervention(interview_history)

        if intervention is not None:
            # Sent outside the interview structure, so the question it
            # interrupted is the last one in the history.
            section_index = self.interview_history.current_section_index
            question_index = self.interview_history.current_question_index
            if question_index is None:
                raise ValueError("Security intervention before any question")
        else:
            section_index = message.section
            question_index = message.main_question

        last_section = self.interview_guide.question_sections[section_index]
        last_question = last_section.questions[question_index]

        try:
            if intervention is not None:
                security_intervention, override = intervention

                if override is None:
                    override = (
                        await self.receive_security_override()
                        if security_intervention.respondent_override
                        else SecurityOverride.ACCEPT
                    )

                if override == SecurityOverride.ACCEPT:
                    self.apply_security_action(security_intervention.action)

            elif message.role == "assistant" and message.can_answer:
                await self.receive_data()

            await self.probe(last_question, last_section.description)
            await self.resolve_security_check()

            self.resume_from_history = True
        except SkipProbesCondition:
            self.resume_from_history = True
        except SkipSectionCondition:
            self.resume_from_history = True
            self.skip_resumed_section = True
        except SkipQuestionCondition:
            await self.handle_skip_question_exception(last_question)
        except SkipQuestionException:
            pass
        finally:
            self.cancel_security_check()

    @staticmethod
    def last_security_intervention(
        interview_history: list,
    ) -> tuple[SecurityIntervention, SecurityOverride | None] | None:
        """The security intervention the interview stopped at, if any, with the
        respondent's answer to it if they gave one."""
        last = interview_history[-1]

        if last.message_type == MessageType.SECURITY_OVERRIDE:
            intervention = interview_history[-2].security_intervention
            if intervention is None:
                raise ValueError("Security override answer without an intervention")
            return intervention, SecurityOverride(last.content)

        if last.security_intervention is not None:
            return last.security_intervention, None

        return None

    async def handle_intro(self, intro: str | InterviewMessage):
        if isinstance(intro, InterviewMessage):
            intro = intro.message

        intro = fill_variables_in_message(
            text=intro,
            referable_values=self.referable_values,
        )

        message = await self.preprocess_message(intro)

        self.interview_history.introduction = HistoryMessage(message=message)

        await self.send_data(
            message,
            can_answer=False,
            with_interview_structure=False,
            is_introduction=True,
        )

        # TODO: Make this configurable by the user, or set it dynamically
        # based on the length of the introduction message
        await asyncio.sleep(0.5)

    async def handle_skip_question_exception(self, question: Question):
        content = question.main_question
        survey_item = question.survey_item
        can_answer = question.can_answer
        include_in_history = not question.exclude_from_history
        image = question.image

        history_message = HistoryMessage(message=content, skipped_by_condition=True)

        self.interview_history.add_question(
            question_description=question.description,
            main_question=Turn(
                question=history_message,
                survey_item=question.survey_item,
            ),
            exclude_from_history=question.exclude_from_history,
            image=ImageHistory(
                primer=HistoryMessage(message=primer)
                if (primer := image.primer)
                else None,
                description=HistoryMessage(message=image.description),
            )
            if image
            else None,
        )

        self.db.insert_message(
            message_id=self.interview_history.current_message_id,
            content=content,
            message_type=MessageType.TEXT
            if not survey_item
            else MessageType.SURVEY_ITEM,
            can_answer=can_answer,
            include_in_history=include_in_history,
            role=MessageRole.ASSISTANT,
            survey_item=survey_item,
            image=image,
            section=(self.interview_history.current_section_index),
            main_question=(self.interview_history.current_question_index),
            sub_question=(self.interview_history.current_probe_index),
            is_introduction=False,
            interview_id=self.interview_id,
            project_id=self.project_id,
            skipped_by_condition=True,
        )

    async def handle_sections(self):
        # NOTE:
        # The ranges are needed when the interview is resumed

        for section in self.interview_guide.question_sections[
            self.interview_history.current_section_index :
        ]:
            if self.resume_from_history and self.skip_resumed_section:
                self.resume_from_history = False
                self.skip_resumed_section = False
                continue

            if self.resume_from_history:
                initial_question_index = (
                    current_question_index + 1
                    if (
                        current_question_index
                        := self.interview_history.current_question_index
                    )
                    is not None
                    else 0
                )

                self.resume_from_history = False
            else:
                self.interview_history.add_section(section.description)
                initial_question_index = 0

            await self.handle_section(section, initial_question_index)

        for _ in range(self.interview_guide.ai_generated_sections):
            transcript = self.interview_history.get_transcript(with_descriptions=True)

            section = await self.guide_agent.generate_question_section(
                interview_transcript=transcript,
                interview_guide=self.interview_guide,
            )
            self.interview_guide.question_sections.append(section)
            self.interview_guide.ai_generated_sections -= 1

            self.db.update_interview_guide(
                self.project_id, self.interview_id, self.interview_guide
            )

            self.interview_history.add_section(section.description)
            await self.handle_section(section)

    async def handle_section(
        self, section: QuestionSection[Question], initial_question_index: int = 0
    ):
        try:
            for question_index, question in enumerate(
                section.questions[initial_question_index:], start=initial_question_index
            ):
                await self.handle_question(
                    question, section.description, question_index=question_index
                )

            for _ in range(section.ai_generated_questions.n):
                transcript = self.interview_history.get_transcript(
                    with_descriptions=True
                )

                question = await self.guide_agent.generate_main_question(
                    interview_transcript=transcript,
                    interview_guide=self.interview_guide,
                    section=section,
                    max_probes_n=section.ai_generated_questions.max_probes_n,
                    max_probes_time=section.ai_generated_questions.max_probes_time,
                )
                section.questions.append(question)
                section.ai_generated_questions.n -= 1

                self.db.update_interview_guide(
                    self.project_id, self.interview_id, self.interview_guide
                )

                # Appended, so its index is the position it just took.
                await self.handle_question(
                    question,
                    section.description,
                    question_index=len(section.questions) - 1,
                )
        except SkipSectionCondition:
            # TODO: We need to handle this somehow in the interview history / database ...
            return

    async def handle_question(
        self,
        question: Question,
        section_description: str,
        question_index: int | None = None,
    ):
        question_reformulated = False
        check_condition_after = False
        # Which question the rule hangs off, as opposed to the ones it reads.
        # Only the guide records that, so it has to travel with the check or it
        # is lost by the time the evaluation is written down.
        condition_carrier = (
            (self.interview_history.current_section_index, question_index)
            if question_index is not None
            else None
        )

        try:
            if conditions := question.conditions:
                for condition in conditions.conditions:
                    if self.should_check_condition_after_question(condition):
                        check_condition_after = True

                if not check_condition_after:
                    await self.check_conditions(conditions, carrier=condition_carrier)

            if (
                self.interview_history.current_question_index
                and question.check_if_answered
                and await self.has_question_been_answered(question.main_question)
            ):
                question.main_question = await self.reformulate_question(
                    question=question,
                    section_description=section_description,
                    reason="already_answered",
                )
                question_reformulated = True

            if question.create_segue and not question_reformulated:
                question.main_question = await self.reformulate_question(
                    question=question,
                    section_description=section_description,
                    reason="segue",
                )

            if not question.check_if_answered and not question.create_segue:
                await asyncio.sleep(1)

            answer = await self.ask_question(question)

            if answer == CustomToken.skip_question:
                return
            elif answer == CustomToken.no_answer:
                await asyncio.sleep(2.5)
                return

            if question.survey_item is None:
                self.start_security_check(question, section_description)

            if conditions is not None and check_condition_after:
                # A security intervention takes precedence over the guide's
                # conditions.
                await self.resolve_security_check()
                await self.check_conditions(conditions, carrier=condition_carrier)

            if question.max_probes_n or question.max_probes_time:
                await self.probe(question, section_description)

            await self.resolve_security_check()

        except SkipProbesCondition:
            pass
        except SkipQuestionCondition:
            if not check_condition_after:
                await self.handle_skip_question_exception(question)
        except SkipQuestionException:
            pass
        finally:
            # Only left pending when something failed before it was resolved.
            self.cancel_security_check()

        # The question group is complete here -- main question, answer, and
        # every probe that followed -- which is the point at which it can be
        # embedded as one unit. Groups that were skipped by a condition, or
        # that drew no free-text answer, produce no chunk.
        await self._emit_qa_pair()

    async def preprocess_answer(self, message: str) -> str:
        # TODO: Add other preprocessing steps, including security measurements
        message = message.strip()

        return message

    async def preprocess_message(self, message: str) -> str:
        message = message.strip()

        if self.one_question:
            message = re.split(r"(?<=\?)", message)[0]

        return message

    async def ask_question(self, question: Question) -> str:
        """Asks the user a question and returns the answer"""

        # TODO: Reimplement
        # if question.alternative_main_questions:
        #     question.main_question = random.choice(
        #         question.alternative_main_questions + [question.main_question]
        #     )

        question.main_question = fill_variables_in_message(
            text=question.main_question,
            referable_values=self.referable_values,
        )

        if (image := question.image) and not image.data:
            image.encode(self.project_id)

            # FIXME: Having and image and segue at the same time does not
            # currently perform very well.

        question_text = question.main_question

        if references := question.references:
            question_references = []

            # Extract the references from the history
            for reference in references:
                section_reference = self.interview_history[reference.question_index[0]]
                question_reference = section_reference[reference.question_index[1]]
                answer = question_reference.main_question.answer
                if not answer:
                    raise ValueError("Answer reference not found.")

                question_references.append(answer.message)

            question_text = question_text.format(*[question_references])

        message = await self.preprocess_message(question_text)

        if self.interview_guide.timed_messages:
            remaining = []

            for tm in self.interview_guide.timed_messages:
                if self.time_spent > tm.time:
                    await self.send_timed_message(tm)
                else:
                    remaining.append(tm)

            self.interview_guide.timed_messages = remaining

        history_message = HistoryMessage(message=message)

        self.interview_history.add_question(
            question_description=question.description,
            main_question=Turn(
                question=history_message,
                survey_item=question.survey_item,
            ),
            exclude_from_history=question.exclude_from_history,
            image=ImageHistory(
                primer=HistoryMessage(message=primer)
                if (primer := image.primer)
                else None,
                description=HistoryMessage(message=image.description),
            )
            if image
            else None,
        )

        await self.send_data(
            message,
            survey_item=question.survey_item,
            user_image=question.user_image,
            questions_asked=self.interview_history.n_questions - 1,
            can_answer=question.can_answer,
            include_in_history=not question.exclude_from_history,
            image=image,
        )

        if isinstance(question, Question) and question.can_answer is False:
            return CustomToken.no_answer

        answer = await self.receive_data()

        # TODO: when the answer is a special token, should it then be added to
        # the interview history?

        return answer

    async def ask_probe(self, question: Question, probe: str):
        message = await self.preprocess_message(probe)

        if self.interview_guide.timed_messages:
            remaining = []

            for tm in self.interview_guide.timed_messages:
                if self.time_spent > tm.time:
                    await self.send_timed_message(tm)
                else:
                    remaining.append(tm)

            self.interview_guide.timed_messages = remaining

        history_message = HistoryMessage(message=message)

        self.interview_history.add_probe(probe=Turn(question=history_message))

        await self.send_data(
            message,
            user_image=question.user_image,
            can_answer=question.can_answer,
            include_in_history=not question.exclude_from_history,
        )

        if isinstance(question, Question) and question.can_answer is False:
            return CustomToken.no_answer

        answer = await self.receive_data()

        # TODO: when the answer is a special token, should it then be added to
        # the interview history?

        return answer

    async def send_timed_message(self, timed_message: TimedMessage):
        # TODO:
        # - This should also be stored in the database, so they wont be
        # send again if somebody reconnects.
        #   Currently this is being handled by looking at the time spent in
        #   last session but this is not bullet proof, so should update to look
        #   for the timed_message in the db.
        #   - Maybe give them and ID or index

        timed_message.message = fill_variables_in_message(
            text=timed_message.message,
            referable_values=self.referable_values,
        )

        timed_message_txt = await self.preprocess_message(timed_message.message)

        self.interview_history.timed_messages.append(
            HistoryMessage(message=timed_message_txt)
        )

        await self.send_data(
            timed_message_txt,
            can_answer=False,
            include_in_history=timed_message.include_in_history,
            with_interview_structure=False,
            timed=True,
        )
        # TODO: Move sleep to frontend
        await asyncio.sleep(2.5)

    def should_check_condition_after_question(self, condition: Condition):
        return bool(
            condition.question_context.section
            == self.interview_history.current_section_index
            and (
                condition.question_context.question - 1
                == self.interview_history.current_question_index
                or condition.question_context.question == 0
                and self.interview_history.current_question_index is None
            )
        )

    async def check_conditions(
        self, conditions: Conditions, carrier: tuple[int, int] | None = None
    ) -> None:
        condition_contexts = [
            self.get_condition_context(condition) for condition in conditions.conditions
        ]

        condition_triggered = await self.condition_evaluator.evaluate_conditions(
            condition_contexts, conditions
        )

        self.db.insert_task(
            message_id=self.interview_history.current_message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="evaluate_condition",
            content=conditions.model_dump_json(),
            response=str(condition_triggered),
            # "<section>:<question>", both zero-based, matching the indices on a
            # message. A rule checked before its question is asked is written
            # against the *previous* question's last message.
            context=f"{carrier[0]}:{carrier[1]}" if carrier else None,
        )

        if condition_triggered:
            raise_condition(conditions.action)

    def start_security_check(
        self, question: Question, section_description: str
    ) -> None:
        """Starts assessing the latest answer in the background, so that it runs
        alongside whatever is generated next.

        `resolve_security_check` must be awaited before the respondent is sent
        another message.
        """
        if self.security_agent is None:
            return

        assert self._security_check is None, "A security check is already pending"

        self._security_check = asyncio.create_task(
            self._assess_security(
                security_agent=self.security_agent,
                interview_framing=self.interview_guide.framing,
                section_description=section_description,
                question_description=question.description or "",
                # Read now, as the history moves on while the check runs.
                transcript=self.interview_history.get_transcript(
                    probing_context=ProbingContext.QUESTION
                ),
                message_id=self.interview_history.current_message_id,
            )
        )

    async def _assess_security(
        self,
        security_agent: SecurityAgent,
        interview_framing: str,
        section_description: str,
        question_description: str,
        transcript: str,
        message_id: int,
    ) -> TriggeredDecision | None:
        start_time = time.time()

        result = await security_agent.assess(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            transcript=transcript,
        )
        triggered = result.most_severe

        self.db.insert_task(
            message_id=message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="assess_security",
            reason=triggered.decision.name if triggered else None,
            response=result.assessment.model_dump_json(),
            model=security_agent.model,
            time_spend=int(time.time() - start_time),
        )

        return triggered

    async def resolve_security_check(self) -> None:
        """Waits for the pending security check, if any, and acts on the
        decision it triggered."""
        if (security_check := self._security_check) is None:
            return

        self._security_check = None

        if (triggered := await security_check) is None:
            return

        decision = triggered.decision
        # Not run through `preprocess_message`, which may cut it at the first
        # question mark.
        message = decision.action_text.strip()

        self.interview_history.security_messages.append(HistoryMessage(message=message))

        # Shown in a modal rather than in the chat, so the chat itself is never
        # answered -- an override is answered in the modal.
        await self.send_data(
            message,
            can_answer=False,
            with_interview_structure=False,
            security_intervention=SecurityIntervention(
                action=decision.action,
                respondent_override=decision.respondent_override,
            ),
        )

        if (
            decision.respondent_override
            and await self.receive_security_override() == SecurityOverride.OVERRIDE
        ):
            return

        self.apply_security_action(decision.action)

    async def receive_security_override(self) -> SecurityOverride:
        """Waits for the respondent's answer to the intervention they were just
        shown."""
        message_id = self.interview_history.current_message_id + 1

        text, _, _ = await self.io.receive_message(
            message_id=message_id,
            message_type=MessageType.SECURITY_OVERRIDE,
        )
        override = SecurityOverride(text)

        self.db.insert_message(
            message_id=message_id,
            content=override,
            message_type=MessageType.SECURITY_OVERRIDE,
            role=MessageRole.USER,
            include_in_history=False,
            interview_id=self.interview_id,
            project_id=self.project_id,
        )
        self.interview_history.security_messages.append(
            HistoryMessage(message=override)
        )

        return override

    def apply_security_action(self, action: SecurityAction) -> NoReturn:
        if action == ConditionAction.END_INTERVIEW:
            self.ended_by_security_check = True

        raise_condition(action)

    def cancel_security_check(self) -> None:
        if (security_check := self._security_check) is not None:
            security_check.cancel()
            self._security_check = None

    def get_condition_context(self, condition: Condition) -> str:
        section_context = self.interview_history[condition.question_context.section]

        if (main_question_index := condition.question_context.question) is not None:
            question_context = section_context[main_question_index]
        else:
            raise NotImplementedError(
                "This feature is not implemented yet. Please specify a main question index for your condition."
            )

        match condition.question_context.part:
            case "main":
                if (main_answer := question_context.main_question.answer) is None:
                    raise ValueError("Answer not found")
                condition_context = main_answer.message

            case "probes":
                condition_context = "\n".join(
                    [
                        probe.answer.message
                        for probe in question_context.probes
                        if probe.answer
                    ]
                )

            case "both":
                if (main_answer := question_context.main_question.answer) is None:
                    raise ValueError("Answer not found")

                condition_context = "\n".join(
                    [
                        main_answer.message,
                        *[
                            probe.answer.message
                            for probe in question_context.probes
                            if probe.answer
                        ],
                    ]
                )
            case _:
                raise ValueError("Invalid part")

        return condition_context

    async def probe(self, question: Question, section_description: str):
        # TODO:
        # - Should we also check if probes have been answered?

        self.probing_time = time.time()

        while await self.can_probe(question):
            transcript = self.interview_history.get_transcript(
                probing_context=question.probing_context
            )

            probe = await self.generate_probe(
                section_description=section_description,
                question=question,
                transcript=transcript,
            )

            if probe.lower().startswith(CustomToken.end_of_probe):
                break

            # The last answer has to be acted on before the next probe is asked.
            await self.resolve_security_check()

            answer = await self.ask_probe(question, probe)

            if answer == CustomToken.skip_question:
                # NOTE: skipping a probe moves the interview to the next main
                # question
                raise SkipQuestionException

            if answer != CustomToken.no_answer:
                self.start_security_check(question, section_description)

    async def generate_probe(
        self,
        section_description: str,
        question: Question,
        transcript: str,
    ):

        if (suggested_probes := question.probes) is not None:
            suggested_probes = "\n".join("- " + probe for probe in suggested_probes)

        if ProbingStrategy.DICE_MASTER_TO_ONE_PROBE in self.config.probing_strategy:
            await self.probing_agent.generate_master_to_one_probe(
                section_description=section_description,
                question_description=question.description,  # ty:ignore[invalid-argument-type]
                main_question=question.main_question,
                transcript=transcript,
                suggested_probes=suggested_probes,
            )
        if (
            ProbingStrategy.DICE_ENSEMBLE_TO_MASTER_PROBE
            in self.config.probing_strategy
        ):
            await self.probing_agent.generate_ensemble_to_master_probe(
                section_description=section_description,
                question_description=question.description,  # ty:ignore[invalid-argument-type]
                main_question=question.main_question,
                transcript=transcript,
                suggested_probes=suggested_probes,
            )
        if (
            ProbingStrategy.DICE_MASTER_TO_ENSEMBLE_TO_ONE_PROBE
            in self.config.probing_strategy
        ):
            await self.probing_agent.generate_master_to_ensemble_to_one_probe(
                section_description=section_description,
                question_description=question.description,  # ty:ignore[invalid-argument-type]
                main_question=question.main_question,
                transcript=transcript,
                suggested_probes=suggested_probes,
            )

        probe = await self.probing_agent.generate_probe(
            section_description=section_description,
            question_description=question.description,  # ty:ignore[invalid-argument-type]
            main_question=question.main_question,
            transcript=transcript,
            suggested_probes=suggested_probes,
        )

        return probe

    async def can_probe(self, question: Question) -> bool:
        if (
            question.max_probes_n is not None
            and self.interview_history.current_probe_index >= question.max_probes_n
        ):
            return False

        if (
            question.max_probes_time is not None
            and question.max_probes_time <= time.time() - self.probing_time
        ):
            return False

        if self.interview_history.current_probe_index > 0:
            contains_refusal = await self.contains_refusal()
            if contains_refusal:
                return False

            if (
                question.check_if_exhausted
                and await self.has_main_question_been_exhausted(question)
            ):
                return False

        return True

    async def has_question_been_answered(self, question: str) -> bool:
        start_time = time.time()
        transcript = self.interview_history.get_transcript(with_excludes=True)
        response = await self.classification_agent.classify(
            question,
            "The question has already been answered by the user in the interview",
            transcript,
        )

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.interview_history.current_message_id + 1,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="has_question_been_answered",
            content=question,
            response=str(response),
            time_spend=int(time_spend),
        )

        return response

    async def has_main_question_been_exhausted(self, question: Question):
        start_time = time.time()
        transcript = self.interview_history.current_question.transcribe(
            with_descriptions=True
        )

        context = json.dumps(
            {
                "section_description": question.description,
                "question_description": question.description,
                "main_question": question.main_question,
            }
        )

        response = await self.classification_agent.classify(
            context,
            "The main question has been extensively and satisfactorily answered by the user",
            transcript,
        )

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.interview_history.current_message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="has_question_been_exhausted",
            content=context,
            response=str(response),
            time_spend=int(time_spend),
        )

        return response

    async def contains_refusal(self) -> bool:
        now = time.time()
        current_question_transcript = (
            self.interview_history.current_question.transcribe()
        )
        contains_refusal = await self.classification_agent.classify(
            current_question_transcript,
            "The respondent explicitly refuses to answer the question or expresses a wish to skip to the next question",
        )
        time_spend = time.time() - now

        self.db.insert_task(
            message_id=self.interview_history.current_message_id + 1,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="contains_multiple_refusals",
            content=current_question_transcript,
            response=str(contains_refusal),
            time_spend=int(time_spend),
        )

        return contains_refusal

    async def reformulate_question(
        self,
        question: Question,
        section_description: str,
        reason: ReformulationReason,
    ) -> str:
        start_time = time.time()

        transcript = self.interview_history.get_transcript(with_excludes=True)

        probing_context: dict[str, str | list[str]] = {
            "section_description": section_description,
        }

        if question.description:
            probing_context["question_description"] = question.description

        probing_context["main_question"] = question.main_question
        if question.probes:
            probing_context["probes"] = question.probes

        message = await self.reformulation_agent.reformulate_question(
            interview_transcript=transcript,
            probing_context=json.dumps(probing_context),
            question=question.main_question,
            reason=reason,
        )

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.interview_history.current_message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="reformulate_question",
            reason=reason,
            content=question.main_question,
            response=message,
            time_spend=int(time_spend),
            model=self.probing_agent.model,
        )

        return message
