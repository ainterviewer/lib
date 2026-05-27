import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any, Literal, Self

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
from ainterviewer.agents.types import ProbingStrategy
from ainterviewer.config import InterviewConfig
from ainterviewer.exceptions import (
    EndInterviewCondition,
    SkipProbesCondition,
    SkipQuestionCondition,
    SkipQuestionException,
    SkipSectionCondition,
)
from ainterviewer.interfaces import (
    IOProtocol,
    OutgoingData,
    OutgoingMessage,
    PersistenceProtocol,
)
from ainterviewer.interview_guides import (
    Condition,
    ConditionAction,
    Image,
    InterviewGuide,
    InterviewMessage,
    Question,
    TimedMessage,
    fill_variables_in_message,
)
from ainterviewer.interview_guides.conditions import ConditionEvaluator, Conditions
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    ImageHistory,
    InterviewHistory,
    Turn,
)
from ainterviewer.interview_guides.references import QuestionIndex
from ainterviewer.interview_guides.sections import QuestionSection
from ainterviewer.interview_guides.survey_items import SurveyItem
from ainterviewer.interview_guides.types import ContextType
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import InterviewStatus, LanguageCode, MessageRole, MessageType


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
    ):
        self.interview_started = datetime.now()

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

        self.translation = language if language != "EN" else None

        self.project_id: UUID4 = project_id
        self.interview_id: UUID4 = interview_id

        self.referable_values = (referable_values or {}) | {
            "project_id": project_id,
            "interview_id": interview_id,
        }

        self.resume_from_history: bool = False

        self._evaluated_conditions: dict[QuestionIndex, str] = {}

        self.probing_agent: ProbingAgent = ProbingAgent(
            interview_framing=interview_guide.framing,
            few_shot_examples=agent_configs.probing.few_shot_examples,
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
        # TODO:
        # - Log exceptions
        # - Store reason

        if exc_type is not None:
            self.db.update_interview_status(
                self.project_id,
                self.interview_id,
                status=InterviewStatus.INACTIVE,
                time_spent=self.time_spent,
            )

        return False

    @property
    def time_spent(self) -> int:
        time_spent = (datetime.now() - self.interview_started).seconds
        return time_spent

    async def receive_data(
        self, message_type_to_receive: MessageType | None = None
    ) -> str:
        text, message_type_received = await self.io.receive_message(
            message_id=self.interview_history.current_message_id + 1,
            message_type=message_type_to_receive,
        )

        processed_text = await self.preprocess_answer(text)

        # TODO:
        # Implement and updated version of the SafetyAgent
        #
        # if self.security_agent and not self.security_agent.is_safe(processed_text):
        #     message = "I'm sorry, but your last message is not within the scope of this interview. Please try again."
        #     await self.send_data(message)

        self.db.insert_message(
            message_id=self.interview_history.current_message_id + 1,
            content=processed_text,
            message_type=message_type_received,
            role=MessageRole.USER,
            section=self.interview_history.current_section_index,
            main_question=self.interview_history.current_question_index,
            sub_question=self.interview_history.current_probe_index,
            interview_id=self.interview_id,
            project_id=self.project_id,
        )

        self.interview_history.add_answer(HistoryMessage(message=processed_text))

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

        if interview_history:
            await self.process_history(interview_history)
        elif intro := self.interview_guide.introduction:
            await self.handle_intro(intro)

        try:
            await self.handle_sections()
        except EndInterviewCondition:
            # Raised by a condition that ends the interview, i.e. missing
            # consent

            # TODO:
            # - Make more fine-grained an configurable.
            if outro := self.interview_guide.alt_outro:
                message = await self.preprocess_message(outro)
                self.interview_history.outro = HistoryMessage(message=message)
                await self.send_data(
                    message,
                    with_interview_structure=False,
                    can_answer=False,
                    outro=True,
                )

        if self.interview_history.outro is None:
            if outro := self.interview_guide.outro:
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

        last_section = self.interview_guide.question_sections[message.section]
        last_question = last_section.questions[message.main_question]

        try:
            if message.role == "assistant" and message.can_answer:
                await self.receive_data()

            await self.probe(last_question, last_section.description)

            self.resume_from_history = True
        except SkipQuestionCondition:
            await self.handle_skip_question_exception(last_question)
        except SkipQuestionException:
            pass

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
            main_question=Turn(question=history_message),
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
            for question in section.questions[initial_question_index:]:
                await self.handle_question(question, section.description)

            for _ in range(section.ai_generated_questions.n):
                transcript = self.interview_history.get_transcript(
                    with_descriptions=True
                )

                question = await self.guide_agent.generate_main_question(
                    interview_transcript=transcript,
                    interview_guide=self.interview_guide,
                    max_probes_n=section.ai_generated_questions.max_probes_n,
                    max_probes_time=section.ai_generated_questions.max_probes_time,
                )
                section.questions.append(question)
                section.ai_generated_questions.n -= 1

                self.db.update_interview_guide(
                    self.project_id, self.interview_id, self.interview_guide
                )

                await self.handle_question(question, section.description)
        except SkipSectionCondition:
            # TODO: We need to handle this somehow in the interview history / database ...
            return

    async def handle_question(self, question: Question, section_description: str):
        question_reformulated = False
        check_condition_after = False

        try:
            if conditions := question.conditions:
                for condition in conditions.conditions:
                    if self.should_check_condition_after_question(condition):
                        check_condition_after = True

                if not check_condition_after:
                    await self.check_conditions(conditions)

            if self.interview_history.current_question_index:
                if question.check_if_answered:
                    if await self.has_question_been_answered(question.main_question):
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
                # TODO:
                # Should skipping main question reformulate it or send
                # it to next main question?

                reformulated_question = await self.reformulate_question(
                    question=question,
                    section_description=section_description,
                    reason="skipped",
                )
                answer = await self.ask_probe(question, reformulated_question)

                if answer == CustomToken.skip_question:
                    return
            elif answer == CustomToken.no_answer:
                await asyncio.sleep(2.5)
                return

            if conditions is not None and check_condition_after:
                await self.check_conditions(conditions)

            if question.max_probes_n or question.max_probes_time:
                await self.probe(question, section_description)

        except SkipProbesCondition:
            pass
        except SkipQuestionCondition:
            if not check_condition_after:
                await self.handle_skip_question_exception(question)
        except SkipQuestionException:
            pass

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

        if image := question.image:
            if not image.data:
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
            main_question=Turn(question=history_message),
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

        if isinstance(question, Question):
            if question.can_answer is False:
                return CustomToken.no_answer

        answer = await self.receive_data(
            message_type_to_receive=MessageType.SURVEY_ITEM
            if question.survey_item
            else None
        )

        # TODO: if the answer is a special token, should it then be added to
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

        if isinstance(question, Question):
            if question.can_answer is False:
                return CustomToken.no_answer

        answer = await self.receive_data()

        # TODO: if the answer is a special token, should it then be added to
        # the interview history?

        return answer

    async def send_timed_message(self, timed_message: TimedMessage):
        # TODO:
        # - This should also be stored in the database, so they wont be
        # send again if somebody reconnects.
        #   Currently this is being handled by looking at the time spent in
        #   last session but this is not bullet proof, so should update to look
        #   for the timed_message in the db.

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
        if (
            condition.question_context.section
            == self.interview_history.current_section_index
        ) and (
            (
                condition.question_context.question - 1
                == self.interview_history.current_question_index
            )
            or (
                condition.question_context.question == 0
                and self.interview_history.current_question_index is None
            )
        ):
            return True

        return False

    async def check_conditions(self, conditions: Conditions) -> None:
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
            task="evaulate_condition",
            content=conditions.model_dump_json(),
            response=str(condition_triggered),
        )

        if condition_triggered:
            match conditions.action:
                case ConditionAction.SKIP_PROBES:
                    raise SkipProbesCondition
                case ConditionAction.SKIP_QUESTION:
                    raise SkipQuestionCondition
                case ConditionAction.SKIP_SECTION:
                    raise SkipSectionCondition
                case ConditionAction.END_INTERVIEW:
                    raise EndInterviewCondition
                case _:
                    raise ValueError("Invalid condition action")

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
            if probing_context := question.probing_context:
                # TODO: Implement with new interview_history
                match probing_context:
                    case ContextType.QUESTION:
                        ...
                    case ContextType.SECTION:
                        ...

            transcript = self.interview_history.get_transcript()

            probe = await self.generate_probe(
                section_description=section_description,
                question=question,
                transcript=transcript,
            )

            if probe.lower().startswith(CustomToken.end_of_probe):
                break

            answer = await self.ask_probe(question, probe)

            if answer == CustomToken.skip_question:
                # NOTE: skipping a probe skips the main question
                raise SkipQuestionException

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
        if question.max_probes_n is not None:
            if self.interview_history.current_probe_index >= question.max_probes_n:
                return False

        if question.max_probes_time is not None:
            if question.max_probes_time <= time.time() - self.probing_time:
                return False

        if self.interview_history.current_probe_index > 0:
            contains_refusal = await self.contains_refusal()
            if contains_refusal:
                return False

            if question.check_if_exhausted:
                if await self.has_main_question_been_exhausted(question):
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
        reason: Literal["already_answered", "segue", "skipped"],
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
