import asyncio
import json
import re
import time
from datetime import datetime
from functools import partial
from typing import Any, Literal, Optional

from jinja2 import BaseLoader
from pydantic import UUID4

from ainterviewer.agents import (
    ClassificationAgent,
    HistoryAgent,
    ProbingAgent,
    SecurityAgent,
    VisualAgent,
)
from ainterviewer.config import AgentConfigs, InterviewConfig
from ainterviewer.exceptions import (
    EndInterviewException,
    SkipQuestionException,
    SkipSectionException,
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
    DecimalString,
    Image,
    InterviewGuide,
    Question,
    evaluate_condition,
)
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    ImageHistory,
    InterviewHistory,
    Turn,
)
from ainterviewer.interview_guides.interview_guide import (
    InterviewMessage,
    TimedMessage,
    fill_variables_in_message,
)
from ainterviewer.interview_guides.survey_item import SurveyItem
from ainterviewer.interview_guides.types import ContextType
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import CustomTokens
from ainterviewer.types import LanguageCode, MessageRole, MessageType


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
        message_id: int = 0,
        template_loader: BaseLoader | None = None,
        frontend_language: LanguageCode = "EN",
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

        if timed_message := interview_guide.timed_message:
            if previous_time_spent > timed_message.time:
                interview_guide.timed_message = None

        self.io = io
        self.db = db

        self.config = config

        self.interview_guide = interview_guide

        self.interview_history = InterviewHistory()

        self.translation = frontend_language if frontend_language != "EN" else None

        self.project_id = project_id
        self.interview_id = interview_id
        self.message_id = message_id

        self.referable_values = (referable_values or {}) | {
            "project_id": project_id,
            "interview_id": interview_id,
            "message_id": message_id,
        }

        self.resume_from_history = False

        self._evaluated_conditions: dict[DecimalString, str] = {}

        _agent_configs = {
            agent: agent_config.model_dump() for agent, agent_config in agent_configs
        }

        include_agent = {
            agent: config.pop("include") for agent, config in _agent_configs.items()
        }

        self.probing_agent = ProbingAgent(
            interview_framing=interview_guide.framing,
            few_shot_examples=_agent_configs["probing"].pop("few_shot_examples"),
            template_loader=template_loader,
            model=agent_configs.probing.model,
            lang=_agent_configs["probing"].pop("lang"),
            chat_api=partial(chat, **_agent_configs["probing"]),
        )

        self.history_agent = HistoryAgent(
            template_loader=template_loader,
            model=agent_configs.history.model,
            lang=_agent_configs["history"].pop("lang"),
            chat_api=partial(chat, **_agent_configs["history"]),
        )

        self.classification_agent = ClassificationAgent(
            template_loader=template_loader,
            model=agent_configs.classification.model,
            lang=_agent_configs["classification"].pop("lang"),
            chat_api=partial(chat, **_agent_configs["classification"]),
        )

        if include_agent["security"]:
            self.security_agent = SecurityAgent(
                template_loader=template_loader,
                model=agent_configs.security.model,
                lang=_agent_configs["security"].pop("lang"),
                chat_api=partial(chat, **_agent_configs["security"]),
            )
        else:
            self.security_agent = None

        if include_agent["visual"]:
            self.visual_agent = VisualAgent(
                template_loader=template_loader,
                model=agent_configs.visual.model,
                lang=_agent_configs["visual"].pop("lang"),
                chat_api=chat,
            )

    async def __aenter__(self):
        self.db.update_interview_status(
            self.project_id,
            self.interview_id,
            is_active=True,
        )

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        # TODO:
        # - Log exceptions
        # - Store reason
        self.db.update_interview_status(
            self.project_id,
            self.interview_id,
            is_active=False,
            time_spent=self.time_spent,
        )

    @property
    def time_spent(self) -> int:
        time_spent = (datetime.now() - self.interview_started).seconds
        return time_spent

    async def receive_data(
        self, message_type_to_receive: MessageType | None = None
    ) -> str:
        self.message_id += 1

        text, message_type_received = await self.io.receive_message(
            message_type=message_type_to_receive
        )

        processed_text = await self.preprocess_answer(text)

        self.db.insert_message(
            message_id=self.message_id,
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
        survey_item: Optional[SurveyItem] = None,
        image: Optional[Image | list[Image]] = None,
        with_interview_structure: bool = True,
        user_image: bool = False,
        questions_asked: Optional[int] = None,
        is_introduction: bool = False,
        outro: bool = False,
        timed: bool = False,
    ) -> None:
        self.message_id += 1

        message_id = self.db.insert_message(
            message_id=self.message_id,
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

        if text in CustomTokens.all:
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
                include_in_history=include_in_history,
                interview_id=self.interview_id,
                role=MessageRole.ASSISTANT,
                can_answer=can_answer,
                progress=progress,
            )

        await self.io.send_data(data)

    async def send_progress(
        self,
        questions_asked: Optional[int],
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
        n_total_questions = sum(
            len(section.questions) for section in self.interview_guide.question_sections
        )

        return questions_asked / n_total_questions * 100

    async def interview(
        self,
        one_question: bool = False,  # TODO: Move this somewhere else?
        probing="restricted",
        interview_history: Optional[list] = None,
    ):
        """
        Main entry point for the interview process
        """

        # TODO:
        # - This should be based on a project setting
        # - Should this happen in init instead?
        # if self.config.preload:
        #     self.preloading_task = asyncio.create_task(self.preload_models())
        #     print("Preloading models")
        # else:
        #     # Create a dummy task that is already done, so we can await it
        #     # without blocking later
        #     self.preloading_task = asyncio.create_task(asyncio.sleep(0))

        self.one_question = one_question

        if interview_history:
            await self.process_history(interview_history)
        elif intro := self.interview_guide.introduction:
            if isinstance(intro, InterviewMessage):
                if intro.variables:
                    intro.message = fill_variables_in_message(
                        text=intro.message,
                        variables=intro.variables,
                        referable_values=self.referable_values,
                    )

                intro = intro.message
            message = await self.preprocess_message(intro)
            await self.send_data(
                message,
                can_answer=False,
                with_interview_structure=False,
                is_introduction=True,
            )
            self.interview_history.introduction = HistoryMessage(message=message)
            await asyncio.sleep(2)

        try:
            match probing:
                case "free":
                    await self.free_probing()
                case "restricted":
                    await self.restricted_probing()
                case _:
                    raise ValueError("Invalid probing method")

            if outro := self.interview_guide.outro:
                if isinstance(outro, InterviewMessage):
                    if outro.variables:
                        outro.message = fill_variables_in_message(
                            text=outro.message,
                            variables=outro.variables,
                            referable_values=self.referable_values,
                        )

                    outro = outro.message

                message = await self.preprocess_message(outro)

                self.interview_history.outro = HistoryMessage(message=message)

                await asyncio.sleep(2)

                await self.send_data(
                    message,
                    with_interview_structure=False,
                    can_answer=False,
                    outro=True,
                )
        except EndInterviewException:
            # Raised by a condition that ends the interview, i.e. missing
            # consent
            if outro := self.interview_guide.alt_outro:
                message = await self.preprocess_message(outro)
                self.interview_history.outro = HistoryMessage(message=message)
                await self.send_data(
                    message,
                    with_interview_structure=False,
                    can_answer=False,
                    outro=True,
                )
            await self.send_data(
                CustomTokens.end_of_interview,
                with_interview_structure=False,
                can_answer=False,
            )
            return

        await self.send_progress(None, finished=True)

        await self.send_data(
            CustomTokens.end_of_interview, with_interview_structure=False
        )

        self.db.update_interview_status(
            self.project_id, self.interview_id, is_complete=True
        )

    async def preload_models(self):
        raise NotImplementedError("Preloading should be done with server_manager")

    async def process_history(self, interview_history: list):
        # FIXME:
        # - Fix image replay
        # If an image has failed being send, the primer might be the last message and the interview stuck.
        message = None

        # Backfill messages to the history
        for message in interview_history:
            history_message = HistoryMessage(message=message.content)

            # TODO: Fix for surveys
            if message.role.value == "assistant":
                if message.is_introduction:
                    self.interview_history.introduction = history_message
                elif message.outro:
                    self.interview_history.outro = history_message
                elif message.timed:
                    self.interview_history.timed_messages.append(history_message)
                else:
                    try:
                        section = self.interview_history[message.section]
                    except IndexError:
                        section = self.interview_history.add_section(
                            self.interview_guide.question_sections[
                                message.section
                            ].description
                        )

                    if message.sub_question == 0:
                        question_description = (
                            self.interview_guide.question_sections[message.section]
                            .questions[message.main_question]
                            .description
                        )
                        section.add_question(
                            question_description,
                            main_question=Turn(question=history_message),
                            exclude_from_history=not message.include_in_history,
                            image=ImageHistory(
                                primer=HistoryMessage(message=message.image.primer),
                                description=HistoryMessage(
                                    message=message.image.description
                                ),
                            )
                            if message.image
                            else None,
                        )
                    else:
                        question = section[message.main_question]
                        question.add_probe(Turn(question=history_message))

            if message.role == "user":
                if message.sub_question == 0:
                    self.interview_history.current_question.main_question.answer = (
                        history_message
                    )
                else:
                    # sub questions in the database are 1 indexed,
                    # since sub question 0 is the main question
                    self.interview_history.current_question.probes[
                        message.sub_question - 1
                    ].answer = history_message

        if not message:
            raise ValueError("No messages in interview history")

        if message.is_introduction:
            return

        history_message = HistoryMessage(message=message.content)

        await self.send_progress(questions_asked=self.interview_history.n_questions - 1)

        if message.role == "assistant" and message.can_answer:
            await self.receive_data()

        last_section = self.interview_guide.question_sections[message.section]
        last_question = last_section.questions[message.main_question]

        await self.probe(last_question, last_section.description)

        self.resume_from_history = True

    async def restricted_probing(self):
        # NOTE:
        # The ranges are needed when the interview is resumed

        for section in self.interview_guide.question_sections[
            self.interview_history.current_section_index :
        ]:
            if self.resume_from_history:
                current_question_index = (
                    self.interview_history.current_question_index + 1
                )
                self.resume_from_history = False
            else:
                self.interview_history.add_section(section.description)

                current_question_index = 0

            for question in section.questions[current_question_index:]:
                await asyncio.sleep(3)
                question_reformulated = False
                check_condition_after = False

                try:
                    if condition := question.condition:
                        # NOTE:
                        # We have to add one to the question index, because it
                        # is only added when the question is asked, and we want
                        # to check for the current question
                        if (
                            condition.question_context.section
                            == self.interview_history.current_section_index
                            and condition.question_context.question
                            == self.interview_history.current_question_index + 1
                        ):
                            check_condition_after = True
                        else:
                            await self.check_condition(condition)

                    if self.interview_history.current_question_index > 0:
                        if question.check_if_answered:
                            if await self.has_question_been_answered(
                                question.main_question
                            ):
                                question.main_question = (
                                    await self.reformulate_question(
                                        question=question,
                                        section_description=section.description,
                                        reason="already_answered",
                                    )
                                )
                                question_reformulated = True

                    if question.create_segue and not question_reformulated:
                        question.main_question = await self.reformulate_question(
                            question=question,
                            section_description=section.description,
                            reason="segue",
                        )

                    if not question.check_if_answered and not question.create_segue:
                        await asyncio.sleep(1)

                    answer = await self.ask_question(question)

                    if answer == CustomTokens.skip_question:
                        question.main_question = await self.reformulate_question(
                            question=question,
                            section_description=section.description,
                            reason="skipped",
                        )
                        answer = await self.ask_question(question)

                        if answer == CustomTokens.skip_question:
                            continue
                    elif answer == CustomTokens.no_answer:
                        await asyncio.sleep(2.5)
                        continue

                    if condition is not None and check_condition_after:
                        await self.check_condition(condition)

                    if question.max_probes_n or question.max_probes_time:
                        await self.probe(question, section.description)

                except SkipSectionException:
                    break  # Exit the section
                except SkipQuestionException:
                    continue  # Skip the question

    async def preprocess_answer(self, message: str) -> str:
        # TODO: Add other preprocessing steps, including security measurements
        message = message.strip()

        return message

    async def preprocess_message(self, message: str, one_question: bool = False) -> str:
        message = message.strip()

        if one_question:
            message = re.split(r"(?<=\?)", message)[0]

        return message

    async def ask_question(self, question: Question) -> str:
        """Asks the user a question and returns the answer"""
        # FIXME: Update security check to fit in the flow

        # https://s.epinionglobal.com/mrIWeb/mrIWeb.srf?I.Project=P2100211_TEST&i.user2=n&i.user5=complete&id=ID1

        if question.alternative_main_questions:
            question.main_question = random.choice(
                question.alternative_main_questions + [question.main_question]
            )

        if question.variables:
            question.main_question = fill_variables_in_message(
                text=question.main_question,
                variables=question.variables,
                referable_values=self.referable_values,
            )

        if image := question.image:
            if not image.data:
                image.encode()

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

        message = await self.preprocess_message(
            question_text,
            one_question=self.one_question,
        )

        if timed_message := self.interview_guide.timed_message:
            if self.time_spent > timed_message.time:
                await self.send_timed_message(timed_message)

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
            questions_asked=question.n_question,
            can_answer=question.can_answer,
            include_in_history=not question.exclude_from_history,
            image=image,
        )

        if isinstance(question, Question):
            if question.can_answer is False:
                return CustomTokens.no_answer

        answer = await self.receive_data(
            message_type_to_receive=MessageType.SURVEY_ITEM
            if question.survey_item
            else None
        )

        # TODO: if the answer is a special token, should it then be added to
        # the interview history?

        # if self.security_agent and not self.security_agent.is_safe(
        #     answer
        # ):
        #     message = "I'm sorry, but your last message is not within the scope of this interview. Please try again."
        #     await self.send_data(message)
        #     continue

        return answer

    async def ask_probe(self, question: Question, probe: str):
        message = await self.preprocess_message(
            probe,
            one_question=self.one_question,
        )

        if timed_message := self.interview_guide.timed_message:
            if self.time_spent > timed_message.time:
                await self.send_timed_message(timed_message)

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
                return CustomTokens.no_answer

        answer = await self.receive_data()

        # TODO: if the answer is a special token, should it then be added to
        # the interview history?

        # if self.security_agent and not self.security_agent.is_safe(
        #     answer
        # ):
        #     message = "I'm sorry, but your last message is not within the scope of this interview. Please try again."
        #     await self.send_data(message)
        #     continue

        return answer

        # FIXME:

    async def send_timed_message(self, timed_message: TimedMessage):
        # - This should also be stored in the database, so they wont be
        # send again if somebody reconnects.
        #   Currently this is being handled by looking at the time spent in
        #   last session but this is not bullet proof, so should update to look
        #   for the timed_message in the db.

        self.interview_guide.timed_message = None

        if timed_message.variables:
            timed_message.message = fill_variables_in_message(
                text=timed_message.message,
                variables=timed_message.variables,
                referable_values=self.referable_values,
            )

        timed_message_txt = await self.preprocess_message(timed_message.message)

        await self.send_data(
            timed_message_txt,
            can_answer=False,
            include_in_history=timed_message.include_in_history,
            with_interview_structure=False,
            timed=True,
        )
        # TODO: Move sleep to frontend
        await asyncio.sleep(2.5)

    async def check_condition(self, condition: Condition):
        question_context = condition.question_context

        conditional_context = self.interview_history[question_context.section]

        if (main_question_index := question_context.question) is not None:
            conditional_context = conditional_context[main_question_index]
        else:
            raise NotImplementedError(
                "This feature is not implemented yet. Please specify a main question index for your condition."
            )

        match question_context.part:
            case "main":
                if (main_answer := conditional_context.main_question.answer) is None:
                    raise ValueError("Answer not found")
                condition_question_context = main_answer.message
            case "probes":
                condition_question_context = [
                    probe.answer.message
                    for probe in conditional_context.probes
                    if probe.answer
                ]
            case "all":
                if (main_answer := conditional_context.main_question.answer) is None:
                    raise ValueError("Answer not found")
                condition_question_context = [main_answer.message]
                condition_question_context.extend(
                    [
                        probe.answer.message
                        for probe in conditional_context.probes
                        if probe.answer
                    ]
                )
            case _:
                raise ValueError("Invalid part")

        condition_triggered = await evaluate_condition(
            condition_question_context, condition.trigger_value, condition.trigger_type
        )

        self.db.insert_task(
            message_id=self.message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="evaulate_condition",
            content=condition.model_dump_json(),
            response=condition_triggered,
        )

        if condition_triggered:
            match condition.action:
                case ConditionAction.SKIP_SECTION:
                    raise SkipSectionException
                case ConditionAction.SKIP_QUESTION:
                    raise SkipQuestionException
                case ConditionAction.END_INTERVIEW:
                    raise EndInterviewException
                case _:
                    raise ValueError("Invalid condition action")

    def can_probe(self, question: Question) -> bool:
        if question.max_probes_n is not None:
            if self.interview_history.current_probe_index >= question.max_probes_n:
                return False

        if question.max_probes_time is not None:
            if question.max_probes_time <= time.time() - self.probing_time:
                return False

        return True

    async def probe(self, question: Question, section_description: str):
        # TODO:
        # - Should we also check if probes have been answered?

        self.probing_time = time.time()

        while self.can_probe(question):
            if self.interview_history.current_probe_index > 0:
                contains_multiple_refusals = await self.contains_multiple_refusals()
                if contains_multiple_refusals:
                    break

                if question.check_if_exhausted:
                    if await self.has_main_question_been_exhausted(question):
                        break

            if probing_context := question.probing_context:
                # TODO: Implement with new interview_history
                match probing_context:
                    case ContextType.QUESTION:
                        ...
                    case ContextType.SECTION:
                        ...

            transcript = self.interview_history.get_transcript()

            if (probes := question.probes) is not None:
                probes = "\n".join("- " + probe for probe in probes)

            probe = await self.probing_agent.generate_probe(
                section_description=section_description,
                question_description=question.description,
                main_question=question.main_question,
                transcript=transcript,
                probes=probes,
                translation=self.translation,
            )

            if probe.lower().startswith(CustomTokens.end_of_probe):
                break

            answer = await self.ask_probe(question, probe)

            if answer == CustomTokens.skip_question:
                # NOTE: skipping a probe skips the main question
                raise SkipQuestionException

    async def free_probing(self):
        raise NotImplementedError("This interview-loop is currently not optimized")

    async def has_question_been_answered(self, question: str) -> bool:
        start_time = time.time()
        transcript = self.interview_history.get_transcript(with_excludes=True)
        response = await self.classification_agent.classify(
            question,
            "has already been answered by the user",
            transcript,
        )

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.message_id + 1,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="has_question_been_answered",
            content=question,
            response=response,
            time_spend=time_spend,
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
            "has been extensively answered by the user to a satisfying degree",
            transcript,
        )

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="has_question_been_exhausted",
            content=context,
            response=response,
            time_spend=time_spend,
        )

        return response

    async def reformulate_question(
        self,
        question: Question,
        section_description: str,
        reason: Literal["already_answered", "segue", "skipped"],
    ) -> str:
        start_time = time.time()
        match reason:
            case "already_answered":
                reason_prompt = "That question has already been answered. Please respond with a reformulated version of the question and nothing else, while taking the previous answer into account"
                additional_guidelines = []
            case "segue":
                reason_prompt = "Question should be reforumlated if it improves the conversational flow. Draw on previous answers in a natural and general way if they are relevant"
                additional_guidelines = []
            case "skipped":
                reason_prompt = "The user tried to skip that question. Please respond with a reformulated version of the question and nothing else"
                additional_guidelines = []

        transcript = self.interview_history.get_transcript(with_excludes=True)

        probing_context: dict[str, str | list[str]] = {
            "section_description": section_description,
        }

        if question.description:
            probing_context["question_description"] = question.description

        probing_context["main_question"] = question.main_question
        if question.probes:
            probing_context["probes"] = question.probes

        prompt = self.probing_agent.prompts.get_template(
            "reformulation_prompt.jinja"
        ).render(
            interview_transcript=transcript,
            probing_context=json.dumps(probing_context),
            question=question.main_question,
            reason=reason_prompt,
            translation=self.translation,
            additional_guidelines=additional_guidelines,
        )

        self.probing_agent.logger.info(f"Reformulating question: {prompt}")
        message = await self.probing_agent.chat_api(
            [{"role": "user", "content": prompt}]
        )
        self.probing_agent.logger.info(f"Reformulating question: {message}")

        time_spend = time.time() - start_time

        self.db.insert_task(
            message_id=self.message_id,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="reformulate_question",
            reason=reason,
            content=question.main_question,
            response=message,
            time_spend=time_spend,
            context=prompt,
            model=self.probing_agent.model,
        )

        return message

    async def contains_multiple_refusals(self) -> bool:
        now = time.time()
        current_question_transcript = (
            self.interview_history.current_question.transcribe()
        )
        contains_multiple_refusals = await self.classification_agent.classify(
            current_question_transcript,
            "contains multiple answers (A:) from the respondent in a row that are explicit refusals to answer",
        )
        time_spend = time.time() - now

        self.db.insert_task(
            message_id=self.message_id + 1,
            interview_id=self.interview_id,
            project_id=self.project_id,
            task="contains_multiple_refusals",
            content=current_question_transcript,
            response=contains_multiple_refusals,
            time_spend=time_spend,
        )

        return contains_multiple_refusals
