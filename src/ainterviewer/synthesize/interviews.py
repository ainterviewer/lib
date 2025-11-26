import asyncio
import json
import random
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

from jinja2 import DictLoader
from pydantic import UUID4

from ainterviewer.agents import AnsweringAgent
from ainterviewer.config import read_agent_configs, read_interview_config
from ainterviewer.interfaces import IOProtocol, OutgoingData, OutgoingMessage
from ainterviewer.interview import AInterviewer
from ainterviewer.interview_guides.interview_guide import Image, InterviewGuideContent
from ainterviewer.interview_guides.survey_item import SurveyItem
from ainterviewer.lpm.clients import chat
from ainterviewer.prompts.models import DEFAULT_PROMPTS
from ainterviewer.synthesize.interviewees import (
    BackgroundInfoOptions,
    generate_synthetic_persons,
)
from ainterviewer.types import LanguageCode, MessageRole, MessageType


class SynytheticIOProtocol(IOProtocol):
    def __init__(
        self,
        answering_agent: AnsweringAgent,
        delay_before_answer: tuple[float, float] | None = None,
    ):
        self.answering_agent = answering_agent
        self._message_queue: asyncio.Queue[tuple[str, MessageType]] = asyncio.Queue()
        self.delay_before_answer = delay_before_answer

    async def send_data(self, data: OutgoingData | OutgoingMessage) -> None:
        match data.type:
            case "data":
                pass

            case "message":
                print(data.content)
                if not data.can_answer:
                    self.answering_agent.messages.append(
                        {"role": MessageRole.USER, "content": data.content}
                    )
                else:
                    response_text = await self.answering_agent.answer(data.content)
                    print(response_text)
                    await self._message_queue.put((response_text, MessageType.TEXT))

    async def receive_message(
        self, message_type: MessageType | None = None
    ) -> tuple[str, MessageType]:
        while True:
            if self.delay_before_answer:
                time.sleep(random.uniform(*self.delay_before_answer))

            return await self._message_queue.get()


async def synthesize(
    project_id: UUID4,
    background_info_options: BackgroundInfoOptions,
    n_interviews: int,
    answering_model: str,
    language: LanguageCode = "EN",
    interview_config_path: Path | str = "config.yaml",
    interview_guide_path: Path | str = "interview_guide.json",
    agent_configs_path: Path | str = "agents.yaml",
    delay_before_answer: tuple[float, float] | None = None,
):
    subjects = generate_synthetic_persons(background_info_options, n_interviews)

    # Create agents
    agents = []
    for subject in subjects:
        agent = AnsweringAgent(
            model=answering_model,
            chat_api=lambda messages, **kwargs: chat(
                messages, model=answering_model, **kwargs
            ),
            interview_subject=subject,
            language=language if language != "EN" else None,
        )
        agents.append(agent)

    # Run agents concurrently but with a small delay between starts
    tasks = []
    for agent in agents:
        task = asyncio.create_task(
            run_synthetic_answering_agent(
                agent=agent,
                project_id=project_id,
                language=language,
                interview_config_path=interview_config_path,
                interview_guide_path=interview_guide_path,
                agent_configs_path=agent_configs_path,
                delay_before_answer=delay_before_answer,
            )
        )
        tasks.append(task)
        await asyncio.sleep(1)  # Small delay between starting agents

    # Wait for all agents to complete
    results = await asyncio.gather(
        *tasks,
        # return_exceptions=True,
    )

    return results


class DB:
    def update_interview_status(
        self,
        project_id: UUID4,
        interview_id: UUID4,
        is_active: bool | None = None,
        is_complete: bool | None = None,
        time_spent: int = 0,
    ): ...

    def insert_message(
        self,
        message_id: int,
        content: str,
        role: MessageRole,
        interview_id: UUID4,
        project_id: UUID4,
        message_type: MessageType = MessageType.TEXT,
        can_answer: bool = True,
        include_in_history: bool = True,
        attachment: Path | None = None,
        survey_item: SurveyItem | None = None,
        image: Optional[Image | list[Image]] = None,
        section: Optional[int] = None,
        main_question: Optional[int] = None,
        sub_question: Optional[int] = None,
        is_introduction: bool = False,
        outro: bool = False,
        timed: bool = False,
    ) -> int:
        return message_id

    def insert_task(
        self,
        message_id: int,
        interview_id: UUID4,
        project_id: UUID4,
        task: str,
        reason: Optional[str] = None,
        context: Optional[str] = None,
        content: Optional[str] = None,
        response: Optional[str] = None,
        model: Optional[str] = None,
        time_spend: Optional[int] = None,
    ): ...

    def save_image(self, image: Image): ...


async def run_synthetic_answering_agent(
    agent: AnsweringAgent,
    project_id: UUID4,
    language: LanguageCode = "EN",
    interview_config_path: Path | str = "config.yaml",
    interview_guide_path: Path | str = "interview_guide.json",
    agent_configs_path: Path | str = "agents.yaml",
    delay_before_answer: tuple[float, float] | None = None,
):
    synth_io = SynytheticIOProtocol(agent, delay_before_answer)

    interview_id = uuid4()
    message_id = 0

    db = DB()

    prompt_loader = DictLoader(DEFAULT_PROMPTS.dump_templates())

    interview_config = read_interview_config(interview_config_path)

    with Path(interview_guide_path).open() as f:
        interview_guide = InterviewGuideContent.model_validate(json.load(f))

    agent_configs = read_agent_configs(agent_configs_path)

    async with AInterviewer(
        io=synth_io,
        db=db,
        interview_guide=interview_guide,
        config=interview_config,
        agent_configs=agent_configs,
        template_loader=prompt_loader,
        project_id=project_id,
        interview_id=interview_id,
        message_id=message_id,
        frontend_language=language,
        referable_values={"test": True},
    ) as interviewer:
        await interviewer.interview(probing="restricted")
