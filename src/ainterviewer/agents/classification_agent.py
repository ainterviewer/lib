from typing import Literal

from pydantic import BaseModel, create_model

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import ClassificationAgentPrompts
from ainterviewer.lpm.types import Message
from ainterviewer.types import MessageRole


class BinaryClassificationModel(BaseModel):
    classification: bool


def generate_classification_model(
    name: str,
    options: list[str],
    multilable: bool = False,
) -> type[BaseModel]:
    """
    Usage:
        CarColor = generate_classification_model("CarColor", ["red", "blue", "green"])
        CarColor(response="red")
    """
    if multilable:
        response = list[Literal[*options]]  # ty: ignore[invalid-type-form]
    else:
        response = Literal[*options]  # ty: ignore[invalid-type-form]

    return create_model(name, response=response)


def generate_scoring_model(
    name: str,
    options: list[int],
):
    values = Literal[*options]  # ty: ignore[invalid-type-form]
    return create_model(name, value=values)


class ClassificationAgent(BaseAgent[ClassificationAgentPrompts]):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.messages += [
            Message(role=MessageRole.SYSTEM, content=self.prompts.system_prompt)
        ]

    async def classify(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples=None,
    ) -> bool:
        message = self.prompts.generate_classification_prompt(
            text,
            next_question_instruction,
            interview_history,
            classification_examples,
        )

        self.logger.info("classifying text", context=message)

        messages: list[Message] = self.messages + [
            {"role": MessageRole.USER, "content": message}
        ]

        response = await self.chat_api(
            messages, response_format=BinaryClassificationModel
        )

        self.logger.info("classification response", context=response)

        return response.classification

    async def classify_multi(
        self,
        text: str,
        options: list[str],
        multilabel: bool = False,
    ) -> BaseModel:
        raise NotImplementedError

    async def score(
        self,
        text: str,
        options: list[int],
    ) -> BaseModel:
        raise NotImplementedError
