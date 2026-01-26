from typing import Literal

from pydantic import BaseModel, create_model

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import ClassificationAgentPrompts
from ainterviewer.lpm.types import Message
from ainterviewer.types import MessageRole


class BinaryClassificationModel(BaseModel):
    value: Literal[0, 1]


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
        response = list[Literal[*options]]
    else:
        response = Literal[*options]

    return create_model(name, response=response)


def generate_scoring_model(
    name: str,
    options: list[int],
):
    values = Literal[*options]
    return create_model(name, value=values)


class ClassificationAgent(BaseAgent[ClassificationAgentPrompts]):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.messages += [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt}
        ]

    async def classify(
        self,
        text: str,
        next_question_instruction: str,
        interview_history: str | None = None,
        classification_examples=None,
        unsafe: bool = True,
    ) -> bool:
        """
        The model will try to parse the response as an int and return it as a boolean.
            If this fails and unsafe is True [default], it will look for the first digit and parse that instead.
        """
        message = self.prompts.genererate_classification_prompt(
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

        return bool(response.value)
