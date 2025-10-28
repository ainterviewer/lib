import re

from ainterviewer.agents.base import BaseAgent
from ainterviewer.exceptions import ClassificationError
from ainterviewer.prompts.agent_prompts import ClassificationAgentPrompts
from ainterviewer.types import MessageRole


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
        guided_choice: list[str] = ["0", "1"],
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

        messages = self.messages + [{"role": "user", "content": message}]
        response = await self.chat_api(messages, guided_choice=guided_choice)

        self.logger.info("classification response", context=response)

        try:
            return bool(int(response))
        except ValueError:
            if unsafe:
                numbers = set(re.findall(r"\d+", response))
                if len(numbers) > 1:
                    raise ClassificationError(
                        "Expected a single number in the response, but got multiple"
                    )
                elif len(numbers) == 0:
                    raise ClassificationError(
                        "Expected a single number in the response, but got none"
                    )
                return bool(int(numbers.pop()))
            else:
                raise ClassificationError(
                    "Response consisted of more than a single digit which is needed to classify the text"
                )
