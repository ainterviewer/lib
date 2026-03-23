from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import SecurityAgentPrompts
from ainterviewer.lpm.types import Message
from ainterviewer.types import MessageRole


class SecurityAgent(BaseAgent[SecurityAgentPrompts]):
    """Agent that monitors the security of the text"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.messages += [
            Message(role=MessageRole.SYSTEM, content=self.prompts.system_prompt),
        ]

    async def is_safe(self, question: str, answer: str):
        security_prompt = self.prompts.generate_security_prompt(question, answer)

        messages: list[Message] = self.messages + [
            Message(role=MessageRole.USER, content=security_prompt)
        ]

        response = await self.chat_api(messages)
        response = response.lower().strip(".\n ")
