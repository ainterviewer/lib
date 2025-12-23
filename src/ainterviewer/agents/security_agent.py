from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import SecurityAgentPrompts
from ainterviewer.exceptions import SecurityException
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
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]

    async def is_safe(self, message):
        last_question = self.shared_memory.ProbingAgent[-1]["content"]

        security_prompt = self.prompts.generate_security_prompt(last_question, message)

        messages = self.messages + [{"role": "user", "content": security_prompt}]

        response = await self.chat_api(
            messages, stop_tokens=[r"\.", r"\ ", ","], include_stop_token=False
        )
        response = response.lower().strip(".\n ")
        if response == "yes":
            return True
        elif response == "no":
            return False
        else:
            raise SecurityException(
                f"Expected 'yes' or 'no' from SecurityAgent, but got '{response}'"
            )
