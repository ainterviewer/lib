from ainterviewer.agents.base import BaseAgent
from ainterviewer.prompts.agent_prompts import HistoryAgentPrompts
from ainterviewer.types import MessageRole


class HistoryAgent(BaseAgent[HistoryAgentPrompts]):
    """Agent that keeps track of the interview history through summarization"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.messages += [
            {"role": MessageRole.SYSTEM, "content": self.prompts.system_prompt},
        ]
