import json

from ainterviewer.agents.base import BaseAgent
from ainterviewer.agents.prompts.agent_prompts import SecurityAgentPrompts
from ainterviewer.agents.security_policy import (
    SecurityAssessmentResult,
    SecurityEvaluation,
    SecurityPolicy,
    TriggeredDecision,
)
from ainterviewer.lpm.types import Message
from ainterviewer.types import MessageRole


class SecurityAgent(BaseAgent[SecurityAgentPrompts]):
    """Agent that assesses the interview transcript against a security policy"""

    def __init__(
        self,
        policy: SecurityPolicy,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy
        self.messages += [
            Message(role=MessageRole.SYSTEM, content=self.prompts.system_prompt),
        ]

    async def assess(
        self,
        interview_framing: str,
        section_description: str,
        question_description: str,
        transcript: str,
    ) -> SecurityAssessmentResult:
        assessment_model = self.policy.assessment_model

        security_prompt = self.prompts.generate_security_prompt(
            interview_framing=interview_framing,
            section_description=section_description,
            question_description=question_description,
            transcript=transcript,
            security_assessment=json.dumps(
                assessment_model.model_json_schema(), indent=2
            ),
        )

        messages: list[Message] = self.messages + [
            Message(role=MessageRole.USER, content=security_prompt)
        ]

        assessment = await self.chat_api(messages, response_format=assessment_model)

        triggered: list[TriggeredDecision] = []
        for decision in self.policy.decisions:
            evaluation: SecurityEvaluation = getattr(assessment, decision.name)
            if evaluation.probability >= decision.threshold:
                triggered.append(TriggeredDecision(decision, evaluation))

        return SecurityAssessmentResult(assessment=assessment, triggered=triggered)
