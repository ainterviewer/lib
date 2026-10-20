import json

from pydantic import BaseModel, ValidationError

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

#: How many times an assessment is asked for before giving up. A reply that
#: does not fit the assessment model is usually a one-off -- the model echoing
#: the shape it was shown rather than filling it in -- and asking again is
#: cheaper than losing the check.
ASSESS_ATTEMPTS = 2


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

    def _assessment_questions(self) -> str:
        return "\n".join(
            f"- `{decision.name}`: {decision.description}"
            for decision in self.policy.decisions
        )

    def _assessment_example(self) -> str:
        """An instance of the assessment, not its JSON Schema.

        Shown the schema, models served without strict structured output
        (gpt-oss through OpenRouter) now and then answer with the schema
        itself -- `$defs`, `properties` and all -- which validates as nothing.
        The questions the schema's descriptions carried are listed separately.
        """
        return json.dumps(
            {
                decision.name: {"reasoning": "...", "probability": 0.0}
                for decision in self.policy.decisions
            },
            indent=2,
        )

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
            assessment_questions=self._assessment_questions(),
            assessment_example=self._assessment_example(),
        )

        messages: list[Message] = self.messages + [
            Message(role=MessageRole.USER, content=security_prompt)
        ]

        assessment = await self._request_assessment(messages, assessment_model)

        triggered: list[TriggeredDecision] = []
        for decision in self.policy.decisions:
            evaluation: SecurityEvaluation = getattr(assessment, decision.name)
            if evaluation.probability >= decision.threshold:
                triggered.append(TriggeredDecision(decision, evaluation))

        return SecurityAssessmentResult(assessment=assessment, triggered=triggered)

    async def _request_assessment(
        self, messages: list[Message], assessment_model: type[BaseModel]
    ) -> BaseModel:
        """Asks for the assessment, again if the reply does not fit the model.

        Raises the last failure once the attempts run out.
        """
        error: Exception | None = None

        for _ in range(ASSESS_ATTEMPTS):
            try:
                assessment = await self.chat_api(
                    messages, response_format=assessment_model
                )
            except ValidationError as exception:
                error = exception
                continue

            # A refusal parses to nothing rather than raising.
            if assessment is not None:
                return assessment

            error = ValueError("The security agent returned no assessment")

        assert error is not None
        raise error
