from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ainterviewer.lpm.types import Temperature
from ainterviewer.settings import settings


class AgentConfigs(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    probing: ProbingAgentConfig = Field(default_factory=lambda: ProbingAgentConfig())
    classification: AgentConfig = Field(
        default_factory=lambda: AgentConfig(temperature=0.0)
    )
    guide: AgentConfig = Field(default_factory=lambda: AgentConfig())
    history: AgentConfig = Field(default_factory=lambda: AgentConfig())
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig())
    answering: AgentConfig = Field(default_factory=lambda: AgentConfig())
    reformulation: AgentConfig = Field(default_factory=lambda: AgentConfig())
    visual: VisualConfig = Field(default_factory=lambda: VisualConfig())


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    model: str = settings.llm.default_model
    temperature: Temperature = Field(default=0.7)
    include: bool = True

    @property
    def chat_kwargs(self) -> dict:
        return {"temperature": self.temperature}


class ProbingPromptSlots(BaseModel):
    """User-editable regions of the probing agent's system and instruction prompts.

    Each field is an override: when left as ``None`` the corresponding default
    from ``DEFAULT_PROBING_SLOTS`` is used at render time, so improvements to the
    defaults flow through to projects that have not customized that slot.
    """

    model_config = ConfigDict(extra="forbid")

    persona: str | None = None
    question_qualities: list[str] | None = None
    guidelines: list[str] | None = None
    instructions: list[str] | None = None

    def resolved(self) -> "ProbingPromptSlots":
        """Return a copy with every unset slot filled from the defaults."""
        return ProbingPromptSlots(
            persona=self.persona or DEFAULT_PROBING_SLOTS.persona,
            question_qualities=self.question_qualities
            or DEFAULT_PROBING_SLOTS.question_qualities,
            guidelines=self.guidelines or DEFAULT_PROBING_SLOTS.guidelines,
            instructions=self.instructions or DEFAULT_PROBING_SLOTS.instructions,
        )


DEFAULT_PROBING_SLOTS = ProbingPromptSlots(
    persona=(
        "You are an experienced qualitative social science interviewer "
        "specializing in conducting in-depth, semi-structured interviews. Your "
        "role is to ask insightful, relevant follow-up questions based on the "
        "context, transcript, and information provided by the user."
    ),
    question_qualities=[
        "Conversational and natural-sounding",
        "Open-ended to encourage detailed responses",
        "Neutral and unbiased",
        "Non-leading and non-suggestive",
        "Non-judgmental and respectful",
        "Clear and concise",
        "Relevant to the research topic and previous responses",
        "Designed to elicit rich, descriptive answers",
    ],
    guidelines=[
        "Ask only one question at a time to maintain focus",
        "Use clear and simple language to avoid misinterpretation in text form",
        'Use text-based probing techniques (e.g., "Could you elaborate on '
        'that?", "What do you mean by...?")',
        "You can also provide clarifications if the interviewee explicitly asks for them",
        "Be attentive to the tone and emotion conveyed through text",
    ],
    instructions=[
        "Aim to elicit new information from the interviewee",
        "Avoid repeating previous questions or topics already covered",
        "You must vary the formulation of the question to avoid repetition "
        "compared to the previously asked questions",
        "Focus on specific actions, events or experiences when relevant",
        "Ask the respondent questions that allow them to draw on their expert "
        "knowledge",
        "Seek enough detail to create a mental image of a moment or an experience",
        "Avoid asking leading questions",
        "Do not ask them to talk about what other people think",
        "Ask open-ended questions that encourage elaboration",
        "Do not provide direct advice or feedback in your response",
        "If the respondent asks for clarification of a concept or the meaning of "
        "the question, provide it succinctly as an answer",
        "Look for interesting and important formulations in the prior answer and "
        "ask questions to these",
        "Look for clues for when the respondents has something to say and ask "
        "questions to these aspects",
        "Ask only one question at a time",
        "Keep your questions brief",
        "Prefer to ask short questions, your reply should ideally be one sentence",
        "Reply only with the question itself without any commentary",
    ],
)


class ProbingAgentConfig(AgentConfig):
    few_shot_examples: list[str] | None = None
    prompt_slots: ProbingPromptSlots = Field(default_factory=ProbingPromptSlots)


class SecurityConfig(AgentConfig):
    sensitive_subjects: list | None = None
    include: bool = False

    @model_validator(mode="after")
    def check_sensitive_subjects(self):
        if self.include and not self.sensitive_subjects:
            raise ValueError(
                "'sensitive_subjects' must be provided when include is True"
            )
        return self


class VisualConfig(AgentConfig):
    model: str = "llava"
    include: bool = False
