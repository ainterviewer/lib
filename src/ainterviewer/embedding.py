"""Which interview text is worth embedding, and how it is rendered.

Assembly lives here because it needs library internals -- walking
`InterviewHistory`, grouping probes with their main question, rendering the
turns -- and because `AInterviewer` emits chunks while an interview is still
running, where it cannot call into a consumer.

*Policy* is a different matter and is deliberately not baked in. Whether a
closed-ended survey answer deserves a vector is a research-methodology
judgement, not a fact about running interviews: the default here excludes them,
because near-identical vectors drawn from a handful of option strings crowd out
real answers in nearest-neighbour results, but a consumer studying response
patterns would want the opposite. Every such decision sits on `ChunkPolicy`, so
it can be replaced without forking this module. Pass a policy to `AInterviewer`
and to `chunks_from_history`; omit it and `DEFAULT_POLICY` applies.

`ChunkPolicy.format_version` rides along on every chunk and is meant to be
stored beside the vector. Rendering and inclusion rules are what a stored
embedding is a function of, alongside the model itself -- so changing them has
to be *detectable*, or a consumer silently ends up with a corpus that is half
one format and half another and no way to tell which. Bump it whenever a policy
changes what text comes out or which chunks appear, and re-embedding becomes a
question the consumer's backfill can answer instead of a mystery in its search
results.
"""

from typing import Protocol, runtime_checkable

from pydantic import UUID4

from ainterviewer.interfaces import EmbeddingChunk
from ainterviewer.interview_guides.history import InterviewHistory, QuestionHistory
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import EmbeddingKind, LanguageCode, MessageRole, MessageType

#: Bump when `DefaultChunkPolicy` changes what it emits or how it renders.
CHUNK_FORMAT_VERSION = "1"

EMBEDDABLE_MESSAGE_TYPES = frozenset({MessageType.TEXT, MessageType.AUDIO})


@runtime_checkable
class ChunkPolicy(Protocol):
    """The decisions that make a corpus what it is: what goes in, and how it
    reads. Implement it to embed a different slice of the same interviews."""

    @property
    def format_version(self) -> str:
        """Identifies this policy's output format. Stored with every vector."""
        ...

    def should_embed_message(
        self,
        *,
        role: MessageRole | str,
        message_type: MessageType | str,
        content: str,
        skipped_by_condition: bool = False,
    ) -> bool: ...

    def should_embed_question(self, question: QuestionHistory) -> bool: ...

    def should_embed_interview(self, history: InterviewHistory) -> bool: ...

    def render_message(self, content: str) -> str: ...

    def render_question(self, question: QuestionHistory) -> str: ...

    def render_interview(self, history: InterviewHistory) -> str: ...


class DefaultChunkPolicy:
    """The rules a qualitative-analysis corpus wants.

    What it leaves out:

    - Interviewer turns on their own. A question is recoverable through the QA
      pair it belongs to; alone it says what the guide asked, not what anyone
      answered.
    - Closed-ended survey answers. Structured data already -- count them, don't
      embed them. They still appear *inside* a QA pair as context when the group
      also drew free text out of the respondent.
    - Anything skipped by a condition, and the control tokens.
    """

    format_version: str = CHUNK_FORMAT_VERSION

    def should_embed_message(
        self,
        *,
        role: MessageRole | str,
        message_type: MessageType | str,
        content: str,
        skipped_by_condition: bool = False,
    ) -> bool:
        if MessageRole(role) != MessageRole.USER:
            return False
        if MessageType(message_type) not in EMBEDDABLE_MESSAGE_TYPES:
            return False
        if skipped_by_condition:
            return False
        text = content.strip()
        return bool(text) and text not in CustomToken

    def should_embed_question(self, question: QuestionHistory) -> bool:
        if question.main_question.question.skipped_by_condition:
            return False
        return question.has_free_text_answer

    def should_embed_interview(self, history: InterviewHistory) -> bool:
        """Only interviews that drew free text somewhere.

        Without this an interview made entirely of survey items gets a vector
        representing its interviewer script.
        """
        return any(
            self.should_embed_question(question)
            for section in history.sections
            for question in section.questions
        )

    def render_message(self, content: str) -> str:
        return content.strip()

    def render_question(self, question: QuestionHistory) -> str:
        return question.transcribe(
            with_descriptions=False,
            with_image=True,
            with_survey_labels=True,
        ).strip()

    def render_interview(self, history: InterviewHistory) -> str:
        return history.get_transcript(with_excludes=True).strip()


DEFAULT_POLICY: ChunkPolicy = DefaultChunkPolicy()


def should_embed_message(
    *,
    role: MessageRole | str,
    message_type: MessageType | str,
    content: str,
    skipped_by_condition: bool = False,
    policy: ChunkPolicy = DEFAULT_POLICY,
) -> bool:
    """Whether a single stored message earns a vector of its own."""
    return policy.should_embed_message(
        role=role,
        message_type=message_type,
        content=content,
        skipped_by_condition=skipped_by_condition,
    )


def message_chunk(
    *,
    project_id: UUID4,
    interview_id: UUID4,
    message_id: int,
    content: str,
    role: MessageRole | str,
    message_type: MessageType | str,
    language: LanguageCode = "EN",
    section: int | None = None,
    main_question: int | None = None,
    sub_question: int | None = None,
    skipped_by_condition: bool = False,
    policy: ChunkPolicy = DEFAULT_POLICY,
) -> EmbeddingChunk | None:
    """A chunk for one respondent message, or None if the policy excludes it."""
    if not policy.should_embed_message(
        role=role,
        message_type=message_type,
        content=content,
        skipped_by_condition=skipped_by_condition,
    ):
        return None

    text = policy.render_message(content)
    if not text:
        return None

    return EmbeddingChunk(
        kind=EmbeddingKind.MESSAGE,
        text=text,
        format_version=policy.format_version,
        project_id=project_id,
        interview_id=interview_id,
        language=language,
        message_id=message_id,
        section=section,
        main_question=main_question,
        sub_question=sub_question,
    )


def qa_pair_chunk(
    question: QuestionHistory,
    *,
    project_id: UUID4,
    interview_id: UUID4,
    section: int,
    main_question: int,
    language: LanguageCode = "EN",
    policy: ChunkPolicy = DEFAULT_POLICY,
) -> EmbeddingChunk | None:
    """A chunk for one question group: the main question, its answer, and every
    probe that followed. None when the policy excludes the group.
    """
    if not policy.should_embed_question(question):
        return None

    text = policy.render_question(question)
    if not text:
        return None

    return EmbeddingChunk(
        kind=EmbeddingKind.QA_PAIR,
        text=text,
        format_version=policy.format_version,
        project_id=project_id,
        interview_id=interview_id,
        language=language,
        section=section,
        main_question=main_question,
    )


def interview_chunk(
    history: InterviewHistory,
    *,
    project_id: UUID4,
    interview_id: UUID4,
    language: LanguageCode = "EN",
    policy: ChunkPolicy = DEFAULT_POLICY,
) -> EmbeddingChunk | None:
    """A chunk for the interview as a whole."""
    if not policy.should_embed_interview(history):
        return None

    text = policy.render_interview(history)
    if not text:
        return None

    return EmbeddingChunk(
        kind=EmbeddingKind.INTERVIEW,
        text=text,
        format_version=policy.format_version,
        project_id=project_id,
        interview_id=interview_id,
        language=language,
    )


def chunks_from_history(
    history: InterviewHistory,
    *,
    project_id: UUID4,
    interview_id: UUID4,
    language: LanguageCode = "EN",
    with_interview_chunk: bool = True,
    policy: ChunkPolicy = DEFAULT_POLICY,
) -> list[EmbeddingChunk]:
    """Every QA-pair chunk in a reconstructed history, plus the interview-level
    one. Message-level chunks are not produced here: `InterviewHistory` holds no
    message ids, so callers that have the stored rows build those with
    `message_chunk` instead.
    """
    chunks: list[EmbeddingChunk] = []

    for section_index, section in enumerate(history.sections):
        for question_index, question in enumerate(section.questions):
            chunk = qa_pair_chunk(
                question,
                project_id=project_id,
                interview_id=interview_id,
                section=section_index,
                main_question=question_index,
                language=language,
                policy=policy,
            )
            if chunk is not None:
                chunks.append(chunk)

    if with_interview_chunk and (
        chunk := interview_chunk(
            history,
            project_id=project_id,
            interview_id=interview_id,
            language=language,
            policy=policy,
        )
    ):
        chunks.append(chunk)

    return chunks
