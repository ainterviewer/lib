from functools import partial
from uuid import uuid4

import pytest

from ainterviewer.embedding import (
    CHUNK_FORMAT_VERSION,
    DEFAULT_POLICY,
    ChunkPolicy,
    DefaultChunkPolicy,
    chunks_from_history,
    interview_chunk,
    message_chunk,
    qa_pair_chunk,
    should_embed_message,
)
from ainterviewer.interview_guides.history import (
    HistoryMessage,
    InterviewHistory,
    QuestionHistory,
    SectionHistory,
    Turn,
)
from ainterviewer.interview_guides.survey_items import LikertItem
from ainterviewer.lpm.types import CustomToken
from ainterviewer.types import EmbeddingKind, MessageRole, MessageType

PROJECT_ID = uuid4()
INTERVIEW_ID = uuid4()


@pytest.fixture
def likert():
    return LikertItem(options=["Disagree", "Neutral", "Agree"])


def question(text: str, answer: str | None = None, survey_item=None):
    return QuestionHistory(
        description="d",
        main_question=Turn(
            question=HistoryMessage(message=text),
            answer=HistoryMessage(message=answer) if answer is not None else None,
            survey_item=survey_item,
        ),
    )


def history(*questions) -> InterviewHistory:
    section = SectionHistory(description="s")
    section.questions.extend(questions)
    return InterviewHistory(sections=[section])


# ── should_embed_message ─────────────────────────────────────────────────────


class TestShouldEmbedMessage:
    def test_respondent_free_text(self):
        assert should_embed_message(
            role="user", message_type="text", content="Busy but good."
        )

    def test_audio_transcript(self):
        assert should_embed_message(
            role="user", message_type="audio", content="Busy but good."
        )

    def test_interviewer_turn_excluded(self):
        assert not should_embed_message(
            role="assistant", message_type="text", content="How is work?"
        )

    def test_survey_answer_excluded(self):
        assert not should_embed_message(
            role="user", message_type="survey_item", content="Agree"
        )

    def test_control_token_excluded(self):
        assert not should_embed_message(
            role="user",
            message_type="text",
            content=str(CustomToken.skip_question),
        )

    def test_skipped_by_condition_excluded(self):
        assert not should_embed_message(
            role="user",
            message_type="text",
            content="Busy.",
            skipped_by_condition=True,
        )

    def test_blank_excluded(self):
        assert not should_embed_message(role="user", message_type="text", content="   ")


class TestMessageChunk:
    def test_carries_structural_coordinates(self):
        chunk = message_chunk(
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            message_id=7,
            content="  Busy but good.  ",
            role="user",
            message_type="text",
            section=1,
            main_question=2,
            sub_question=3,
        )
        assert chunk is not None
        assert chunk.kind == EmbeddingKind.MESSAGE
        assert chunk.text == "Busy but good."
        assert (chunk.message_id, chunk.section, chunk.main_question) == (7, 1, 2)
        assert chunk.sub_question == 3

    def test_returns_none_when_not_embeddable(self):
        assert (
            message_chunk(
                project_id=PROJECT_ID,
                interview_id=INTERVIEW_ID,
                message_id=1,
                content="Agree",
                role="user",
                message_type="survey_item",
            )
            is None
        )


# ── qa_pair_chunk ────────────────────────────────────────────────────────────


class TestQaPairChunk:
    def make(self, group):
        return qa_pair_chunk(
            group,
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            section=0,
            main_question=0,
        )

    def test_includes_probes(self):
        group = question("How is work?", "Busy but good.")
        group.add_probe(
            Turn(
                question=HistoryMessage(message="Why busy?"),
                answer=HistoryMessage(message="We shipped a release."),
            )
        )
        chunk = self.make(group)
        assert chunk is not None
        assert chunk.text == (
            "Q: How is work?\nA: Busy but good.\nQ: Why busy?\nA: We shipped a release."
        )

    def test_survey_only_group_excluded(self, likert):
        assert self.make(question("I like my job.", "Agree", likert)) is None

    def test_probed_survey_group_keeps_the_closed_answer_as_context(self, likert):
        group = question("I feel supported.", "Neutral", likert)
        group.add_probe(
            Turn(
                question=HistoryMessage(message="What would help?"),
                answer=HistoryMessage(message="Clearer priorities."),
            )
        )
        chunk = self.make(group)
        assert chunk is not None
        assert "A (likert): Neutral" in chunk.text
        assert "A: Clearer priorities." in chunk.text

    def test_unanswered_group_excluded(self):
        assert self.make(question("How is work?")) is None

    def test_control_token_answer_excluded(self):
        assert (
            self.make(question("Anything else?", str(CustomToken.skip_question)))
            is None
        )

    def test_skipped_by_condition_excluded(self):
        group = QuestionHistory(
            description="d",
            main_question=Turn(
                question=HistoryMessage(message="Skipped?", skipped_by_condition=True),
                answer=HistoryMessage(message="x"),
            ),
        )
        assert self.make(group) is None


class TestTranscribeStaysStableForAgents:
    """The agents' transcript must not change: survey labelling is opt-in."""

    def test_survey_labels_off_by_default(self, likert):
        group = question("I feel supported.", "Neutral", likert)
        assert "A: Neutral" in group.transcribe()
        assert "A (likert)" not in group.transcribe()

    def test_survey_labels_when_requested(self, likert):
        group = question("I feel supported.", "Neutral", likert)
        assert "A (likert): Neutral" in group.transcribe(with_survey_labels=True)


# ── whole-history assembly ───────────────────────────────────────────────────


class TestChunksFromHistory:
    def test_only_qualifying_groups(self, likert):
        chunks = chunks_from_history(
            history(
                question("How is work?", "Busy but good."),
                question("I like my job.", "Agree", likert),
                question("Anything else?", "Not really, no."),
            ),
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
        )
        pairs = [c for c in chunks if c.kind == EmbeddingKind.QA_PAIR]
        assert [c.main_question for c in pairs] == [0, 2]
        assert sum(c.kind == EmbeddingKind.INTERVIEW for c in chunks) == 1

    def test_survey_only_interview_produces_nothing(self, likert):
        assert (
            chunks_from_history(
                history(question("I like my job.", "Agree", likert)),
                project_id=PROJECT_ID,
                interview_id=INTERVIEW_ID,
            )
            == []
        )

    def test_interview_chunk_suppressed_without_free_text(self, likert):
        assert (
            interview_chunk(
                history(question("I like my job.", "Agree", likert)),
                project_id=PROJECT_ID,
                interview_id=INTERVIEW_ID,
            )
            is None
        )

    def test_language_is_carried_through(self):
        chunks = chunks_from_history(
            history(question("Hvordan går det?", "Fint nok.")),
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            language="DA",
        )
        assert chunks and all(c.language == "DA" for c in chunks)


# ── policy is replaceable ────────────────────────────────────────────────────


class SurveyInclusivePolicy(DefaultChunkPolicy):
    """A consumer studying closed-ended response patterns wants the opposite of
    the default: survey answers in, and labelled."""

    format_version = "survey-inclusive-1"

    def should_embed_message(self, *, message_type, **kwargs) -> bool:
        if MessageType(message_type) == MessageType.SURVEY_ITEM:
            return MessageRole(kwargs["role"]) == MessageRole.USER
        return super().should_embed_message(message_type=message_type, **kwargs)

    def should_embed_question(self, question) -> bool:
        if question.main_question.question.skipped_by_condition:
            return False
        return any(turn.answer is not None for turn in question.turns)


class TestChunkPolicy:
    def test_default_satisfies_the_protocol(self):
        assert isinstance(DEFAULT_POLICY, ChunkPolicy)

    def test_default_stamps_the_format_version(self, likert):
        chunks = chunks_from_history(
            history(question("How is work?", "Busy but good.")),
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
        )
        assert chunks
        assert all(c.format_version == CHUNK_FORMAT_VERSION for c in chunks)

    def test_custom_policy_includes_survey_only_groups(self, likert):
        group = question("I like my job.", "Agree", likert)
        chunk_for = partial(
            qa_pair_chunk,
            group,
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            section=0,
            main_question=0,
        )

        assert chunk_for() is None
        chunk = chunk_for(policy=SurveyInclusivePolicy())
        assert chunk is not None
        assert "A (likert): Agree" in chunk.text

    def test_custom_policy_stamps_its_own_version(self, likert):
        chunk = qa_pair_chunk(
            question("I like my job.", "Agree", likert),
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            section=0,
            main_question=0,
            policy=SurveyInclusivePolicy(),
        )
        assert chunk is not None
        assert chunk.format_version == "survey-inclusive-1"

    def test_custom_policy_reaches_message_chunks(self):
        chunk_for = partial(
            message_chunk,
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
            message_id=1,
            content="Agree",
            role="user",
            message_type="survey_item",
        )

        assert chunk_for() is None
        assert chunk_for(policy=SurveyInclusivePolicy()) is not None

    def test_custom_policy_reaches_interview_chunks(self, likert):
        survey_only = history(question("I like my job.", "Agree", likert))
        chunk_for = partial(
            interview_chunk,
            survey_only,
            project_id=PROJECT_ID,
            interview_id=INTERVIEW_ID,
        )

        assert chunk_for() is None
        assert chunk_for(policy=SurveyInclusivePolicy())
