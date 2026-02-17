from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

type SectionsRange = int | slice | list[int] | None


class DictLikeModel:
    def get(self, key: str, default: Any = None) -> Any:
        """
        Implements dictionary-like .get() method for Pydantic models
        """
        return getattr(self, key, default)


class HistoryMessage(BaseModel):
    message: str
    skipped_by_condition: bool = False


class Turn(BaseModel):
    question: HistoryMessage
    answer: HistoryMessage | None = None


class InterviewHistory(BaseModel):
    introduction: HistoryMessage | None = None
    sections: list[SectionHistory] = Field(default_factory=list)
    outro: HistoryMessage | None = None
    timed_messages: list[HistoryMessage] = Field(default_factory=list)

    @property
    def current_section(self) -> SectionHistory:
        return self.sections[-1]

    @property
    def current_question(self) -> QuestionHistory:
        return self.current_section.questions[-1]

    @property
    def current_section_index(self) -> int:
        return max(len(self.sections) - 1, 0)

    @property
    def current_question_index(self) -> int:
        try:
            return max(len(self.current_section.questions) - 1, 0)
        except IndexError:
            return 0

    @property
    def current_probe_index(self) -> int:
        try:
            return max(len(self.current_question.probes), 0)
        except IndexError:
            return 0

    @property
    def n_questions(self) -> int:
        return sum([len(section.questions) for section in self.sections])

    @property
    def current_message_id(self) -> int:
        count = 0

        if self.introduction:
            count += 1

        count += len(self.timed_messages)

        for section in self.sections:
            for question in section.questions:
                count += 1
                if question.main_question.answer:
                    count += 1

                for probe in question.probes:
                    count += 1
                    if probe.answer:
                        count += 1

        if self.outro:
            count += 1

        return count

    def __getitem__(self, key: int) -> SectionHistory:
        return self.sections[key]

    def add_section(self, section_description: str):
        section = SectionHistory(description=section_description)
        self.sections.append(section)

        return section

    def add_question(
        self,
        question_description: str | None,
        main_question: Turn,
        image: ImageHistory | None = None,
        exclude_from_history: bool = False,
    ):
        question = QuestionHistory(
            description=question_description,
            main_question=main_question,
            image=image,
            exclude_from_history=exclude_from_history,
        )
        self.current_section.questions.append(question)

        return question

    def add_probe(self, probe: Turn):
        self.current_question.probes.append(probe)

    def add_answer(self, answer: HistoryMessage):
        if self.current_question.probes:
            self.current_question.probes[-1].answer = answer
        else:
            self.current_question.main_question.answer = answer

    def get_transcript(
        self,
        section_range: SectionsRange = None,
        with_introduction: bool = True,
        with_descriptions: bool = False,
        with_images: bool = True,
        with_excludes: bool = False,
    ) -> str:
        sections = self._get_sections(section_range)

        transcript = ""

        if with_introduction and (introduction := self.introduction):
            transcript += "Q: " + introduction.message + "\n\n"

        for section in sections:
            transcript += (
                section.transcribe(
                    with_descriptions=with_descriptions,
                    with_images=with_images,
                    with_excludes=with_excludes,
                )
                + "\n"
            )

        if self.outro:
            transcript += self.outro.message + "\n\n"

        return transcript.strip()

    def _get_sections(
        self, section_range: SectionsRange = None
    ) -> list[SectionHistory]:

        if section_range is None:
            sections = self.sections
        elif isinstance(section_range, int):
            sections = [self.sections[section_range]]
        elif isinstance(section_range, slice):
            sections = self.sections[section_range]
        elif isinstance(section_range, (list, tuple)):
            sections = [self.sections[i] for i in section_range]
        else:
            raise TypeError(f"Unsupported section_range type: {type(section_range)}")

        return sections


class SectionHistory(BaseModel):
    description: str
    questions: list[QuestionHistory] = Field(default_factory=list)

    def __getitem__(self, key: int) -> QuestionHistory:
        return self.questions[key]

    def add_question(
        self,
        question_description: str,
        main_question: Turn,
        image: ImageHistory | None = None,
        exclude_from_history: bool = False,
    ):
        question = QuestionHistory(
            description=question_description,
            main_question=main_question,
            image=image,
            exclude_from_history=exclude_from_history,
        )
        self.questions.append(question)

        return question

    def transcribe(
        self,
        with_descriptions: bool = False,
        with_images: bool = True,
        with_excludes: bool = False,
    ) -> str:
        transcript = ""

        if with_descriptions:
            transcript += self.description + "\n\n"

        for i, question in enumerate(self.questions):
            # Exclude questions skipped by condition
            if question.main_question.question.skipped_by_condition:
                continue

            if (
                with_excludes
                # Always add the transcript of the last question:
                or i == len(self.questions) - 1
                or not question.exclude_from_history
            ):
                transcript += (
                    question.transcribe(
                        with_descriptions=with_descriptions,
                        with_image=with_images,
                    )
                    + "\n"
                )

        return transcript.strip()


class QuestionHistory(BaseModel):
    description: str | None
    main_question: Turn
    probes: list[Turn] = Field(default_factory=list)
    image: ImageHistory | None = None
    exclude_from_history: bool = False

    @property
    def answers(self) -> list[HistoryMessage]:
        answers = [answer] if (answer := self.main_question.answer) else []
        answers.extend([answer for probe in self.probes if (answer := probe.answer)])
        return answers

    @property
    def questions(self) -> list[HistoryMessage]:
        questions = [self.main_question.question] + [
            question for probe in self.probes if (question := probe.question)
        ]
        return questions

    @property
    def turns(self) -> list[Turn]:
        return [self.main_question] + self.probes

    def __getitem__(self, key: int) -> Turn:
        return self.probes[key]

    def add_probe(self, probe: Turn):
        self.probes.append(probe)

    def transcribe(
        self,
        with_descriptions: bool = False,
        with_image: bool = True,
    ) -> str:
        transcript = ""

        if with_descriptions and self.description:
            transcript += self.description + "\n\n"

        if with_image and self.image:
            if primer := self.image.primer:
                transcript += f"Q: {primer.message}\n"
            transcript += (
                "Q: An image has been shown to the user with the following description:\n"
                f"{self.image.description.message}\n"
            )

        transcript += "Q: " + self.main_question.question.message + "\n"
        if answer := self.main_question.answer:
            transcript += "A: " + answer.message + "\n"
        else:
            transcript += "\n"

        for probe in self.probes:
            transcript += "Q: " + probe.question.message + "\n"
            if answer := probe.answer:
                transcript += "A: " + answer.message + "\n"
            transcript += "\n"

        return transcript.strip()


class ImageHistory(BaseModel):
    primer: HistoryMessage | None = None
    description: HistoryMessage
