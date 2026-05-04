# TODO:
# - How should we handle language/translations here?

from __future__ import annotations

import json
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from random import choice, randint, uniform
from typing import Any, List

from jinja2 import Template
from pydantic import BaseModel, Field, computed_field


class AnswerLength(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


ANSWER_LENGTH_MAP: dict[AnswerLength, str] = {
    AnswerLength.SHORT: "short (i.e. 1-2 sentences)",
    AnswerLength.MEDIUM: "medium (i.e. 2-4 sentences)",
    AnswerLength.LONG: "long (i.e. 3-5 sentences)",
}

INTERVIEWEE_INFORMATION_TEMPLATE = Template("""
You are a{{ 'n' if (age == 18) or (age | string | first | lower == '8') else '' }} {{ age }} old {{ gender }} named {{ name }}.
You are a{{ 'n' if occupation | first | lower in 'aeiou' else '' }} {{ occupation }} living in {{ location }}.
Your highest educational level is {{ education }}.

The length of your answers should be {{ communication_trait.length_description }}, the style should be {{ communication_trait.style }} and in a {{ communication_trait.tone }} tone.
{% if extra_traits %}
{{ extra_traits }}
{% endif %}
{% if additional_instructions %}
{{ additional_instructions }}
{% endif %}
""")


class InterviewSubject(BaseModel):
    """Dataclass representing the interview subject"""

    name: str
    age: int
    gender: str
    education: str
    occupation: str
    location: str
    personality: str
    communication_trait: CommunicationTrait
    extra_traits: dict[str, str] | str | None = None
    refusal_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    def __str__(self):
        return (
            f"Name: {self.name}\n"
            f"Age: {self.age}\n"
            f"Gender: {self.gender}\n"
            f"Education: {self.education}\n"
            f"Occupation: {self.occupation}\n"
            f"Location: {self.location}\n"
            f"Personality: {self.personality}\n"
            f"Communication Traits:\n"
            f"\t- Answer length: {self.communication_trait.length_description}\n"
            f"\t- Answer style: {self.communication_trait.style}\n"
            f"\t- Answer tone: {self.communication_trait.tone}\n"
            f"Extra traits: {self.extra_traits}\n"
            # f"Refusal Rate: {self.refusal_rate}\n" # NOTE: Excluded from description since it's handled in the logic
        )


class BackgroundInfoOptions(BaseModel):
    """Dataclass representing the possible values for the background info of the synthetic agents"""

    names_gender: list[tuple[str, str]]
    age_range: tuple[int, int] = (18, 80)
    educations: list[str]
    occupations: list[str]
    locations: list[str]
    personalities: list[str]
    communication_traits: CommunicationTraits
    extra_traits: list[dict[str, str]] | list[str] | None = None
    refusal_rate: tuple[float, float] | None = None


class CommunicationTrait(BaseModel):
    length: AnswerLength
    style: str
    tone: str

    @computed_field
    @property
    def length_description(self) -> str:
        return ANSWER_LENGTH_MAP[self.length]


class CommunicationTraits(BaseModel):
    length: List[AnswerLength | str]
    style: List[str]
    tone: List[str]

    def __iter__(self) -> Iterator[tuple[str, Any]]:  # ty:ignore[invalid-method-override]
        yield from self.__dict__.items()


def generate_synthetic_person(
    background_info: BackgroundInfoOptions,
) -> InterviewSubject:
    name, gender = choice(background_info.names_gender)
    education = choice(background_info.educations)
    occupation = choice(background_info.occupations)
    location = choice(background_info.locations)
    age = randint(*background_info.age_range)
    personality = choice(background_info.personalities)
    communication_traits: CommunicationTrait = {  # ty: ignore[invalid-assignment]
        key: choice(value) for key, value in background_info.communication_traits
    }
    extra_traits = (
        choice(background_info.extra_traits) if background_info.extra_traits else None
    )
    refusal_rate = (
        round(uniform(*background_info.refusal_rate), 2)
        if background_info.refusal_rate is not None
        else InterviewSubject.refusal_rate
    )

    return InterviewSubject(
        name=name,
        gender=gender,
        education=education,
        occupation=occupation,
        location=location,
        age=age,
        personality=personality,
        communication_trait=communication_traits,
        extra_traits=extra_traits,
        refusal_rate=refusal_rate,
    )


def generate_synthetic_persons(
    background_info: BackgroundInfoOptions, num_agents: int
) -> list[InterviewSubject]:
    return [
        generate_synthetic_person(background_info=background_info)
        for _ in range(num_agents)
    ]


DEFAULT_BACKGROUND_INFO_OPTIONS = BackgroundInfoOptions(
    age_range=(18, 80),
    names_gender=[
        ("Bob", "Male"),
        ("Richard", "Male"),
        ("Alice", "Female"),
        ("Emily", "Female"),
    ],
    educations=["High School", "College", "Graduate School"],
    occupations=[
        "Student",
        "Engineer",
        "Researcher",
        "Teacher",
        "Farmer",
        "Mechanic",
        "Baker",
        "Mailman",
        "Policeman",
    ],
    locations=["a large city", "a small town", "a rural area"],
    communication_traits=CommunicationTraits(
        length=["short", "medium", "long"],
        style=["concise", "vague", "elaborate", "detailed"],
        tone=["formal", "casual"],
    ),
    personalities=[
        "confident",
        "humble",
        "shy",
        "analytical",
        "creative",
        "emotional",
        "enthusiastic",
        "provocative",
    ],
    refusal_rate=(0, 0.7),
    extra_traits=[
        "You think that all people deserve to be treated with respect and dignity, regardless of their background or situation.",
        "You think that some people deserve respect and dignity, but it depends on their background or specific situation.",
        "You think that dignity and respect must be hard earned by long established relationships.",
    ],
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="This module help create synthetic subjects."
    )
    parser.add_argument(
        "--background-info",
        type=Path,
        default="data/background_info/en.json",
        help="Path to the folder or file containing background info. If a folder is specified the language will determine the file, eg. `en.json`",
    )

    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument(
        "--extra-traits", type=Path, default="data/background_info/traits.json"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.background_info) as f:
        _background_info_data = json.load(f)

    if args.extra_traits:
        with open(args.extra_traits) as f:
            _background_info_data["extra_traits"] = json.load(f)

    background_info = BackgroundInfoOptions(**_background_info_data)

    synthetic_persons = generate_synthetic_persons(background_info, args.num_agents)
    for person in synthetic_persons:
        print(person)
        person_dump = person.model_dump()
        print(person_dump)
        print(INTERVIEWEE_INFORMATION_TEMPLATE.render(**person_dump))
