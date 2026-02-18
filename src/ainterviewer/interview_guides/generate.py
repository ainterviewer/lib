import json

from openai import AsyncOpenAI

from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewGuideTemplate,
    Question,
    QuestionSection,
    QuestionSectionTemplate,
)
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.lpm.types import Message, MessageRole
from ainterviewer.settings import settings

# TODO:
# Should use the lpm.clients chat function instead
client = AsyncOpenAI(api_key=settings.secrets.openai_api_key.get_secret_value())

_DUMMY_PROMPT = "Create an interview guide targeted at participants of the IC2S2 computational social science conference. The interview guide should be structured to gather insights on the participants' experiences, motivations, and challenges in computational social science research."


async def generate_interview_guide(
    prompt: str, *, output_path: str | None
) -> InterviewGuide:
    """Generate an interview guide based on the InterviewGuideTemplate structure and a given prompt."""
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                "Create an interview guide in a json format to be used by an AI interviewer for a social science qualitative interview, based on the information provided by the user. "
                "Make the section and question descriptions and framing elaborate so that the interviewer can understand the context and purpose of the interview and each section and question. "
                "Unless anything else is specified, the interview guide should contain about 2-3 sections, each with about 1-3 questions."
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=prompt,
        ),
    ]

    response = await client.responses.parse(
        model="gpt-5-mini",
        input=messages,
        text_format=InterviewGuideTemplate,
        reasoning={
            "effort": "minimal",
        },
    )

    interview_guide = InterviewGuide.model_validate_json(response.output_text)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(interview_guide.model_dump(), f, indent=4)

    return interview_guide


async def generate_section(
    prompt: str,
    guide: InterviewGuide,
) -> QuestionSection[Question]:
    """Generate a question section based on the QuestionSectionTemplate structure and a given prompt."""
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                "Create a new question section in a json format to be used by an AI interviewer for a social science qualitative interview, based on the information provided by the user. "
                "Make the section and question descriptions elaborate so that the interviewer can understand the context and purpose of each section and question. "
                "Unless anything else is specified, the section should contains about 1-3 questions."
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                f"# Instructions:\n{prompt}\n\n"
                f"# Interview Guide Context\n\n{guide}\n\n"
                f"# Response Schema\n\nResponse in the following format:\n```\n{QuestionSectionTemplate.model_json_schema()}\n```\n"
            ),
        ),
    ]

    response = await client.responses.parse(
        model="gpt-5-mini",
        input=messages,
        text_format=QuestionSectionTemplate,
        reasoning={
            "effort": "minimal",
        },
    )

    question_section = QuestionSection.model_validate_json(response.output_text)

    return question_section


async def generate_question(
    prompt: str,
    guide: InterviewGuide,
    section_idx: int,
) -> Question:
    """Generate a question based on the QuestionBase structure and a given prompt."""
    messages = [
        Message(
            role="system",
            content=(
                "Create a new question in a json format to be used by an AI interviewer for a social science qualitative interview, based on the information provided by the user. "
                "Make the question description elaborate so that the interviewer can understand the context and purpose of the question. "
            ),
        ),
        Message(
            role="user",
            content=(
                f"# Instructions:\n{prompt}\n\n"
                f"# Interview Guide Context\n\n{guide}\n\n"
                f"# Relevant Section\n\nThe new question will be added to the end of this specific section\n```\n{guide.question_sections[section_idx]}\n```\n\n"
                f"# Response Schema\n\nResponse in the following format:\n```\n{QuestionBase.model_json_schema()}\n```\n"
            ),
        ),
    ]

    response = await client.responses.parse(
        model="gpt-5-mini",
        input=messages,
        text_format=QuestionBase,
        reasoning={
            "effort": "minimal",
        },
    )

    question_section = Question.model_validate_json(response.output_text)

    return question_section


if __name__ == "__main__":
    raise NotImplementedError("CLI not implemented because of async")
