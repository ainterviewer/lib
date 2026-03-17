import json

from ainterviewer.interview_guides.interview_guide import (
    InterviewGuide,
    InterviewGuideTemplate,
    Question,
    QuestionSection,
)
from ainterviewer.interview_guides.questions import QuestionBase
from ainterviewer.interview_guides.sections import QuestionSectionTemplate
from ainterviewer.lpm.clients import chat
from ainterviewer.lpm.types import Message, MessageRole

_DUMMY_PROMPT = "Create an interview guide targeted at participants of the IC2S2 computational social science conference. The interview guide should be structured to gather insights on the participants' experiences, motivations, and challenges in computational social science research."


async def generate_interview_guide(
    prompt: str,
    model: str,
    *,
    output_path: str | None,
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

    template = await chat(
        messages=messages,
        model=model,
        response_format=InterviewGuideTemplate,
    )

    interview_guide = InterviewGuide.model_validate(template.model_dump())

    if output_path:
        with open(output_path, "w") as f:
            json.dump(interview_guide.model_dump(), f, indent=4)

    return interview_guide


async def generate_section(
    instruction: str,
    model: str,
    guide: InterviewGuide,
) -> QuestionSection[Question]:
    """Generate a question section based on the QuestionSectionTemplate structure and an instruction."""
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
                f"# Instructions:\n{instruction}\n\n"
                f"# Interview Guide Context\n\n{guide}\n\n"
                f"# Response Schema\n\nRespond in the following format:\n```\n{QuestionSectionTemplate.model_json_schema()}\n```\n"
            ),
        ),
    ]

    template = await chat(
        messages=messages,
        model=model,
        response_format=QuestionSectionTemplate,
    )

    question_section = QuestionSection[Question].model_validate(template.model_dump())

    return question_section


async def generate_question(
    instructions: str,
    model: str,
    guide: InterviewGuide,
    section: QuestionSection[Question] | None = None,
) -> Question:
    """Generate a question based on the QuestionBase structure and an instruction."""
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content=(
                "Create a new question in a json format to be used by an AI interviewer for a social science qualitative interview, based on the information provided by the user. "
                "Make the question description elaborate so that the interviewer can understand the context and purpose of the question. "
            ),
        ),
        Message(
            role=MessageRole.USER,
            content=(
                f"# Instructions:\n{instructions}\n\n"
                f"# Interview Guide Context\n\n{guide}\n\n"
                + (
                    f"# Relevant Section\n\nThe new question will be added to the end of this specific section\n```\n{section}\n```\n\n"
                    if section
                    else ""
                )
                + f"# Response Schema\n\nRespond in the following format:\n```\n{QuestionBase.model_json_schema()}\n```\n"
            ),
        ),
    ]

    base = await chat(
        messages=messages,
        model=model,
        response_format=QuestionBase,
    )

    question = Question.model_validate(base.model_dump())

    return question


if __name__ == "__main__":
    raise NotImplementedError("CLI not implemented because of async")
