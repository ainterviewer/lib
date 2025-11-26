import json

from openai import AsyncOpenAI
from typer import Typer

from ainterviewer.interview_guides.interview_guide import InterviewGuideContent
from ainterviewer.lpm.types import Message
from ainterviewer.settings import settings

cli = Typer()

client = AsyncOpenAI(api_key=settings.secrets.openai_api_key.get_secret_value())

_DUMMY_PROMPT = "Create an interview guide targeted at participants of the IC2S2 computational social science conference. The interview guide should be structured to gather insights on the participants' experiences, motivations, and challenges in computational social science research."


@cli.command()
async def generate_interview_guide(prompt: str, *, output_path: str | None):
    """Generate an interview guide based on the InterviewGuideContent structure and a given prompt."""
    messages = [
        Message(
            role="system",
            content=(
                "Create an interview guide in a json format based to be used by an AI interviewer for a social science qualitative interview, based on the information provided by the user. "
                "Make the section and question descriptions and framing elaborate so that the interviewer can understand the context and purpose of the interview and each section and question. "
                "Unless anything else is specified, the interview guide should contain about 2-3 sections, each with about 1-2 questions."
            ),
        ),
        Message(
            role="user",
            content=prompt,
        ),
    ]

    response = await client.responses.parse(
        model="gpt-5-mini",
        input=messages,
        text_format=InterviewGuideContent,
        reasoning={
            "effort": "minimal",
        },
    )

    interview_guide = response.output_parsed

    if interview_guide is None:
        raise ValueError(
            "Failed to parse the response into an InterviewGuideContent object."
        )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(interview_guide.model_dump(), f, indent=4)

    return interview_guide


if __name__ == "__main__":
    cli()
