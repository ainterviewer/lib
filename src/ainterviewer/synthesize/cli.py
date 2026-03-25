import asyncio
from pathlib import Path
from uuid import uuid4

from typer import Typer

from ainterviewer.synthesize.interviewees import DEFAULT_BACKGROUND_INFO_OPTIONS
from ainterviewer.synthesize.interviews import synthesize

cli = Typer()


@cli.command()
def main(
    interview_config_path: Path,
    interview_guide_path: Path,
    agent_configs_path: Path,
    language: str = "EN",
    delay_before_answer: tuple[float, float] | None = None,
):
    asyncio.run(
        synthesize(
            uuid4(),
            DEFAULT_BACKGROUND_INFO_OPTIONS,
            language=language,
            interview_config_path=interview_config_path,
            interview_guide_path=interview_guide_path,
            agent_configs_path=agent_configs_path,
            n_interviews=1,
            answering_model="gemma3-27b",
            delay_before_answer=delay_before_answer,
        )
    )


if __name__ == "__main__":
    cli()
