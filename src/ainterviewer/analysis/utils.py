from typing import Literal

import polars as pl
from rich.console import Console
from user_agents import parse

from ainterviewer.lpm.types import CustomTokens


def get_device(user_agent: str) -> Literal["mobile", "tablet", "pc", "bot"] | None:
    ua = parse(user_agent)
    if ua.is_mobile:
        return "mobile"
    elif ua.is_tablet:
        return "tablet"
    elif ua.is_pc:
        return "pc"
    elif ua.is_bot:
        return "bot"

    return


def calculate_response_times(
    messages: pl.DataFrame,
    who: Literal["interviewer"] | Literal["user"] = "interviewer",
) -> list[pl.Datetime]:
    response_times: list[pl.Datetime] = []
    for row in messages.filter(role="USER").iter_rows(named=True):
        if not (
            next_message := messages.filter(
                message_id=row["message_id"] + (-1 if who == "user" else 1)
            )
        ).is_empty():
            assert next_message["role"][0] == "ASSISTANT"
            response_time = (
                next_message["created_at"] - row["created_at"]
                if who == "interviewer"
                else row["created_at"] - next_message["created_at"]
            )
            response_times.append(response_time)

    return response_times


def print_interview(
    interview_id: str,
    messages: pl.DataFrame,
    timestamp_format="%H:%M:%S",
    interviewer: Literal["human", "ai"] = "human",
):
    console = Console(highlight=False)

    interview_transcript = (
        f"Interview id: [deep_sky_blue1]{interview_id}[/deep_sky_blue1]\n"
        f"Interviewer: [deep_sky_blue1]{interviewer}[/deep_sky_blue1]\n\n"
    )

    for row in messages.filter(interview_id=interview_id).iter_rows(named=True):
        role = row["role"]
        role_color = "turquoise4" if role == "ASSISTANT" else "orange_red1"
        content = row["content"]
        if content.strip() in CustomTokens.all:
            interview_transcript += f"\n[purple]{content.strip()}[/purple]\n\n"
        else:
            timestamp = row["created_at"].strftime(timestamp_format)

            prefix = f"[purple]{timestamp}[/purple]"
            if row["section"] is not None:
                interview_position = f"{int(row['section'])} - {int(row['main_question'])} - {int(row['sub_question'])}"
                prefix += f" [yellow]{interview_position}[/yellow]"
            prefix += f" [{role_color}]{role}[/{role_color}]"
            interview_transcript += f"{prefix}:\n{content.strip()}\n\n"

    console.print(interview_transcript, highlight=False)
