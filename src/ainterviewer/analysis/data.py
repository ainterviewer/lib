import sqlite3

import polars as pl

from ainterviewer.analysis.utils import get_device


def read_from_database(
    db_path: str = "app/db.sqlite",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    conn = sqlite3.connect(db_path)

    projects = pl.read_database(
        query="SELECT * FROM project",
        connection=conn,
    )

    interviews = pl.read_database(query="SELECT * FROM interview", connection=conn)
    interviews = interviews.with_columns(
        created_at=pl.col("created_at").str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%.6f",
            time_unit="ms",
            strict=False,
        ),
        device=pl.col("user_agent").map_elements(get_device),
    )

    messages = (
        pl.read_database(
            query="SELECT * FROM message",
            connection=conn,
            schema_overrides={"feedback": pl.String},
        )
        .with_columns(
            created_at=pl.col("created_at").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S%.6f", time_unit="ms"
            )
        )
        .sort(["interview_id", "message_id"])
        .with_columns(
            time_since_previous=pl.col("created_at").diff().dt.total_seconds(),
            is_response=pl.col("role") != pl.col("role").shift(1),
        )
    )

    return projects, interviews, messages
