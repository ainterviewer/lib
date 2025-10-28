import polars as pl

from .utils import get_device


def read_from_database(
    db_uri: str = "sqlite://app/db.sqlite",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    projects = pl.read_database_uri(
        uri=db_uri, query="SELECT * FROM project", engine="adbc"
    )

    interviews = pl.read_database_uri(
        uri=db_uri,
        query="SELECT * FROM interview",
        engine="adbc",
    ).with_columns(
        created_at=pl.col("created_at").str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%.6f", time_unit="ms"
        ),
        device=pl.col("user_agent").map_elements(get_device),
    )

    messages = (
        pl.read_database_uri(uri=db_uri, query="SELECT * FROM message", engine="adbc")
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
