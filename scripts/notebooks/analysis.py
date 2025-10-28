import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")

with app.setup:
    import json
    from typing import Iterator

    import marimo as mo
    import plotly.express as px
    import polars as pl

    from ainterviewer.analysis.data import read_from_database
    from ainterviewer.analysis.utils import print_interview


@app.cell
def _():
    mo.md("""# Analysis""")
    return


@app.cell
def data_imports():
    messagesmessagesprojects: pl.DataFrame
    interviews: pl.DataFrame
    messages: pl.DataFrame

    projects, interviews, messages = read_from_database()
    epinion: pl.DataFrame = pl.read_csv(
        "data/analysis/P2100211_pilot_anonymized(in).csv", encoding="utf-8"
    )
    user_ids: list[str] = epinion["Id"].to_list()

    messages
    projects
    return epinion, interviews, messages, projects, user_ids


@app.cell
def _(epinion: pl.DataFrame):
    epinion_q = (
        epinion[
            ["Id", "Q1a", "Q1b", "Q2a", "Q2b", "Q3a", "Q3b", "Q4a", "Q4b", "Q5a", "Q5b"]
        ]
        .filter(pl.all_horizontal(pl.col("*") != ""))
        .unpivot(
            index="Id",
            variable_name="question",
            value_name="content",
        )
        .with_columns(
            [
                pl.col("question")
                .str.extract(r"Q(\d+)")
                .cast(pl.Int64)
                .sub(1)
                .alias("main_question"),
                pl.col("question")
                .str.extract(r"Q\d+([ab])")
                .map_elements(lambda x: {"a": 0, "b": 1}.get(x), return_dtype=pl.Int8)
                .alias("sub_question"),
            ]
        )
        .drop("question")
        .sort(["Id"])
    )
    return (epinion_q,)


@app.cell
def _(epinion: pl.DataFrame):
    gender_bar = px.bar(epinion["gender"].value_counts(), x="gender", y="count")
    age_bar = px.bar(epinion["age"].value_counts().sort(by="age"), x="age", y="count")
    edu_bar = px.bar(
        epinion["uddannelse"].value_counts().sort(by="uddannelse"),
        x="uddannelse",
        y="count",
    )
    pol_bar = px.bar(
        epinion["Q_selvplacering_o1"]
        .value_counts()
        .filter(pl.col("Q_selvplacering_o1") != "NA")
        .cast({"Q_selvplacering_o1": pl.Int8})
        .sort(by="Q_selvplacering_o1"),
        x="Q_selvplacering_o1",
        y="count",
    )

    mo.vstack(
        [
            mo.hstack([gender_bar, age_bar]),
            mo.hstack([edu_bar, pol_bar]),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell
def data_filtering(
    interviews: pl.DataFrame,
    messages: pl.DataFrame,
    projects,
    user_ids: list[str],
):
    filtered_projects: pl.DataFrame = projects.filter(
        pl.col("title").str.starts_with("Deservingness")
    )
    filtered_interviews: pl.DataFrame = interviews.with_columns(
        userid=pl.col("external_params").map_elements(
            lambda x: json.loads(x)["user1"], return_dtype=pl.String
        )
    ).filter(
        pl.col("project_id").is_in(filtered_projects["id"].to_list()),
        pl.col("userid").is_in(user_ids),
    )

    filtered_messages: pl.DataFrame = messages.filter(
        pl.col("interview_id").is_in(filtered_interviews["id"].to_list())
    )

    project_id_to_title: dict[str, str] = dict(
        filtered_projects[["id", "title"]].iter_rows()
    )
    return filtered_interviews, filtered_messages, project_id_to_title


@app.cell
def subset_data(
    filtered_interviews: pl.DataFrame,
    filtered_messages: pl.DataFrame,
):
    completed_interviews: list[str] = filtered_interviews.filter(
        pl.col("is_complete") == 1,
    )["id"].to_list()

    all_replies: pl.DataFrame = filtered_messages.filter(
        pl.col("interview_id").is_in(completed_interviews),
    )
    user_replies: pl.DataFrame = all_replies.filter(
        pl.col("role") == "USER",
    )
    assistant_replies: pl.DataFrame = all_replies.filter(
        pl.col("sub_question") > 0,
        pl.col("role") == "ASSISTANT",
    )

    interview_id_iter: Iterator[str] = iter(completed_interviews)
    return (
        assistant_replies,
        completed_interviews,
        interview_id_iter,
        user_replies,
    )


@app.cell
def _():
    return


@app.cell
def _(filtered_interviews: pl.DataFrame):
    filtered_interviews.with_columns(
        userid=pl.col("external_params").map_elements(
            lambda x: json.loads(x)["user1"], return_dtype=str
        )
    )[["project_id", "id", "userid", "device"]].rename(
        mapping={"id": "interview_id"}
    ).write_csv("id_mappings.csv")
    return


@app.cell
def _():
    mo.md("""## Plots""")
    return


@app.cell
def overview_plot(
    filtered_interviews: pl.DataFrame,
    project_id_to_title: dict[str, str],
):
    interviews_per_project: pl.DataFrame = (
        filtered_interviews["project_id"]
        .value_counts()
        .with_columns(
            project_title=pl.col("project_id").map_elements(project_id_to_title.get)
        )
        .sort(by="project_title")
    )

    px.bar(interviews_per_project, x="project_title", y="count")
    return


@app.cell
def _(filtered_interviews: pl.DataFrame):
    px.bar(filtered_interviews["device"].value_counts(), x="device", y="count")
    return


@app.cell
def message_len_plots(
    epinion_q,
    filtered_messages: pl.DataFrame,
    project_id_to_title: dict[str, str],
):
    mean_answer_length = (
        filtered_messages.filter(pl.col("role") == "USER")
        .group_by("project_id", "interview_id")
        .agg(pl.col("content").str.len_chars().mean())
        .with_columns(
            project_title=pl.col("project_id").map_elements(project_id_to_title.get),
            role=pl.lit("USER"),
        )
        .sort(by="project_title", descending=True)
    )

    mean_answer_length_epinion = (
        epinion_q.group_by("Id")
        .agg(pl.col("content").str.len_chars().mean())
        .with_columns(role=pl.lit("USER"))
        .with_columns(project_title=pl.lit("Survey"))
        .drop("Id")
    )

    mean_answer_length = pl.concat(
        [mean_answer_length, mean_answer_length_epinion], how="align_full"
    )

    mean_question_length = (
        filtered_messages.filter(pl.col("role") == "ASSISTANT")
        .group_by("project_id", "interview_id")
        .agg(pl.col("content").str.len_chars().mean())
        .with_columns(
            project_title=pl.col("project_id").map_elements(project_id_to_title.get),
            role=pl.lit("ASSISTANT"),
        )
        .sort(by="project_title", descending=True)
    )

    combined = (
        pl.concat([mean_answer_length, mean_question_length])
        .sort("project_title", descending=True)
        .rename({"content": "chars"})
    )

    px.box(
        combined,
        x="chars",
        y="role",
        title="Mean message length",
        color="project_title",
        category_orders={
            "project_title": list(reversed(project_id_to_title.values())) + ["Survey"]
        },
    )
    return


@app.cell
def reply_time_plot(assistant_replies: pl.DataFrame):
    px.histogram(assistant_replies, "time_since_previous")
    return


@app.cell
def _(user_replies: pl.DataFrame):
    px.bar(
        user_replies.sort(by="created_at", descending=True)["content"].str.len_chars()
    )
    return


@app.cell
def _(assistant_replies: pl.DataFrame):
    px.bar(
        assistant_replies.sort(by="created_at", descending=True)[
            "content"
        ].str.len_chars()
    )
    return


@app.cell
def _(
    completed_interviews: list[str],
    interview_id_iter: Iterator[str],
    messages: pl.DataFrame,
):
    try:
        interview_id = next(interview_id_iter)

        print_interview(
            interview_id=completed_interviews[-1],
            messages=messages,
            timestamp_format="%d-%m %H:%M:%S",
            interviewer="ai",
        )
    except StopIteration:
        print("No more conversations to display")
    return


if __name__ == "__main__":
    app.run()
