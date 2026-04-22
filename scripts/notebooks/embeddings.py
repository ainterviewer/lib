import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")

with app.setup:
    import textwrap
    import marimo as mo

    import plotly.express as px
    import polars as pl
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    from ainterviewer.analysis.data import read_from_database

    EMBEDDING_MODEL = SentenceTransformer("google/embeddinggemma-300m")


@app.cell
def _():
    interviews: pl.DataFrame
    messages: pl.DataFrame
    projects: pl.DataFrame

    projects, interviews, messages = read_from_database()
    messages.filter(
        pl.col("project_id").is_in(
            projects.filter(pl.col("title") == "Deservingness A")["id"].to_list()
        ),
    )
    return messages, projects


@app.cell
def _(messages: pl.DataFrame, projects: pl.DataFrame):
    @mo.cache
    def get_embeddings(documents: list[str] | str):
        return EMBEDDING_MODEL.encode(documents, show_progress_bar=True)

    project_ids = projects.filter(pl.col("title").str.starts_with("Deservingness"))[
        "id"
    ].to_list()

    des_a = pl.col("project_id").is_in(
        projects.filter(pl.col("title") == "Deservingness A")["id"].to_list()
    )

    all_messages = messages.filter(
        pl.col("project_id").is_in(project_ids),
        pl.col("main_question") >= 0,
    ).with_columns(
        main_question=pl.when(des_a)
        .then(pl.col("section"))
        .otherwise(pl.col("main_question")),
        sub_question=pl.when(
            des_a,
            pl.col("main_question").mod(2).cast(bool),
        )
        .then(pl.lit(1))
        .otherwise(pl.col("sub_question")),
        section=pl.lit(0),
    )

    all_messages = all_messages.with_columns(
        embeddings=get_embeddings(all_messages["content"])
    )

    n_main_questions: list[int] = all_messages["main_question"].unique().to_list()

    return all_messages, n_main_questions


@app.cell
def _(all_messages):
    main_questions: pl.DataFrame = all_messages.filter(
        pl.col("sub_question") == 0,
        pl.col("role") == "ASSISTANT",
    ).unique(subset="content")

    probes: pl.DataFrame = all_messages.filter(
        pl.col("sub_question") > 0,
        pl.col("role") == "ASSISTANT",
    )
    answers: pl.DataFrame = all_messages.filter(
        pl.col("role") == "USER",
    )
    return answers, main_questions, probes


@app.cell
def _(
    answers: pl.DataFrame,
    main_questions: pl.DataFrame,
    n_main_questions: list[int],
    probes: pl.DataFrame,
):
    all_embeddings = (
        main_questions["embeddings"].to_list()
        + probes["embeddings"].to_list()
        + answers["embeddings"].to_list()
    )
    umap_embeddings = UMAP().fit_transform(all_embeddings)

    df: pl.DataFrame = pl.DataFrame(
        {
            "x": umap_embeddings[:, 0],
            "y": umap_embeddings[:, 1],
            "content": [
                "<br />".join(textwrap.wrap(t))
                for t in main_questions["content"].to_list()
                + probes["content"].to_list()
                + answers["content"].to_list()
            ],
            "type": ["main_question" for _ in range(main_questions.height)]
            + ["probe" for _ in range(probes.height)]
            + ["answer" for _ in range(answers.height)],
            "main_question": main_questions["main_question"].to_list()
            + probes["main_question"].to_list()
            + answers["main_question"].to_list(),
        }
    )

    df
    return (df,)


@app.cell
def _(df):
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="type",
        hover_data=["content"],
        facet_col="main_question",
        facet_col_wrap=2,
    )
    fig.update_traces(selector={"name": "probe"}, opacity=0.2)
    fig.update_traces(selector={"name": "answer"}, opacity=0.2)
    return


@app.cell
def _(answers: pl.DataFrame):
    @mo.cache
    def get_answers_embeddings():
        answers_embeddings = EMBEDDING_MODEL.encode(answers["content"])

        return UMAP().fit_transform(answers_embeddings)

    umap_answers_embeddings = get_answers_embeddings()

    answers_df = pl.DataFrame(
        {
            "x": umap_answers_embeddings[:, 0],
            "y": umap_answers_embeddings[:, 1],
            "content": answers["content"],
            "main_question": answers["main_question"],
        }
    ).with_columns(pl.col("main_question").cast(str))

    px.scatter(
        answers_df,
        x="x",
        y="y",
        color="main_question",
        hover_data=["content"],
    )
    return


@app.cell
def _(main_questions, probes):
    answers_df = pl.DataFrame(
        {
            "x": umap_questions_embeddings[:, 0],
            "y": umap_questions_embeddings[:, 1],
            "content": main_questions["content"] + probes["content"],
            "main_question": main_questions["main_question"] + probes["content"],
        }
    ).with_columns(pl.col("main_question").cast(str))

    px.scatter(
        answers_df,
        x="x",
        y="y",
        color="main_question",
        hover_data=["content"],
    )
    return


if __name__ == "__main__":
    app.run()
