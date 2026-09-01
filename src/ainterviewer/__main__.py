from typing import Annotated

import typer
from typer import Typer

from . import __version__

cli = Typer()


def version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@cli.callback()
def callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", help="Show the version and exit.", callback=version_callback
        ),
    ] = None,
) -> None:
    pass


if __name__ == "__main__":
    cli()
