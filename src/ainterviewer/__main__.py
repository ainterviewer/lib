from . import __version__

from typing import Annotated, Optional
from typer import Typer
import typer

cli = Typer()


def version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit()


@cli.callback()
def callback(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", help="Show the version and exit.", callback=version_callback
        ),
    ] = None,
) -> None:
    pass


if __name__ == "__main__":
    cli()
