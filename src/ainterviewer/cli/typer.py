"""
https://github.com/fastapi/typer/issues/309
"""

import inspect

import typer
from typer.main import get_command_name


class TyperMixin:
    """Mixin class to add Typer command functionality to any class."""

    def __init__(self, **kwargs):
        """
        Register all methods starting with _cmd_ as Typer commands.
        """

        if not (name := kwargs.get("name")):
            name = self.__class__.__name__

        self.app = typer.Typer(name=name)

        for method_name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not method_name.startswith("_cmd_"):
                continue

            method_name_stripped = method_name.removeprefix("_cmd_")
            command_name = get_command_name(method_name_stripped)
            # Register the method as a Typer command
            self.app.command(name=command_name)(method)

            # Register the method stripped from prefix to the instance
            setattr(self, method_name_stripped, method)


if __name__ == "__main__":
    # Example usage
    class MyClass(TyperMixin):
        def __init__(self, value: int = 10, **kwargs):
            super().__init__(**kwargs)
            self.value = value

        def _cmd_hello(self, name: str):
            """Say hello to someone."""
            print(f"Hello, {name}!")

        def _cmd_double_value(self):
            print(self.value * 2)

    my_class = MyClass(value=20)
    my_class.hello("Bob")  # No type hints :(
    # Run it as standalone app:
    my_class.app()

    # Or add it to another Typer app:
    app = typer.Typer()
    app.add_typer(my_class.app)
    app()
