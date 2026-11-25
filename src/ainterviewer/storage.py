"""Where media storage's roots are.

Each root holds a directory per project, interview or experiment, named by
its id. The library defines only the roots: it reads and writes no media
itself. Guide images and interview recordings pass through it by name, and
every directory under these roots is the application's to define, make and
serve.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class BaseStorage(BaseModel):
    """A storage root. Made absolute, and created if missing."""

    base_path: Path

    @field_validator("base_path")
    @classmethod
    def validate_base_path(cls, v: Path) -> Path:
        """Ensure base_path is absolute and create if it doesn't exist."""
        path = v.resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


class ProjectStorage(BaseStorage):
    """The root of project media."""

    base_path: Path = Field(default=Path("storage/projects/"))


class InterviewStorage(BaseStorage):
    """The root of interview media."""

    base_path: Path = Field(default=Path("storage/interviews/"))


class ExperimentStorage(BaseStorage):
    """The root of experiment media."""
