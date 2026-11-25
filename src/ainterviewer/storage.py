"""Where media storage's roots are.

Each root holds a directory per project, interview or experiment, named by
its id. The library defines the roots and only what it reads itself: guide
images, which the interview loop sends to the respondent
(`ProjectStorage.image_path`). Interview recordings and images it only
passes along by name, from `IOProtocol` to `PersistenceProtocol`; every
directory like those, which only an application using the library makes and
serves, is that application's to define under these roots.
"""

from pathlib import Path

from pydantic import UUID4, BaseModel, Field, field_validator


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

    def image_path(self, project_id: UUID4) -> Path:
        """The project's guide images. Created if missing."""
        path = self.base_path / str(project_id) / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path


class InterviewStorage(BaseStorage):
    """The root of interview media."""

    base_path: Path = Field(default=Path("storage/interviews/"))


class ExperimentStorage(BaseStorage):
    """The root of experiment media."""
