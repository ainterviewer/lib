from pathlib import Path, PureWindowsPath
from typing import Literal, Self

from pydantic import UUID4, BaseModel, Field

from ainterviewer.settings import settings
from ainterviewer.utils import encode_image

MediaType = Literal["audio", "image", "video"]


class MediaModel(BaseModel):
    type: MediaType
    name: str = Field(description="The filename")
    data: str | None = Field(repr=False)

    def encode(self, project_id: UUID4):
        """Read the file `name` from the project's image directory into
        `data`. Only its last path component counts: a name is written by
        whoever edits the guide, and "../" in it would otherwise read any
        file the process can, to send it to the respondent. Windows paths
        split on both "/" and "\\", so they drop either kind of path."""
        name = PureWindowsPath(self.name).name
        if name in {"", ".", ".."}:
            raise FileNotFoundError(f"Not an image file name: {self.name!r}")
        self.data = encode_image(
            settings.storage.project_storage.image_path(project_id) / name
        )

    @classmethod
    def read(cls, filepath: Path, **kwargs) -> Self:
        """Reads the data file from the full path and encodes it to Base64, saving it to the data attribute"""
        media = cls.model_construct(filename=filepath.name, **kwargs)
        media.data = encode_image(filepath)

        return media


class Audio(MediaModel):
    type: MediaType = "audio"


class Video(MediaModel):
    type: MediaType = "video"


class Image(MediaModel):
    """An image to show the interviewee"""

    type: MediaType = "image"

    primer: str | None = Field(
        None, description="A primer to show the interviewee before showing the image"
    )
    description: str = Field(
        description="A description of the image used to guide the probing"
    )
    alt: str = Field(description="The alt text for the image for accessibility")
