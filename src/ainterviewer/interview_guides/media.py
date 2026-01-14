from pathlib import Path
from typing import Literal, Optional, Self

from pydantic import BaseModel, Field

from ainterviewer.utils import encode_image

MediaType = Literal["audio", "image", "video"]


class MediaModel(BaseModel):
    type: MediaType
    filename: str = Field(description="The filename")
    data: str | bytes = Field(repr=False)

    @classmethod
    def read(cls, filepath: Path) -> Self:
        """Reads the data file from the full path and encodes it to Base64, saving it to the data attribute"""
        media = cls.model_construct(filename=filepath.name)
        media.data = encode_image(filepath)

        return media


class Audio(MediaModel):
    type: MediaType = "audio"


class Video(MediaModel):
    type: MediaType = "video"


class Image(MediaModel):
    """An image to show the interviewee"""

    type: MediaType = "image"

    primer: Optional[str] = Field(
        None, description="A primer to show the interviewee before showing the image"
    )
    description: str = Field(
        description="A description of the image used to guide the probing"
    )
    alt: str = Field(description="The alt text for the image for accessibility")
