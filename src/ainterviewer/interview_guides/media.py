from typing import Literal

from pydantic import BaseModel, Field

MediaType = Literal["audio", "image", "video"]


class MediaModel(BaseModel):
    """A file shown to the interviewee, by name.

    Only ever the name: the library never reads the file, and the
    application serving the interview resolves and serves it.
    """

    type: MediaType
    name: str = Field(description="The filename")


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
