from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ainterviewer.constants import FP_ASSETS_DIR
from ainterviewer.utils import encode_image


class Image(BaseModel):
    """An image to show the interviewee"""

    primer: Optional[str] = Field(
        None, description="A primer to show the interviewee before showing the image"
    )
    description: str = Field(
        description="A description of the image used to guide the probing"
    )
    alt: str = Field(description="The alt text for the image for accessibility")
    name: str = Field(description="The filename")
    data: str | bytes | None = Field(None, repr=False)

    @property
    def path(self) -> Path:
        return FP_ASSETS_DIR / "images" / self.name

    def encode(self) -> None:
        """Reads the image file and encodes it to Base64, saving it to the data attribute"""
        self.data = encode_image(FP_ASSETS_DIR / "images" / self.name)
