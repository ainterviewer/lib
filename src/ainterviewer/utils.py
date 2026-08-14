import base64
from datetime import datetime
from functools import cache
from pathlib import Path

from ainterviewer.constants import LANGUAGES
from ainterviewer.exceptions import LanguageNotSupportedError
from ainterviewer.lpm.types import Message
from ainterviewer.settings import settings
from ainterviewer.types import LanguageCode, LanguageDict


def get_language_dict(
    language_code: LanguageCode | None, name: str | None = None
) -> LanguageDict:
    if language_code is not None:
        code = language_code.upper()
        try:
            return next(filter(lambda lang: lang["code"] == code, LANGUAGES))
        except StopIteration:
            raise LanguageNotSupportedError(f"Language {language_code} not supported.")
    if name is not None:
        try:
            return next(filter(lambda lang: lang["name"] == name, LANGUAGES))
        except StopIteration:
            raise LanguageNotSupportedError(f"Language {name} not supported.")

    raise ValueError("Either language_code or name must be provided.")


def dict2xml(
    d: dict[str, str | None], root_node: str | None = None, with_indent: bool = False
) -> str:
    """
    Convert a dictionary to an XML string.
    """
    xml = "\n".join(
        f"<{tag}>\n\t{value}\n</{tag}>"
        for key, value in d.items()
        if (tag := key.replace("_", "-")) and value is not None
    )

    if root_node:
        xml = f"<{root_node}>\n{xml.replace('\n', '\n\t')}\n</{root_node}>"
    if with_indent:
        xml = "\t" + xml.replace("\n", "\n\t")

    return xml


def now() -> datetime:
    return datetime.now(settings.tzinfo)


@cache
def encode_image(image_path: Path | str) -> str:
    """Reads a file and encodes it to Base64."""
    with open(image_path, "rb") as f:
        image = f.read()
        # Encode the binary data to Base64
        base64_encoded_data = base64.b64encode(image)

        # Convert the Base64 bytes to a string
        base64_encoded_string = base64_encoded_data.decode("utf-8")

    return "data:image/jpeg;base64," + base64_encoded_string


def create_transcript(messages: list[Message], interviewee: bool = False) -> str:
    if interviewee is False:
        qa_mapper = {"assistant": "Q: ", "user": "A: "}
    elif interviewee is True:
        qa_mapper = {"assistant": "A: ", "user": "Q: "}
    else:
        raise ValueError("interviewee must be either True or False")

    transcript = "\n\n".join(
        [
            f"{role}{message['content']}"
            for message in messages
            if (role := qa_mapper.get(message["role"]))
        ]
    )
    return transcript
