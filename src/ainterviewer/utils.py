import base64
import inspect
import logging
import time
from datetime import datetime
from functools import cache, wraps
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from requests.exceptions import RequestException

from ainterviewer.constants import LANGUAGES
from ainterviewer.lpm.types import Message
from ainterviewer.types import LanguageCode, LanguageDict


def get_language_dict(
    language_code: LanguageCode | None, name: str | None = None
) -> LanguageDict:
    if language_code is not None:
        return next(filter(lambda lang: lang["code"] == language_code, LANGUAGES))
    if name is not None:
        return next(filter(lambda lang: lang["name"] == name, LANGUAGES))

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


def now(zone="Europe/Copenhagen") -> datetime:
    return datetime.now(ZoneInfo(zone))


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


def get_function_signature_as_kwargs(
    func: Callable, local_vars: dict[str, Any]
) -> dict[str, Any]:
    params = inspect.signature(func).parameters.keys()

    params = {name: value for name, value in local_vars.items() if name in params}

    return params


def retry(max_retries: int = 3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RequestException as e:
                    if i < max_retries - 1:
                        sleep_time = (1 + i) ** 2
                        logging.warning(
                            f"Calling {func.__name__} failed. Attempt {i + 1}. Retrying in {sleep_time} seconds..."
                        )
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"All {max_retries} attempts failed.")
                        raise e

        return wrapper

    return decorator
