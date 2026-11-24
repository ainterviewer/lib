import base64
import re
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
            raise LanguageNotSupportedError(
                f"Language {language_code} not supported."
            ) from None
    if name is not None:
        lowered = name.casefold()
        try:
            return next(
                filter(lambda lang: lang["name"].casefold() == lowered, LANGUAGES)
            )
        except StopIteration:
            raise LanguageNotSupportedError(f"Language {name} not supported.") from None

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


# gpt-oss tends to open a Danish question with the English hesitation "Er"
# (a single token, shared with the Danish verb), then write the real question:
# "Er er der andre ...", "Er hvad var grunden ...". "Er" is only stripped when
# what follows could not continue a question that starts with the verb "er":
# punctuation, the verb repeated, a question word, or another finite verb.
_DA_LEADING_ER = re.compile(
    r"""
    ^(?P<lead>\s*)Er
    (?:
        \s*[,.…-]+\s*                    # "Er, ..." / "Er... "
      | \s+(?=(?i:er|hvad|hvordan|hvorfor|hvilke|hvilken|hvilket|hvem|hvornår
                 |kan|kunne|har|havde|vil|ville|skal|skulle|må|bør|synes|tror
                 |oplever|føler|gør|går|bruger|tænker|husker)\b)
    )
    """,
    re.VERBOSE,
)


def strip_leading_filler(text: str, language: LanguageCode) -> str:
    """Remove a hesitation "Er" an LLM put in front of a Danish question."""
    if language.upper() != "DA":
        return text
    match = _DA_LEADING_ER.match(text)
    if match is None:
        return text
    rest = text[match.end() :]
    if not rest:
        return text
    return match["lead"] + rest[0].upper() + rest[1:]
