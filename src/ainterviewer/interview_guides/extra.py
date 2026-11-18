from typing import Annotated

from pydantic import BaseModel, BeforeValidator, EmailStr, model_validator

#: A contact email, or none: an emptied field (`""`) reads as none.
OptionalEmail = Annotated[EmailStr | None, BeforeValidator(lambda value: value or None)]

#: Why a welcome screen without a contact email is invalid while it shows
#: the box that tells respondents how to withdraw.
EMAIL_REQUIRED = "A contact email is required while the contact box is shown."


class Consent(BaseModel):
    title: str
    text: str


class Welcome(BaseModel):
    title: str
    text: str
    #: Required while `with_id`: the box it shows tells respondents to write
    #: to it, with their interview's ID, to withdraw.
    email: OptionalEmail = None
    video_file_name: str | None = None
    #: Show the box with the contact email and the interview's ID.
    with_id: bool = True

    @model_validator(mode="after")
    def _email_when_shown(self):
        if self.with_id and self.email is None:
            raise ValueError(EMAIL_REQUIRED)
        return self
