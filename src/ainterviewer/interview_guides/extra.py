from pydantic import BaseModel, EmailStr, model_validator


class Consent(BaseModel):
    title: str
    text: str


class Welcome(BaseModel):
    title: str
    text: str
    email: EmailStr
    video_file_name: str | None = None
    with_id: bool = True

    @model_validator(mode="after")
    def validate_model(self):
        # TODO:
        # We should probably implement optional email so the validation below
        # is actually correct
        if self.with_id and self.email is None:
            raise ValueError("email is required when with_id is True")
        return self
