"""Request and response models for Gnani speech endpoints."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TranscriptionResponse(BaseModel):
    transcript: str
    model: str
    request_id: str | None = None
    detected_language_code: str | None = None
    status: Literal["transcribed", "no_speech"]
    message: str | None = None


class SpeechSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)
    response_language: Literal["en", "hi", "gu"]

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Please provide text for Saar to speak.")
        return cleaned
