"""Shared request and response models for document question answering."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class DocumentPageText(BaseModel):
    page_number: int = Field(ge=1)
    text: str = Field(max_length=500_000)


class DocumentQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    pages: list[DocumentPageText] = Field(default_factory=list, max_length=1_000)
    response_language: Literal["en", "hi", "gu"] = "en"

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Please enter a question about the notice.")
        return cleaned


class SourcePassage(BaseModel):
    page_number: int
    text: str


class DocumentAnswerResponse(BaseModel):
    answer: str
    sources: list[SourcePassage]
    status: Literal["answered", "not_found", "empty_document"]
    mode: Literal["evon", "extractive"]
    message: str | None = None
