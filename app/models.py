"""Request and response shapes for the HTTP API."""

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

Category = Literal["billing", "technical", "account", "other"]
Priority = Literal["low", "medium", "high"]
Status = Literal["pending", "classified", "failed"]

SECOND_SENTENCE = re.compile(r"[.!?][\"')\]]*\s+\S")


class TicketIn(BaseModel):
    # URL-safe, so every ticket can be fetched at /tickets/{id}.
    id: Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]+$", max_length=128)]
    subject: Annotated[str, Field(max_length=500)] = ""
    body: Annotated[str, Field(max_length=20_000)]

    @field_validator("body")
    @classmethod
    def body_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class TicketOut(BaseModel):
    id: str
    subject: str
    body: str
    status: Status
    category: Category | None
    priority: Priority | None
    summary: str | None
    failure_reason: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime


class TicketPage(BaseModel):
    items: list[TicketOut]
    total: int
    limit: int
    offset: int


class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Category
    priority: Priority
    summary: Annotated[StrictStr, Field(max_length=300)]

    @field_validator("summary")
    @classmethod
    def summary_is_one_sentence(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        if "\n" in value or "\r" in value or SECOND_SENTENCE.search(value):
            raise ValueError("must be one sentence")
        return value
