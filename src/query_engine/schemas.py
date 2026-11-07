from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class RouteDecision(BaseModel):
    intent: Literal[
        "SQL_QUERY",
        "VECTOR_SEARCH",
        "HYBRID_SEARCH",
        "GENERAL_QUESTION",
    ]

    rationale: str = ""
    search_query: str | None = None

    model_config = ConfigDict(extra="ignore")


class AssistantRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000,
    )


class AssistantResponse(BaseModel):
    answer: str
    route: str
    sources: list[dict] = Field(default_factory=list)
    sql: str | None = None