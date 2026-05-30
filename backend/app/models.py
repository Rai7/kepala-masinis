from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Role(str, Enum):
    public = "public"
    internal = "internal"


Intent = Literal[
    "station_query",
    "train_query",
    "city_to_city_query",
    "unknown",
    "search_train_schedule",
    "search_train_by_name",
    "search_route_stops",
    "station_info",
    "railway_disruption_news",
    "kai_policy_question",
    "travel_context",
    "fallback_web_question",
    "general_chat",
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    role: Role = Role.public
    use_llm: bool | None = None
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    ui_metadata: dict[str, Any] | None = None


class Clarification(BaseModel):
    question: str
    options: list[str] | None = None


class ChatResponse(BaseModel):
    intent: Intent
    reply: str
    data: dict[str, Any] | None = None
    clarification: Clarification | None = None
    metadata: dict[str, Any] | None = None
    session_id: str
    request_id: str
    cache_hit: bool = False
    user_message_id: str | None = None
    assistant_message_id: str | None = None


class StationStopRow(BaseModel):
    train_no: str
    train_name: str | None = None
    arrival_time: str | None = None
    departure_time: str | None = None


class StationStopsResponse(BaseModel):
    station_code: str
    total: int
    limited: bool
    results: list[StationStopRow]


class TrainStopRow(BaseModel):
    station_order: int
    station_name: str
    station_code: str
    arrival_time: str | None = None
    departure_time: str | None = None


class TrainStopsResponse(BaseModel):
    train_no: str
    train_name: str | None = None
    route: str | None = None
    total: int
    limited: bool
    results: list[TrainStopRow]


class SearchStationRow(BaseModel):
    station_code: str
    station_name: str


class SearchStationsResponse(BaseModel):
    query: str
    total: int
    results: list[SearchStationRow]


class SearchTrainRow(BaseModel):
    train_no: str
    train_name: str | None = None
    route: str | None = None


class SearchTrainsResponse(BaseModel):
    query: str
    total: int
    results: list[SearchTrainRow]


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    rating: str = Field(min_length=1, max_length=32)
    score: int | None = Field(default=None, ge=1, le=5)
    correction: str | None = Field(default=None, max_length=4000)
    reviewer_note: str | None = Field(default=None, max_length=4000)
    is_golden_example: bool = False
    corrected_intent: str | None = Field(default=None, max_length=128)


class FeedbackResponse(BaseModel):
    feedback_id: str
    stored: bool
