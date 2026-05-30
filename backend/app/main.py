from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .chat_orchestrator import process_chat_request
from .config import settings
from .feedback_service import save_feedback
from .llm_client import llm_provider
from .models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    Role,
    SearchStationsResponse,
    SearchTrainsResponse,
    StationStopsResponse,
    TrainStopsResponse,
)
from .repository import fetch_station_stops, fetch_train_stops, search_stations, search_trains
from .supabase_client import get_supabase


app = FastAPI(title="GAPEKA 2025 Chatbot API")

origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stations/{station_code}/stops", response_model=StationStopsResponse)
def station_stops(station_code: str, limit: int = 30) -> StationStopsResponse:
    station_code = station_code.upper().strip()
    if not station_code:
        raise HTTPException(status_code=400, detail="station_code wajib")
    if limit <= 0 or limit > 200:
        raise HTTPException(status_code=400, detail="limit tidak valid")
    client = get_supabase()
    try:
        rows, total = fetch_station_stops(client, station_code, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase error: {exc}") from exc
    total_val = total if total is not None else len(rows)
    return StationStopsResponse(
        station_code=station_code,
        total=total_val,
        limited=bool(total is not None and total > limit),
        results=rows,
    )


@app.get("/trains/{train_no}/stops", response_model=TrainStopsResponse)
def train_stops(train_no: str, limit: int = 100) -> TrainStopsResponse:
    train_no = train_no.upper().strip()
    if not train_no:
        raise HTTPException(status_code=400, detail="train_no wajib")
    if limit <= 0 or limit > 500:
        raise HTTPException(status_code=400, detail="limit tidak valid")
    client = get_supabase()
    try:
        rows, total = fetch_train_stops(client, train_no, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase error: {exc}") from exc
    total_val = total if total is not None else len(rows)
    train_name = rows[0].get("train_name") if rows else None
    route = rows[0].get("route") if rows else None
    mapped = [
        {
            "station_order": r.get("station_order"),
            "station_name": r.get("station_name"),
            "station_code": r.get("station_code"),
            "arrival_time": r.get("arrival_time"),
            "departure_time": r.get("departure_time"),
        }
        for r in rows
    ]
    return TrainStopsResponse(
        train_no=train_no,
        train_name=train_name,
        route=route,
        total=total_val,
        limited=bool(total is not None and total > limit),
        results=mapped,
    )


@app.get("/search/stations", response_model=SearchStationsResponse)
def search_stations_api(q: str, limit: int = 20) -> SearchStationsResponse:
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="q wajib")
    if limit <= 0 or limit > 100:
        raise HTTPException(status_code=400, detail="limit tidak valid")
    client = get_supabase()
    try:
        results = search_stations(client, q, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase error: {exc}") from exc
    return SearchStationsResponse(query=q, total=len(results), results=results)


@app.get("/search/trains", response_model=SearchTrainsResponse)
def search_trains_api(q: str, limit: int = 20) -> SearchTrainsResponse:
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="q wajib")
    if limit <= 0 or limit > 100:
        raise HTTPException(status_code=400, detail="limit tidak valid")
    client = get_supabase()
    try:
        results = search_trains(client, q, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase error: {exc}") from exc
    return SearchTrainsResponse(query=q, total=len(results), results=results)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return process_chat_request(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/debug/llm")
def debug_llm() -> dict[str, str | bool]:
    return {
        "enabled": bool(settings.llm_enabled),
        "provider": llm_provider(),
        "response_formatting": bool(settings.llm_response_formatting),
    }


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    feedback_id = f"fb_{uuid4().hex[:12]}"
    stored = save_feedback(
        {
            "feedback_id": feedback_id,
            "session_id": req.session_id,
            "request_id": req.request_id,
            "rating": req.rating,
            "score": req.score,
            "correction": req.correction,
            "reviewer_note": req.reviewer_note,
            "is_golden_example": req.is_golden_example,
            "corrected_intent": req.corrected_intent,
        }
    )
    return FeedbackResponse(feedback_id=feedback_id, stored=stored)
