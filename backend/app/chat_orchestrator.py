from __future__ import annotations

import re
import time
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .chat_history_service import upsert_chat_session
from .config import settings
from .llm_client import format_reply_with_llm, llm_model_name, llm_provider
from .models import ChatRequest, ChatResponse, Role
from .query_cache_service import (
    build_cache_key,
    get_cached_response,
    normalize_query,
    set_cached_response,
    should_cache_response,
)
from .service import (
    detect_intent,
    extract_entities,
    handle_city_to_city_query,
    handle_station_query,
    handle_train_query,
)
from .supabase_client import get_supabase
from .tavily_service import build_tavily_query, search_web_context, should_use_tavily
from .trace_log_service import write_trace_log


GraphIntent = Literal[
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


class RailwayAssistantState(TypedDict, total=False):
    request_id: str
    session_id: str
    user_id: str | None
    role: str
    allow_llm: bool
    ui_metadata: dict[str, Any] | None
    user_message: str
    normalized_query: str
    intent: GraphIntent | str | None
    raw_intent: str | None
    parsed_intent: Any
    entities: dict[str, Any]
    previous_entities: dict[str, Any]
    cache_key: str
    cache_hit: bool
    cache_result: dict[str, Any] | None
    supabase_used: bool
    supabase_result: dict[str, Any] | None
    supabase_reply: str | None
    supabase_clarification: dict[str, Any] | None
    supabase_query_summary: dict[str, Any] | None
    supabase_latency_ms: float | None
    tavily_requested: bool
    tavily_used: bool
    tavily_query: str | None
    tavily_result: dict[str, Any] | None
    tavily_sources: list[dict[str, str]]
    tavily_latency_ms: float | None
    tavily_result_count: int
    tavily_error_message: str | None
    weather_data: dict[str, Any] | None
    llm_provider: str | None
    llm_model: str | None
    llm_latency_ms: float | None
    final_answer: str | None
    response_data: dict[str, Any] | None
    response_metadata: dict[str, Any] | None
    clarification: dict[str, Any] | None
    fallback_path: str | None
    error: str | None
    events: list[dict[str, str]]
    created_at: str
    updated_at: str
    user_message_id: str
    assistant_message_id: str
    started_perf: float


_GRAPH_CHECKPOINTER = InMemorySaver()
_GRAPH = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _add_event(state: RailwayAssistantState, step: str, message: str) -> list[dict[str, str]]:
    return [
        *(state.get("events") or []),
        {"step": step, "message": message, "timestamp": _now_iso()},
    ]


def _keyword_match(message: str, keywords: set[str]) -> bool:
    lower = message.lower()
    return any(keyword in lower for keyword in keywords)


def _extract_temporal_entities(message: str) -> dict[str, Any]:
    lower = message.lower()
    entities: dict[str, Any] = {}
    if "hari ini" in lower:
        entities["date"] = "hari ini"
    elif "besok" in lower:
        entities["date"] = "besok"

    if "pagi" in lower:
        entities["time_preference"] = "pagi"
    elif "siang" in lower:
        entities["time_preference"] = "siang"
    elif "sore" in lower:
        entities["time_preference"] = "sore"
    elif "malam" in lower:
        entities["time_preference"] = "malam"
    return entities


def _merge_memory_entities(message: str, entities: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    lower = message.lower()
    follow_up_tokens = {"itu", "nya", "tersebut", "lanjut", "yang tadi"}
    is_follow_up = any(token in lower for token in follow_up_tokens)
    merged = dict(previous or {}) if is_follow_up else {}
    merged.update({k: v for k, v in entities.items() if v is not None})
    return merged


def _map_graph_intent(message: str, raw_intent: str, entities: dict[str, Any]) -> tuple[GraphIntent, bool]:
    lower = message.lower()
    has_web_trigger = should_use_tavily(raw_intent or "unknown", message)

    if _keyword_match(lower, {"aturan", "refund", "reschedule", "bagasi", "check-in"}):
        return "kai_policy_question", True
    if _keyword_match(lower, {"sekitar stasiun", "hotel dekat", "transportasi dari stasiun"}):
        return "travel_context", True
    if _keyword_match(lower, {"gangguan", "delay", "terlambat", "batal", "banjir", "longsor", "cuaca", "berita", "update", "terbaru"}):
        if raw_intent in {"station_query", "train_query", "city_to_city_query"}:
            if raw_intent == "train_query" and (
                "berhenti dimana" in lower or "stop list" in lower or "rute" in lower
            ):
                return "search_route_stops", True
            if raw_intent == "train_query":
                return "search_train_by_name", True
            return "search_train_schedule", True
        return "railway_disruption_news", True

    if raw_intent == "station_query":
        return "search_train_schedule", has_web_trigger
    if raw_intent == "train_query":
        if "berhenti dimana" in lower or "stop list" in lower or "rute" in lower:
            return "search_route_stops", has_web_trigger
        return "search_train_by_name", has_web_trigger
    if raw_intent == "city_to_city_query":
        return "search_train_schedule", has_web_trigger

    if re.search(r"\bapa itu\b|\bbeda apa\b|\bmelayani apa saja\b", lower):
        return "fallback_web_question", settings.enable_tavily_search
    if entities.get("station_name") and not any(v for v in entities.values() if v != entities.get("station_name")):
        return "station_info", settings.enable_tavily_search
    return "general_chat", has_web_trigger


def _merge_response_data(
    *,
    supabase_result: dict[str, Any] | None,
    tavily_sources: list[dict[str, str]],
    tavily_used: bool,
) -> dict[str, Any] | None:
    if supabase_result is None and not tavily_used:
        return None
    return {
        "supabase_result": supabase_result,
        "web_context_used": tavily_used,
        "sources": tavily_sources,
    }


def _build_response_metadata(state: RailwayAssistantState) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "entities": state.get("entities") or {},
        "tavily_used": bool(state.get("tavily_used")),
        "tavily_query": state.get("tavily_query"),
        "tavily_result_count": state.get("tavily_result_count", 0),
        "events": state.get("events") or [],
    }


def _append_web_sources(reply: str, tavily_results: list[dict[str, str]], tavily_failed: bool) -> str:
    lines = [reply.rstrip()]
    if tavily_results:
        lines.extend(["", "Catatan informasi terbaru dari web:"])
        for item in tavily_results[:3]:
            title = item.get("title") or "Sumber web"
            snippet = (item.get("snippet") or "").strip()[:220]
            lines.append(f"- {title}: {snippet}" if snippet else f"- {title}")
        lines.extend(["", "Sumber:"])
        for item in tavily_results[:5]:
            title = item.get("title") or item.get("url") or "Sumber"
            url = item.get("url") or "-"
            lines.append(f"- {title}: {url}")
    elif tavily_failed:
        lines.extend(["", "Catatan informasi terbaru dari web belum tersedia saat ini."])

    text = "\n".join(lines).strip()
    if "Bersumber dari GAPEKA 2025" not in text:
        text = "\n".join([text, "Bersumber dari GAPEKA 2025"]).strip()
    return text


def _safe_template_answer(state: RailwayAssistantState) -> str:
    lines: list[str] = []
    supabase_reply = state.get("supabase_reply")
    if supabase_reply:
        lines.extend(["Berdasarkan database jadwal:", supabase_reply])

    if not lines and state.get("cache_hit") and state.get("cache_result"):
        cached_reply = ((state.get("cache_result") or {}).get("response_payload") or {}).get("reply")
        if cached_reply:
            lines.append(cached_reply)

    if not lines:
        if state.get("intent") in {"search_train_schedule", "search_train_by_name", "search_route_stops"}:
            lines.append(
                "Saya belum menemukan jawaban yang cukup lengkap. Tolong perjelas nama kereta, kode stasiun, asal, tujuan, atau tanggal perjalanan."
            )
        else:
            lines.append(
                "Saya belum bisa menjawab pertanyaan itu dengan yakin. Tolong perjelas konteks seperti stasiun, kereta, rute, atau kebutuhan info terbaru."
            )

    text = "\n".join(lines).strip()
    return _append_web_sources(
        text,
        state.get("tavily_sources") or [],
        bool(state.get("tavily_requested")) and not bool(state.get("tavily_sources")),
    )


def _run_supabase_logic(state: RailwayAssistantState) -> dict[str, Any]:
    client = get_supabase()
    role = Role(state.get("role") or Role.public.value)
    parsed_state = state.get("raw_intent")
    if parsed_state == "station_query":
        reply, data, clarification, query_meta = handle_station_query(
            client,
            state["parsed_intent"],  # type: ignore[index]
            limit=30,
        )
    elif parsed_state == "train_query":
        reply, data, clarification, query_meta = handle_train_query(
            client,
            state["parsed_intent"],  # type: ignore[index]
            message=state["user_message"],
            role=role,
            limit=100,
        )
    elif parsed_state == "city_to_city_query":
        reply, data, clarification, query_meta = handle_city_to_city_query()
    else:
        reply = None
        data = None
        clarification = None
        query_meta = {"used_supabase": False, "latency_ms": 0, "query_summary": None}
    return {
        "supabase_reply": reply,
        "supabase_result": data,
        "clarification": clarification,
        "supabase_used": bool(query_meta.get("used_supabase")),
        "supabase_latency_ms": query_meta.get("latency_ms"),
        "supabase_query_summary": query_meta.get("query_summary"),
    }


def normalize_input(state: RailwayAssistantState) -> RailwayAssistantState:
    now = _now_iso()
    session_id = state.get("session_id") or f"sess_{uuid4().hex[:12]}"
    return {
        "request_id": state.get("request_id") or f"req_{uuid4().hex[:12]}",
        "session_id": session_id,
        "user_id": state.get("user_id"),
        "role": state.get("role") or Role.public.value,
        "allow_llm": bool(state.get("allow_llm", True)),
        "ui_metadata": state.get("ui_metadata"),
        "user_message": state.get("user_message") or "",
        "normalized_query": normalize_query(state.get("user_message") or ""),
        "cache_key": build_cache_key(
            normalized_query=normalize_query(state.get("user_message") or ""),
            role=state.get("role") or Role.public.value,
        ),
        "cache_hit": False,
        "cache_result": None,
        "supabase_used": False,
        "supabase_result": None,
        "supabase_reply": None,
        "supabase_clarification": None,
        "supabase_query_summary": None,
        "supabase_latency_ms": None,
        "tavily_requested": False,
        "tavily_used": False,
        "tavily_query": None,
        "tavily_result": None,
        "tavily_sources": [],
        "tavily_latency_ms": None,
        "tavily_result_count": 0,
        "tavily_error_message": None,
        "weather_data": None,
        "llm_provider": llm_provider() if state.get("allow_llm", True) else None,
        "llm_model": llm_model_name() if state.get("allow_llm", True) else None,
        "llm_latency_ms": 0.0,
        "final_answer": None,
        "response_data": None,
        "response_metadata": None,
        "clarification": None,
        "fallback_path": None,
        "error": None,
        "events": [{"step": "normalize_input", "message": "Memahami pertanyaan...", "timestamp": now}],
        "created_at": state.get("created_at") or now,
        "updated_at": now,
        "user_message_id": f"msg_{uuid4().hex[:12]}",
        "assistant_message_id": f"msg_{uuid4().hex[:12]}",
        "started_perf": time.perf_counter(),
        "previous_entities": state.get("entities") or {},
    }


def detect_intent_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "detect_intent", "Menganalisis intent dan entity...")
    try:
        parsed, intent_meta = detect_intent(state["user_message"], allow_llm=state["allow_llm"])
        entities = extract_entities(state["user_message"], parsed, None)
        entities.update(_extract_temporal_entities(state["user_message"]))
        entities = _merge_memory_entities(state["user_message"], entities, state.get("previous_entities") or {})
        intent, tavily_requested = _map_graph_intent(state["user_message"], parsed.intent, entities)
        return {
            "intent": intent,
            "raw_intent": parsed.intent,
            "parsed_intent": parsed,  # type: ignore[typeddict-item]
            "entities": entities,
            "tavily_requested": tavily_requested,
            "llm_provider": intent_meta.get("llm_provider") or state.get("llm_provider"),
            "llm_model": intent_meta.get("llm_model") or state.get("llm_model"),
            "llm_latency_ms": float(intent_meta.get("llm_latency_ms") or 0),
            "updated_at": _now_iso(),
            "events": events,
        }
    except Exception as exc:
        return {
            "intent": "general_chat",
            "raw_intent": "unknown",
            "entities": state.get("previous_entities") or {},
            "error": str(exc),
            "updated_at": _now_iso(),
            "events": events,
        }


def check_mongo_cache_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "check_mongo_cache", "Mengecek cache...")
    try:
        cached = get_cached_response(state["cache_key"])
        if cached is None:
            return {"cache_hit": False, "cache_result": None, "updated_at": _now_iso(), "events": events}
        return {"cache_hit": True, "cache_result": cached, "fallback_path": "cache", "updated_at": _now_iso(), "events": events}
    except Exception as exc:
        return {
            "cache_hit": False,
            "cache_result": None,
            "error": state.get("error") or str(exc),
            "updated_at": _now_iso(),
            "events": events,
        }


def route_intent(state: RailwayAssistantState) -> str:
    if state.get("cache_hit"):
        return "generate_final_answer"
    if state.get("intent") in {"search_train_schedule", "search_train_by_name", "search_route_stops"}:
        if state.get("tavily_requested"):
            return "hybrid_supabase_tavily"
        if state.get("intent") == "search_train_schedule":
            return "query_supabase_schedule"
        if state.get("intent") == "search_train_by_name":
            return "query_supabase_train"
        return "query_supabase_route_stops"
    if state.get("intent") in {
        "station_info",
        "railway_disruption_news",
        "kai_policy_question",
        "travel_context",
        "fallback_web_question",
    }:
        return "query_tavily_external_context"
    return "fallback_answer"


def query_supabase_schedule_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "query_supabase_schedule", "Mencari jadwal di database...")
    try:
        result = _run_supabase_logic(state)
        return {"events": events, "updated_at": _now_iso(), **result}
    except Exception as exc:
        return {
            "supabase_used": False,
            "error": str(exc),
            "fallback_path": "supabase_error",
            "updated_at": _now_iso(),
            "events": events,
        }


def query_supabase_train_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "query_supabase_train", "Mencari kereta di database...")
    try:
        result = _run_supabase_logic(state)
        return {"events": events, "updated_at": _now_iso(), **result}
    except Exception as exc:
        return {
            "supabase_used": False,
            "error": str(exc),
            "fallback_path": "supabase_error",
            "updated_at": _now_iso(),
            "events": events,
        }


def query_supabase_route_stops_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "query_supabase_route_stops", "Mencari stop dan rute di database...")
    try:
        result = _run_supabase_logic(state)
        return {"events": events, "updated_at": _now_iso(), **result}
    except Exception as exc:
        return {
            "supabase_used": False,
            "error": str(exc),
            "fallback_path": "supabase_error",
            "updated_at": _now_iso(),
            "events": events,
        }


def query_tavily_external_context_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "query_tavily_external_context", "Mengecek informasi terbaru dari web...")
    if not settings.enable_tavily_search:
        return {
            "tavily_requested": False,
            "tavily_used": False,
            "updated_at": _now_iso(),
            "events": events,
        }
    try:
        query = build_tavily_query(str(state.get("intent") or "general_chat"), state.get("entities") or {}, state["user_message"])
        result = search_web_context(query, options={"max_results": settings.tavily_max_results})
        return {
            "tavily_requested": True,
            "tavily_used": True,
            "tavily_query": query,
            "tavily_result": result,
            "tavily_sources": result.get("results") or [],
            "tavily_latency_ms": result.get("latency_ms"),
            "tavily_result_count": result.get("result_count", 0),
            "tavily_error_message": result.get("error"),
            "updated_at": _now_iso(),
            "events": events,
        }
    except Exception as exc:
        return {
            "tavily_requested": True,
            "tavily_used": False,
            "tavily_error_message": str(exc),
            "updated_at": _now_iso(),
            "events": events,
        }


def hybrid_supabase_tavily_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "hybrid_supabase_tavily", "Menggabungkan data jadwal dan info terbaru...")
    merged: RailwayAssistantState = {"events": events, "updated_at": _now_iso()}
    try:
        merged.update(_run_supabase_logic(state))
    except Exception as exc:
        merged.update({"supabase_used": False, "error": str(exc), "fallback_path": "supabase_error"})

    try:
        query = build_tavily_query(str(state.get("intent") or "general_chat"), state.get("entities") or {}, state["user_message"])
        result = search_web_context(query, options={"max_results": settings.tavily_max_results})
        merged.update(
            {
                "tavily_requested": True,
                "tavily_used": True,
                "tavily_query": query,
                "tavily_result": result,
                "tavily_sources": result.get("results") or [],
                "tavily_latency_ms": result.get("latency_ms"),
                "tavily_result_count": result.get("result_count", 0),
                "tavily_error_message": result.get("error"),
            }
        )
    except Exception as exc:
        merged.update(
            {
                "tavily_requested": True,
                "tavily_used": False,
                "tavily_error_message": str(exc),
            }
        )
    return merged


def fetch_weather_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "fetch_weather", "Mengecek cuaca stasiun...")
    from .openweather_service import resolve_station_coordinates, get_weather
    
    weather_data = {}
    stations_to_check = set()
    
    entities = state.get("entities") or {}
    if entities.get("station_name"):
        stations_to_check.add(entities["station_name"])
    if entities.get("origin_station_code") or entities.get("origin_station"):
        stations_to_check.add(entities.get("origin_station") or entities.get("origin_station_code"))
    if entities.get("destination_station_code") or entities.get("destination_station"):
        stations_to_check.add(entities.get("destination_station") or entities.get("destination_station_code"))
        
    supabase_result = state.get("supabase_result")
    if supabase_result and isinstance(supabase_result.get("data"), list):
        for item in supabase_result["data"][:2]:
            if "station_name" in item:
                stations_to_check.add(item["station_name"])
            
    if not stations_to_check:
        return {"weather_data": None, "updated_at": _now_iso(), "events": events}
        
    for station in list(stations_to_check)[:2]:
        if not isinstance(station, str):
            continue
        lat, lon = resolve_station_coordinates(station)
        if lat is not None and lon is not None:
            w = get_weather(lat, lon)
            if w and w.get("used"):
                weather_data[station] = w
                
    if weather_data:
        events = _add_event({"events": events}, "fetch_weather", f"Berhasil mendapatkan cuaca untuk {len(weather_data)} stasiun.")
        return {"weather_data": weather_data, "updated_at": _now_iso(), "events": events}
        
    return {"weather_data": None, "updated_at": _now_iso(), "events": events}

def fallback_answer_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "fallback_answer", "Menyiapkan fallback aman...")
    return {
        "fallback_path": state.get("fallback_path") or "fallback",
        "updated_at": _now_iso(),
        "events": events,
    }


def generate_final_answer_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "generate_final_answer", "Menyusun jawaban...")
    if state.get("cache_hit") and state.get("cache_result"):
        payload = ((state.get("cache_result") or {}).get("response_payload") or {})
        reply = payload.get("reply") or _safe_template_answer(state)
        data = payload.get("data")
        clarification = payload.get("clarification")
        metadata = payload.get("metadata") or _build_response_metadata(state)
        return {
            "final_answer": reply,
            "response_data": data,
            "clarification": clarification,
            "response_metadata": metadata,
            "updated_at": _now_iso(),
            "events": events,
        }

    response_data = _merge_response_data(
        supabase_result=state.get("supabase_result"),
        tavily_sources=state.get("tavily_sources") or [],
        tavily_used=bool(state.get("tavily_sources")),
    )
    if state.get("weather_data"):
        response_data["weather"] = state["weather_data"]

    reply = None
    llm_started = None
    if state.get("allow_llm"):
        llm_started = time.perf_counter()
        try:
            payload = {
                "intent": state.get("intent"),
                "role": state.get("role"),
                "message": state.get("user_message"),
                "data": state.get("supabase_result"),
                "clarification": state.get("clarification") or state.get("supabase_clarification"),
                "tavily_context": state.get("tavily_sources") or [],
                "web_context_used": bool(state.get("tavily_sources")),
                "events": events,
            }
            if state.get("weather_data"):
                payload["weather"] = state["weather_data"]
            reply = format_reply_with_llm(payload)
        except Exception:
            reply = None

    llm_latency_ms = float(state.get("llm_latency_ms") or 0)
    if llm_started is not None:
        llm_latency_ms = round(llm_latency_ms + (time.perf_counter() - llm_started) * 1000, 2)

    if not reply:
        reply = _safe_template_answer(state)
    else:
        reply = _append_web_sources(
            reply,
            state.get("tavily_sources") or [],
            bool(state.get("tavily_requested")) and not bool(state.get("tavily_sources")),
        )

    if state.get("supabase_reply") and "Berdasarkan database jadwal" not in reply:
        reply = "\n".join(["Berdasarkan database jadwal:", reply]).strip()

    metadata = _build_response_metadata(state)
    fallback_path = state.get("fallback_path") or "heuristic"
    if state.get("cache_hit"):
        fallback_path = "cache"
    elif state.get("tavily_requested") and state.get("supabase_used"):
        fallback_path = "hybrid_supabase_tavily"
    elif state.get("tavily_requested"):
        fallback_path = "tavily_only"
    elif state.get("supabase_used"):
        fallback_path = "supabase_only"

    return {
        "final_answer": reply,
        "response_data": response_data,
        "clarification": state.get("clarification") or state.get("supabase_clarification"),
        "response_metadata": metadata,
        "llm_latency_ms": llm_latency_ms,
        "fallback_path": fallback_path,
        "updated_at": _now_iso(),
        "events": events,
    }


def save_chat_history_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "save_chat_history", "Menyimpan riwayat...")
    try:
        upsert_chat_session(
            session_id=state["session_id"],
            user_id=state.get("user_id"),
            detected_intent=str(state.get("intent") or "general_chat"),
            entities=state.get("entities") or {},
            last_query_summary={
                "supabase": state.get("supabase_query_summary"),
                "tavily": {
                    "used": state.get("tavily_requested"),
                    "query": state.get("tavily_query"),
                    "result_count": state.get("tavily_result_count", 0),
                },
                "events": events,
            },
            ui_metadata=state.get("ui_metadata"),
            user_message={
                "message_id": state["user_message_id"],
                "role": "user",
                "content": state["user_message"],
                "timestamp": datetime.now(UTC),
            },
            assistant_message={
                "message_id": state["assistant_message_id"],
                "role": "assistant",
                "content": state.get("final_answer"),
                "timestamp": datetime.now(UTC),
            },
        )
    except Exception:
        pass
    return {"updated_at": _now_iso(), "events": events}


def save_trace_log_node(state: RailwayAssistantState) -> RailwayAssistantState:
    events = _add_event(state, "save_trace_log", "Menyimpan trace...")
    total_latency_ms = round((time.perf_counter() - state.get("started_perf", time.perf_counter())) * 1000, 2)
    try:
        write_trace_log(
            {
                "request_id": state["request_id"],
                "session_id": state["session_id"],
                "endpoint": "/chat",
                "user_message": state["user_message"],
                "assistant_answer": state.get("final_answer"),
                "detected_intent": state.get("intent"),
                "entities": state.get("entities") or {},
                "supabase_query_summary": state.get("supabase_query_summary"),
                "supabase_used": state.get("supabase_used", False),
                "supabase_latency_ms": state.get("supabase_latency_ms"),
                "tavily_used": state.get("tavily_requested", False),
                "tavily_query": state.get("tavily_query"),
                "tavily_latency_ms": state.get("tavily_latency_ms"),
                "tavily_result_count": state.get("tavily_result_count", 0),
                "tavily_sources": state.get("tavily_sources") or [],
                "tavily_error_message": state.get("tavily_error_message"),
                "weather_data": state.get("weather_data"),
                "llm_provider": state.get("llm_provider"),
                "llm_model": state.get("llm_model"),
                "llm_latency_ms": state.get("llm_latency_ms"),
                "total_latency_ms": total_latency_ms,
                "fallback_path": state.get("fallback_path"),
                "events": events,
                "error_message": state.get("error"),
            }
        )
    except Exception:
        pass
    return {"updated_at": _now_iso(), "events": events}


def _build_graph():
    workflow = StateGraph(RailwayAssistantState)
    workflow.add_node("normalize_input", normalize_input)
    workflow.add_node("detect_intent", detect_intent_node)
    workflow.add_node("check_mongo_cache", check_mongo_cache_node)
    workflow.add_node("query_supabase_schedule", query_supabase_schedule_node)
    workflow.add_node("query_supabase_train", query_supabase_train_node)
    workflow.add_node("query_supabase_route_stops", query_supabase_route_stops_node)
    workflow.add_node("query_tavily_external_context", query_tavily_external_context_node)
    workflow.add_node("hybrid_supabase_tavily", hybrid_supabase_tavily_node)
    workflow.add_node("fallback_answer", fallback_answer_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("generate_final_answer", generate_final_answer_node)
    workflow.add_node("save_chat_history", save_chat_history_node)
    workflow.add_node("save_trace_log", save_trace_log_node)

    workflow.add_edge(START, "normalize_input")
    workflow.add_edge("normalize_input", "detect_intent")
    workflow.add_edge("detect_intent", "check_mongo_cache")
    workflow.add_conditional_edges(
        "check_mongo_cache",
        route_intent,
        {
            "generate_final_answer": "generate_final_answer",
            "query_supabase_schedule": "query_supabase_schedule",
            "query_supabase_train": "query_supabase_train",
            "query_supabase_route_stops": "query_supabase_route_stops",
            "query_tavily_external_context": "query_tavily_external_context",
            "hybrid_supabase_tavily": "hybrid_supabase_tavily",
            "fallback_answer": "fallback_answer",
        },
    )
    workflow.add_edge("query_supabase_schedule", "fetch_weather")
    workflow.add_edge("query_supabase_train", "fetch_weather")
    workflow.add_edge("query_supabase_route_stops", "fetch_weather")
    workflow.add_edge("query_tavily_external_context", "fetch_weather")
    workflow.add_edge("hybrid_supabase_tavily", "fetch_weather")
    workflow.add_edge("fallback_answer", "fetch_weather")
    workflow.add_edge("fetch_weather", "generate_final_answer")
    workflow.add_edge("generate_final_answer", "save_chat_history")
    workflow.add_edge("save_chat_history", "save_trace_log")
    workflow.add_edge("save_trace_log", END)
    return workflow.compile(checkpointer=_GRAPH_CHECKPOINTER)


def get_railway_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def process_chat_request(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id or f"sess_{uuid4().hex[:12]}"
    allow_llm = settings.llm_enabled if req.use_llm is None else bool(req.use_llm)
    graph = get_railway_graph()
    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(
        {
            "session_id": session_id,
            "user_id": req.user_id,
            "role": (req.role or Role.public).value,
            "allow_llm": allow_llm,
            "ui_metadata": req.ui_metadata,
            "user_message": req.message,
        },
        config,
    )
    metadata = dict(final_state.get("response_metadata") or {})
    metadata["events"] = final_state.get("events") or []
    return ChatResponse(
        intent=str(final_state.get("intent") or "general_chat"),
        reply=final_state.get("final_answer")
        or "\n".join(
            [
                "Saya belum bisa menyusun jawaban yang tepat. Tolong perjelas nama kereta, stasiun, atau kebutuhan info terbaru.",
                "Bersumber dari GAPEKA 2025",
            ]
        ),
        data=final_state.get("response_data"),
        clarification=final_state.get("clarification"),
        metadata=metadata,
        session_id=final_state.get("session_id") or session_id,
        request_id=final_state.get("request_id") or f"req_{uuid4().hex[:12]}",
        cache_hit=bool(final_state.get("cache_hit")),
        user_message_id=final_state.get("user_message_id"),
        assistant_message_id=final_state.get("assistant_message_id"),
    )
