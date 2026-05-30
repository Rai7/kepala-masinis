from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from .config import settings


TAVILY_TRIGGER_KEYWORDS = {
    "hari ini",
    "terbaru",
    "update",
    "berita",
    "gangguan",
    "delay",
    "terlambat",
    "batal",
    "banjir",
    "longsor",
    "cuaca",
    "aturan",
    "refund",
    "reschedule",
    "bagasi",
    "check-in",
    "sekitar stasiun",
    "hotel dekat",
    "transportasi dari stasiun",
}


def get_tavily_client() -> httpx.Client | None:
    if not settings.enable_tavily_search:
        return None
    if not settings.tavily_api_key:
        return None
    return httpx.Client(
        base_url="https://api.tavily.com",
        timeout=httpx.Timeout(8.0, connect=3.0),
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
    )


def should_use_tavily(intent: str, user_message: str) -> bool:
    if not settings.enable_tavily_search or not settings.tavily_api_key:
        return False

    lower = user_message.lower()
    if any(keyword in lower for keyword in TAVILY_TRIGGER_KEYWORDS):
        return True

    fallback_patterns = [
        "apa itu",
        "beda apa",
        "melayani apa saja",
        "sekarang",
        "info terbaru",
    ]
    if intent == "unknown" and any(pattern in lower for pattern in fallback_patterns):
        return True

    if intent in {"station_query", "train_query", "city_to_city_query"} and any(
        keyword in lower for keyword in {"hari ini", "gangguan", "delay", "update"}
    ):
        return True

    return False


def build_tavily_query(intent: str, entities: dict[str, Any], user_message: str) -> str:
    lower = user_message.lower()
    if any(keyword in lower for keyword in {"refund", "reschedule", "bagasi", "check-in", "aturan"}):
        return f"site:kai.id {user_message} aturan resmi KAI Indonesia"

    if any(keyword in lower for keyword in {"gangguan", "delay", "terlambat", "batal", "banjir", "longsor", "cuaca"}):
        route_bits = [
            entities.get("station_name"),
            entities.get("station_code"),
            entities.get("origin_station_code"),
            entities.get("destination_station_code"),
            entities.get("train_no"),
            entities.get("train_name"),
        ]
        route_context = " ".join(str(bit) for bit in route_bits if bit)
        return f"{user_message} {route_context} KAI Indonesia terbaru"

    if any(keyword in lower for keyword in {"sekitar stasiun", "hotel dekat", "transportasi dari stasiun"}):
        return f"{user_message} Indonesia"

    if intent == "unknown":
        return f"{user_message} KAI Indonesia"

    return f"{user_message} info terbaru KAI Indonesia"


def normalize_tavily_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in results[: max(1, settings.tavily_max_results)]:
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or item.get("snippet") or "").strip()
        if not url and not title and not snippet:
            continue
        normalized.append(
            {
                "title": title[:200],
                "url": url,
                "snippet": snippet[:500],
            }
        )
    return normalized


def search_web_context(query: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    client = get_tavily_client()
    if client is None:
        return {
            "used": False,
            "query": query,
            "results": [],
            "latency_ms": 0,
            "result_count": 0,
            "error": "Tavily disabled or missing API key",
        }

    payload = {
        "query": query,
        "search_depth": settings.tavily_search_depth,
        "max_results": settings.tavily_max_results,
        "include_answer": False,
        "include_raw_content": False,
        "topic": "general",
    }
    if options:
        payload.update({k: v for k, v in options.items() if v is not None})

    started = perf_counter()
    try:
        response = client.post("/search", json=payload)
        response.raise_for_status()
        raw = response.json()
        results = normalize_tavily_results(raw.get("results") or [])
        return {
            "used": True,
            "query": query,
            "results": results,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "result_count": len(results),
            "error": None,
        }
    except Exception as exc:
        return {
            "used": True,
            "query": query,
            "results": [],
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "result_count": 0,
            "error": str(exc),
        }
    finally:
        client.close()
