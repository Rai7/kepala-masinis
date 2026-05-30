from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from pymongo import ReturnDocument

from .config import settings
from .mongodb import QUERY_CACHE_COLLECTION, get_collection, redact_sensitive


def normalize_query(text: str) -> str:
    return " ".join(text.lower().strip().split())


def build_cache_key(*, normalized_query: str, role: str) -> str:
    raw = f"{role}:{normalized_query}"
    digest = sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"chat:{role}:{digest}"


def get_cached_response(cache_key: str) -> dict[str, Any] | None:
    if not settings.enable_mongo_cache:
        return None

    collection = get_collection(QUERY_CACHE_COLLECTION)
    if collection is None:
        return None

    now = datetime.now(UTC)
    try:
        return collection.find_one_and_update(
            {"cache_key": cache_key, "expires_at": {"$gt": now}},
            {"$inc": {"hit_count": 1}, "$set": {"updated_at": now}},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
    except Exception:
        return None


def should_cache_response(
    *,
    intent: str,
    clarification: dict[str, Any] | None,
    data: dict[str, Any] | None,
) -> bool:
    if not settings.enable_mongo_cache:
        return False
    if clarification:
        return False
    if intent not in {
        "station_query",
        "train_query",
        "city_to_city_query",
        "search_train_schedule",
        "search_train_by_name",
        "search_route_stops",
    }:
        return False
    return data is not None


def set_cached_response(
    *,
    cache_key: str,
    normalized_query: str,
    intent: str,
    entities: dict[str, Any],
    supabase_result_summary: dict[str, Any] | None,
    response_payload: dict[str, Any],
) -> bool:
    if not settings.enable_mongo_cache:
        return False

    collection = get_collection(QUERY_CACHE_COLLECTION)
    if collection is None:
        return False

    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=max(60, settings.mongo_cache_ttl_seconds))
    doc = {
        "cache_key": cache_key,
        "normalized_query": normalized_query,
        "intent": intent,
        "entities": redact_sensitive(entities),
        "supabase_result_summary": redact_sensitive(supabase_result_summary or {}),
        "response_payload": redact_sensitive(response_payload),
        "hit_count": 0,
        "expires_at": expires_at,
        "created_at": now,
        "updated_at": now,
    }

    try:
        collection.update_one(
            {"cache_key": cache_key},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return True
    except Exception:
        return False
