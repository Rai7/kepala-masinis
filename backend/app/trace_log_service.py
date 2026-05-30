from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .config import settings
from .mongodb import TRACE_LOGS_COLLECTION, get_collection, redact_sensitive


def write_trace_log(payload: dict[str, Any]) -> bool:
    if not settings.enable_mongo_logging:
        return False

    collection = get_collection(TRACE_LOGS_COLLECTION)
    if collection is None:
        return False

    doc = redact_sensitive(payload)
    doc.setdefault("created_at", datetime.now(UTC))

    try:
        collection.insert_one(doc)
        return True
    except Exception:
        return False


def get_trace_log_by_request_id(request_id: str) -> dict[str, Any] | None:
    collection = get_collection(TRACE_LOGS_COLLECTION)
    if collection is None:
        return None

    try:
        return collection.find_one({"request_id": request_id}, {"_id": 0})
    except Exception:
        return None
