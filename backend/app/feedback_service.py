from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .mongodb import FEEDBACK_COLLECTION, get_collection, redact_sensitive
from .trace_log_service import get_trace_log_by_request_id


def save_feedback(payload: dict[str, Any]) -> bool:
    collection = get_collection(FEEDBACK_COLLECTION)
    if collection is None:
        return False

    request_id = payload["request_id"]
    trace_log = get_trace_log_by_request_id(request_id) or {}
    now = datetime.now(UTC)
    doc = {
        "feedback_id": payload["feedback_id"],
        "session_id": payload["session_id"],
        "request_id": request_id,
        "user_query": trace_log.get("user_message"),
        "assistant_answer": trace_log.get("assistant_answer"),
        "rating": payload["rating"],
        "score": payload.get("score"),
        "correction": payload.get("correction"),
        "corrected_intent": payload.get("corrected_intent"),
        "reviewer_note": payload.get("reviewer_note"),
        "is_golden_example": payload.get("is_golden_example", False),
        "created_at": now,
        "updated_at": now,
    }

    try:
        collection.update_one(
            {"feedback_id": doc["feedback_id"]},
            {"$set": redact_sensitive(doc)},
            upsert=True,
        )
        return True
    except Exception:
        return False
