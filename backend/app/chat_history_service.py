from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .mongodb import CHAT_SESSIONS_COLLECTION, get_collection, redact_sensitive


def upsert_chat_session(
    *,
    session_id: str,
    user_id: str | None,
    detected_intent: str,
    entities: dict[str, Any] | None,
    last_query_summary: dict[str, Any] | None,
    ui_metadata: dict[str, Any] | None,
    user_message: dict[str, Any],
    assistant_message: dict[str, Any],
) -> bool:
    collection = get_collection(CHAT_SESSIONS_COLLECTION)
    if collection is None:
        return False

    now = datetime.now(UTC)
    try:
        collection.update_one(
            {"session_id": session_id},
            {
                "$setOnInsert": {
                    "session_id": session_id,
                    "created_at": now,
                },
                "$set": {
                    "user_id": user_id,
                    "detected_intent": detected_intent,
                    "entities": redact_sensitive(entities or {}),
                    "last_query_summary": redact_sensitive(last_query_summary or {}),
                    "ui_metadata": redact_sensitive(ui_metadata or {}),
                    "updated_at": now,
                },
                "$push": {
                    "messages": {
                        "$each": [
                            redact_sensitive(user_message),
                            redact_sensitive(assistant_message),
                        ]
                    }
                },
            },
            upsert=True,
        )
        return True
    except Exception:
        return False
