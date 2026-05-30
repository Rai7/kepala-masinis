from __future__ import annotations

from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from .config import settings


CHAT_SESSIONS_COLLECTION = "chat_sessions"
TRACE_LOGS_COLLECTION = "trace_logs"
FEEDBACK_COLLECTION = "feedback"
QUERY_CACHE_COLLECTION = "query_cache"

_mongo_client: MongoClient[dict[str, Any]] | None = None
_mongo_db: Database[dict[str, Any]] | None = None
_indexes_ensured = False


def mongo_enabled() -> bool:
    return bool(settings.mongodb_uri and settings.mongodb_db_name)


def redact_sensitive(value: Any) -> Any:
    sensitive_markers = {
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "authorization",
        "service_role_key",
        "mongodb_uri",
    }

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_lower = key.lower()
            if any(marker in key_lower for marker in sensitive_markers):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def _ensure_indexes(db: Database[dict[str, Any]]) -> None:
    global _indexes_ensured
    if _indexes_ensured:
        return

    db[CHAT_SESSIONS_COLLECTION].create_index([("session_id", ASCENDING)], unique=True)
    db[CHAT_SESSIONS_COLLECTION].create_index([("user_id", ASCENDING)])
    db[CHAT_SESSIONS_COLLECTION].create_index([("updated_at", DESCENDING)])

    db[TRACE_LOGS_COLLECTION].create_index([("request_id", ASCENDING)], unique=True)
    db[TRACE_LOGS_COLLECTION].create_index([("session_id", ASCENDING)])
    db[TRACE_LOGS_COLLECTION].create_index([("created_at", DESCENDING)])

    db[FEEDBACK_COLLECTION].create_index([("feedback_id", ASCENDING)], unique=True)
    db[FEEDBACK_COLLECTION].create_index([("session_id", ASCENDING)])
    db[FEEDBACK_COLLECTION].create_index([("rating", ASCENDING)])
    db[FEEDBACK_COLLECTION].create_index([("is_golden_example", ASCENDING)])
    db[FEEDBACK_COLLECTION].create_index([("created_at", DESCENDING)])

    db[QUERY_CACHE_COLLECTION].create_index([("cache_key", ASCENDING)], unique=True)
    db[QUERY_CACHE_COLLECTION].create_index([("expires_at", ASCENDING)], expireAfterSeconds=0)

    _indexes_ensured = True


def get_mongo_db() -> Database[dict[str, Any]] | None:
    global _mongo_client, _mongo_db
    if not mongo_enabled():
        return None

    if _mongo_db is not None:
        return _mongo_db

    try:
        _mongo_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=2000,
            connectTimeoutMS=2000,
            socketTimeoutMS=4000,
            appname="gapeka-chatbot",
        )
        _mongo_db = _mongo_client[settings.mongodb_db_name]
        _ensure_indexes(_mongo_db)
        return _mongo_db
    except Exception:
        _mongo_client = None
        _mongo_db = None
        return None


def get_collection(name: str) -> Collection[dict[str, Any]] | None:
    db = get_mongo_db()
    if db is None:
        return None
    return db[name]
