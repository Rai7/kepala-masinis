from __future__ import annotations

import re
import time
from typing import Any

from supabase import Client

from .intents import ParsedIntent, extract_station_codes, guess_intent
from .llm_client import classify_intent_with_llm, llm_model_name, llm_provider
from .models import Role
from .repository import fetch_station_stops, fetch_train_stops, search_stations, search_trains


TRAIN_NO_STRICT_RE = re.compile(r"^(?=.*\d)[0-9A-Z./-]{1,10}$")
STATION_CODE_STRICT_RE = re.compile(r"^[A-Z]{2,5}$")
QUERY_STOPWORDS = {"cari", "kereta", "stasiun", "jadwal", "tampilkan"}


def detect_intent(message: str, allow_llm: bool = True) -> tuple[ParsedIntent, dict[str, Any]]:
    baseline = guess_intent(message)
    if baseline.intent != "unknown":
        return baseline, {"method": "heuristic", "llm_latency_ms": 0}

    if not allow_llm:
        return baseline, {"method": "heuristic", "llm_latency_ms": 0}

    started = time.perf_counter()
    llm = classify_intent_with_llm(message)
    llm_latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if llm and isinstance(llm, dict):
        intent = (llm.get("intent") or "unknown").strip()
        station_code = (llm.get("station_code") or "").strip().upper() or None
        station_query = (llm.get("station_query") or "").strip() or None
        train_no = (llm.get("train_no") or "").strip().upper() or None
        train_query = (llm.get("train_query") or "").strip() or None

        if intent not in {"station_query", "train_query", "city_to_city_query", "unknown"}:
            fallback = guess_intent(message)
            return fallback, {"method": "heuristic", "llm_latency_ms": llm_latency_ms}

        if station_code and not STATION_CODE_STRICT_RE.match(station_code):
            station_code = None
        if train_no and not TRAIN_NO_STRICT_RE.match(train_no):
            train_no = None
        if station_query and (len(station_query) < 2 or station_query.lower() in QUERY_STOPWORDS):
            station_query = None
        if train_query and (len(train_query) < 2 or train_query.lower() in QUERY_STOPWORDS):
            train_query = None

        if intent == "station_query" and not station_code and not station_query:
            fallback = guess_intent(message)
            return fallback, {"method": "heuristic", "llm_latency_ms": llm_latency_ms}
        if intent == "train_query" and not train_no and not train_query:
            fallback = guess_intent(message)
            return fallback, {"method": "heuristic", "llm_latency_ms": llm_latency_ms}

        return (
            ParsedIntent(
                intent=intent,
                station_code=station_code,
                station_query=station_query,
                train_no=train_no,
                train_query=train_query,
            ),
            {
                "method": "llm",
                "llm_latency_ms": llm_latency_ms,
                "llm_provider": llm_provider(),
                "llm_model": llm_model_name(),
            },
        )
    return guess_intent(message), {"method": "heuristic", "llm_latency_ms": llm_latency_ms}


def parse_intent(message: str, allow_llm: bool = True) -> ParsedIntent:
    parsed, _ = detect_intent(message, allow_llm=allow_llm)
    return parsed


def format_footer() -> str:
    return "Bersumber dari GAPEKA 2025"


def extract_entities(message: str, parsed: ParsedIntent, data: dict[str, Any] | None) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    if parsed.train_no:
        entities["train_no"] = parsed.train_no
    if parsed.train_query:
        entities["train_name"] = parsed.train_query
    if parsed.station_code:
        entities["station_code"] = parsed.station_code
    if parsed.station_query:
        entities["station_name"] = parsed.station_query

    station_codes = extract_station_codes(message)
    if station_codes:
        if len(station_codes) >= 1:
            entities.setdefault("origin_station_code", station_codes[0])
        if len(station_codes) >= 2:
            entities["destination_station_code"] = station_codes[-1]

    if data:
        if data.get("segment_to"):
            entities["destination_station_code"] = data["segment_to"]
        if data.get("route"):
            entities["route"] = data["route"]
    return entities


def format_station_results(
    station_code: str,
    rows: list[dict[str, Any]],
    total_count: int | None,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    limited = bool(total_count is not None and total_count > limit)
    lines: list[str] = []
    for r in rows:
        train_no = r.get("train_no")
        train_name = r.get("train_name") or ""
        arrival = r.get("arrival_time") or "-"
        departure = r.get("departure_time") or "-"
        lines.append(f"KA {train_no} {train_name} — datang {arrival}, berangkat {departure}".strip())

    header = f"Kereta yang berhenti di {station_code} (maks {limit}):"
    if not rows:
        header = f"Tidak menemukan data kereta berhenti di {station_code}."

    if limited:
        header = header + " Hasil dibatasi."

    reply = "\n".join([header, *lines, format_footer()]).strip()
    data = {
        "station_code": station_code,
        "total": total_count if total_count is not None else len(rows),
        "limited": limited,
        "results": rows,
    }
    return reply, data


def format_train_results(
    train_no: str,
    rows: list[dict[str, Any]],
    total_count: int | None,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    limited = bool(total_count is not None and total_count > limit)
    train_name = rows[0].get("train_name") if rows else None
    route = rows[0].get("route") if rows else None

    lines: list[str] = []
    for r in rows:
        order = r.get("station_order")
        station_name = r.get("station_name")
        station_code = r.get("station_code")
        arrival = r.get("arrival_time") or "-"
        departure = r.get("departure_time") or "-"
        lines.append(f"{order}. {station_name} ({station_code}) — datang {arrival}, berangkat {departure}")

    title = f"Stop list KA {train_no}"
    if train_name:
        title = f"{title} ({train_name})"
    if route:
        title = f"{title} — {route}"
    if not rows:
        title = f"Tidak menemukan stop list untuk KA {train_no}."
    if limited:
        title = title + " Hasil dibatasi."

    reply = "\n".join([title, *lines, format_footer()]).strip()
    data = {
        "train_no": train_no,
        "train_name": train_name,
        "route": route,
        "total": total_count if total_count is not None else len(rows),
        "limited": limited,
        "results": [
            {
                "station_order": r.get("station_order"),
                "station_name": r.get("station_name"),
                "station_code": r.get("station_code"),
                "arrival_time": r.get("arrival_time"),
                "departure_time": r.get("departure_time"),
            }
            for r in rows
        ],
    }
    return reply, data


def handle_station_query(
    client: Client,
    parsed: ParsedIntent,
    limit: int,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    station_code = (parsed.station_code or "").upper().strip()
    if station_code:
        rows, total = fetch_station_stops(client, station_code, limit=limit)
        reply, data = format_station_results(station_code, rows, total, limit)
        return reply, data, None, {
            "used_supabase": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_summary": {
                "table": "train_schedules",
                "steps": [
                    {
                        "operation": "fetch_station_stops",
                        "filters": {"station_code": station_code},
                        "rows_returned": len(rows),
                    }
                ],
                "rows_returned": len(rows),
            },
        }

    q = (parsed.station_query or "").strip()
    if not q:
        reply = "\n".join(["Saya perlu kode stasiun (mis. GMR) atau nama stasiun untuk dicari.", format_footer()])
        return reply, None, {"question": "Sebutkan kode stasiun (mis. GMR) atau nama stasiun yang Anda maksud."}, {
            "used_supabase": False,
            "latency_ms": 0,
            "query_summary": None,
        }

    matches = search_stations(client, q, limit=10)
    if not matches:
        reply = "\n".join([f"Tidak menemukan stasiun untuk query: {q}", format_footer()])
        return reply, {"query": q, "total": 0, "results": []}, None, {
            "used_supabase": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_summary": {
                "table": "train_schedules",
                "steps": [
                    {
                        "operation": "search_stations",
                        "filters": {"query": q},
                        "rows_returned": 0,
                    }
                ],
                "rows_returned": 0,
            },
        }
    if len(matches) > 1:
        options = [f"{m['station_name']} ({m['station_code']})" for m in matches]
        reply = "\n".join([f"Saya menemukan beberapa stasiun untuk '{q}'. Pilih salah satu kode stasiun:", *options, format_footer()])
        return (
            reply,
            {"query": q, "total": len(matches), "results": matches},
            {"question": "Stasiun mana yang Anda maksud? Kirim kode stasiunnya.", "options": options},
            {
                "used_supabase": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "query_summary": {
                    "table": "train_schedules",
                    "steps": [
                        {
                            "operation": "search_stations",
                            "filters": {"query": q},
                            "rows_returned": len(matches),
                        }
                    ],
                    "rows_returned": len(matches),
                },
            },
        )

    station_code = matches[0]["station_code"]
    rows, total = fetch_station_stops(client, station_code, limit=limit)
    reply, data = format_station_results(station_code, rows, total, limit)
    return reply, data, None, {
        "used_supabase": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "query_summary": {
            "table": "train_schedules",
            "steps": [
                {
                    "operation": "search_stations",
                    "filters": {"query": q},
                    "rows_returned": 1,
                },
                {
                    "operation": "fetch_station_stops",
                    "filters": {"station_code": station_code},
                    "rows_returned": len(rows),
                },
            ],
            "rows_returned": len(rows),
        },
    }


def handle_train_query(
    client: Client,
    parsed: ParsedIntent,
    message: str,
    role: Role,
    limit: int,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    started = time.perf_counter()
    lower = message.lower()
    wants_full_public = any(
        k in lower
        for k in [
            "seluruh",
            "semua",
            "rute lengkap",
            "berhenti dimana saja",
            "stop list",
        ]
    )
    wants_full_internal = any(k in lower for k in ["seluruh", "semua", "rute lengkap", "seluruh rute"])
    station_codes = extract_station_codes(message)
    dest_code = station_codes[-1] if station_codes else None

    train_no = (parsed.train_no or "").upper().strip()
    if not train_no:
        q = (parsed.train_query or "").strip()
        if not q:
            reply = "\n".join(["Saya perlu nomor KA (mis. KA 447) atau nama kereta untuk dicari.", format_footer()])
            return reply, None, {"question": "Sebutkan nomor KA atau nama kereta yang Anda maksud."}, {
                "used_supabase": False,
                "latency_ms": 0,
                "query_summary": None,
            }
        matches = search_trains(client, q, limit=10)
        if not matches:
            reply = "\n".join([f"Tidak menemukan kereta untuk query: {q}", format_footer()])
            return reply, {"query": q, "total": 0, "results": []}, None, {
                "used_supabase": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "query_summary": {
                    "table": "train_schedules",
                    "steps": [
                        {
                            "operation": "search_trains",
                            "filters": {"query": q},
                            "rows_returned": 0,
                        }
                    ],
                    "rows_returned": 0,
                },
            }
        if len(matches) > 1:
            options = [f"KA {m['train_no']} {m.get('train_name') or ''}".strip() for m in matches]
            reply = "\n".join([f"Saya menemukan beberapa kereta untuk '{q}'. Pilih salah satu:", *options, format_footer()])
            return (
                reply,
                {"query": q, "total": len(matches), "results": matches},
                {"question": "Kereta mana yang Anda maksud? Kirim nomor KA-nya.", "options": options},
                {
                    "used_supabase": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "query_summary": {
                        "table": "train_schedules",
                        "steps": [
                            {
                                "operation": "search_trains",
                                "filters": {"query": q},
                                "rows_returned": len(matches),
                            }
                        ],
                        "rows_returned": len(matches),
                    },
                },
            )
        train_no = matches[0]["train_no"]

    if role == Role.internal:
        wants_full = wants_full_internal
        if not wants_full and not dest_code:
            reply = "\n".join(
                [
                    f"Anda meminta jadwal KA {train_no}. Apakah butuh segmen tertentu atau seluruh rute? Jika segmen tertentu, sebutkan stasiun tujuan (kode/nama).",
                    format_footer(),
                ]
            )
            return reply, {"train_no": train_no}, {"question": "Butuh segmen tertentu atau seluruh rute? Jika segmen, sebutkan tujuan."}, {
                "used_supabase": False,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "query_summary": None,
            }

        rows, total = fetch_train_stops(client, train_no, limit=500)
        if dest_code:
            idx = next((i for i, r in enumerate(rows) if r.get("station_code") == dest_code), None)
            if idx is None:
                reply = "\n".join([f"Saya tidak menemukan stasiun tujuan {dest_code} pada rute KA {train_no}.", format_footer()])
                return reply, {"train_no": train_no, "destination": dest_code}, None, {
                    "used_supabase": True,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "query_summary": {
                        "table": "train_schedules",
                        "steps": [
                            {
                                "operation": "fetch_train_stops",
                                "filters": {"train_no": train_no},
                                "rows_returned": len(rows),
                            }
                        ],
                        "rows_returned": len(rows),
                    },
                }
            segment = rows[: idx + 1]
            display = segment[:limit]
            reply, data = format_train_results(train_no, display, len(segment), limit)
            data["segment_to"] = dest_code
            return reply, data, None, {
                "used_supabase": True,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "query_summary": {
                    "table": "train_schedules",
                    "steps": [
                        {
                            "operation": "fetch_train_stops",
                            "filters": {"train_no": train_no, "destination": dest_code},
                            "rows_returned": len(segment),
                        }
                    ],
                    "rows_returned": len(segment),
                },
            }

        display = rows[:limit]
        reply, data = format_train_results(train_no, display, total, limit)
        return reply, data, None, {
            "used_supabase": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_summary": {
                "table": "train_schedules",
                "steps": [
                    {
                        "operation": "fetch_train_stops",
                        "filters": {"train_no": train_no},
                        "rows_returned": len(display),
                    }
                ],
                "rows_returned": len(display),
            },
        }

    wants_full = wants_full_public
    if not wants_full and not dest_code:
        reply = "\n".join(
            [
                f"Untuk KA {train_no}, Anda butuh segmen tertentu atau seluruh rute? Jika segmen, sebutkan stasiun tujuan (kode/nama).",
                format_footer(),
            ]
        )
        return reply, {"train_no": train_no}, {"question": "Butuh segmen tertentu atau seluruh rute? Jika segmen, sebutkan tujuan."}, {
            "used_supabase": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "query_summary": None,
        }

    rows, total = fetch_train_stops(client, train_no, limit=limit)
    reply, data = format_train_results(train_no, rows, total, limit)
    return reply, data, None, {
        "used_supabase": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "query_summary": {
            "table": "train_schedules",
            "steps": [
                {
                    "operation": "fetch_train_stops",
                    "filters": {"train_no": train_no},
                    "rows_returned": len(rows),
                }
            ],
            "rows_returned": len(rows),
        },
    }


def handle_city_to_city_query() -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    reply = "\n".join(
        [
            "Fitur city-to-city masih skeleton untuk MVP karena belum ada mapping kota→stasiun. "
            "Untuk saat ini, gunakan kode stasiun asal/tujuan (mis. GMR, SBI) atau query stasiun/kereta.",
            format_footer(),
        ]
    )
    return reply, None, None, {"used_supabase": False, "latency_ms": 0, "query_summary": None}
