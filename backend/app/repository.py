from __future__ import annotations

from typing import Any

from supabase import Client


def fetch_station_stops(
    client: Client,
    station_code: str,
    limit: int,
) -> tuple[list[dict[str, Any]], int | None]:
    query = (
        client.table("train_schedules")
        .select("train_no,train_name,arrival_time,departure_time", count="exact")
        .eq("station_code", station_code)
        .order("departure_time", desc=False)
        .limit(limit)
    )
    resp = query.execute()
    return resp.data or [], getattr(resp, "count", None)


def fetch_train_stops(
    client: Client,
    train_no: str,
    limit: int,
) -> tuple[list[dict[str, Any]], int | None]:
    query = (
        client.table("train_schedules")
        .select("train_no,train_name,route,station_order,station_name,station_code,arrival_time,departure_time", count="exact")
        .eq("train_no", train_no)
        .order("station_order", desc=False)
        .limit(limit)
    )
    resp = query.execute()
    return resp.data or [], getattr(resp, "count", None)


def search_stations(
    client: Client,
    q: str,
    limit: int,
) -> list[dict[str, Any]]:
    q_norm = q.strip()
    pattern = f"%{q_norm}%"
    resp = (
        client.table("train_schedules")
        .select("station_code,station_name")
        .or_(f"station_code.ilike.{pattern},station_name.ilike.{pattern}")
        .limit(2000)
        .execute()
    )

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for row in resp.data or []:
        code = row.get("station_code")
        name = row.get("station_name")
        if not code or not name:
            continue
        if code in seen:
            continue
        seen.add(code)
        results.append({"station_code": code, "station_name": name})
        if len(results) >= limit:
            break
    return results


def search_trains(
    client: Client,
    q: str,
    limit: int,
) -> list[dict[str, Any]]:
    q_norm = q.strip()
    pattern = f"%{q_norm}%"
    resp = (
        client.table("train_schedules")
        .select("train_no,train_name,route")
        .or_(f"train_no.ilike.{pattern},train_name.ilike.{pattern}")
        .limit(2000)
        .execute()
    )

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for row in resp.data or []:
        train_no = row.get("train_no")
        if not train_no:
            continue
        if train_no in seen:
            continue
        seen.add(train_no)
        results.append(
            {
                "train_no": train_no,
                "train_name": row.get("train_name"),
                "route": row.get("route"),
            }
        )
        if len(results) >= limit:
            break
    return results
