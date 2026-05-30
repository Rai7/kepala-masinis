from __future__ import annotations

import re
from dataclasses import dataclass


STATION_CODE_RE = re.compile(r"\b([A-Z]{2,5})\b")
STATION_CODE_RE_LOWER = re.compile(r"\b([a-z]{2,5})\b")
TRAIN_NO_RE = re.compile(r"\bKA\s*([0-9A-Z./-]{1,10})\b", re.IGNORECASE)
RESERVED_CODES = {"KA", "API", "GAPEKA"}
COMMON_WORDS = {
    "apa",
    "yang",
    "saja",
    "dan",
    "di",
    "ke",
    "dari",
    "stasiun",
    "kereta",
    "jadwal",
    "berhenti",
    "tampilkan",
    "cari",
}


@dataclass
class ParsedIntent:
    intent: str
    station_code: str | None = None
    station_query: str | None = None
    train_no: str | None = None
    train_query: str | None = None


def guess_intent(message: str) -> ParsedIntent:
    text = message.strip()
    lower = text.lower()

    if lower.startswith("cari stasiun"):
        q = text.split(" ", 2)[-1].strip()
        return ParsedIntent(intent="station_query", station_query=q)

    if lower.startswith("cari kereta"):
        q = text.split(" ", 2)[-1].strip()
        return ParsedIntent(intent="train_query", train_query=q)

    train_match = TRAIN_NO_RE.search(text)
    if train_match:
        train_no = train_match.group(1).upper()
        return ParsedIntent(intent="train_query", train_no=train_no)

    if any(key in lower for key in ["berhenti di", "jadwal stasiun", "stasiun"]):
        code = extract_station_code(text)
        if code:
            return ParsedIntent(intent="station_query", station_code=code)
        q = extract_station_name_query(text)
        return ParsedIntent(intent="station_query", station_query=q)

    if any(key in lower for key in ["tampilkan jadwal kereta", "jadwal kereta", "kereta "]):
        code = extract_station_code(text)
        if code and "kereta" in lower and "di" in lower:
            return ParsedIntent(intent="station_query", station_code=code)
        q = extract_train_name_query(text)
        return ParsedIntent(intent="train_query", train_query=q)

    if any(key in lower for key in ["dari", "ke", "asal", "tujuan"]):
        return ParsedIntent(intent="city_to_city_query")

    code = extract_station_code(text)
    if code:
        return ParsedIntent(intent="station_query", station_code=code)

    return ParsedIntent(intent="unknown")


def extract_station_code(text: str) -> str | None:
    codes = extract_station_codes(text)
    return codes[0] if codes else None


def extract_station_codes(text: str) -> list[str]:
    candidates = [c for c in STATION_CODE_RE.findall(text) if c not in RESERVED_CODES]
    if candidates:
        return candidates

    lower = text.lower()
    for trigger in ["di ", "stasiun "]:
        idx = lower.rfind(trigger)
        if idx >= 0:
            tail = lower[idx + len(trigger) :]
            for token in STATION_CODE_RE_LOWER.findall(tail):
                if token in COMMON_WORDS:
                    continue
                if token in RESERVED_CODES:
                    continue
                return [token.upper()]

    return []


def extract_station_name_query(text: str) -> str:
    cleaned = re.sub(r"\b(stasiun|jadwal|berhenti|di)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_train_name_query(text: str) -> str:
    cleaned = re.sub(r"\b(ka|kereta|jadwal|tampilkan)\b", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()
