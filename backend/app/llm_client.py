from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from .config import settings


def llm_enabled() -> bool:
    if not settings.llm_enabled:
        return False
    groq_on = bool(settings.groq_api_key and settings.groq_model)
    featherless_on = bool(settings.featherless_api_key and settings.featherless_model)
    return groq_on or featherless_on


def _llm_client() -> tuple[OpenAI, str] | None:
    if not settings.llm_enabled:
        return None
    if settings.groq_api_key and settings.groq_model:
        return (
            OpenAI(
                base_url=settings.groq_base_url,
                api_key=settings.groq_api_key,
                timeout=8.0,
                max_retries=1,
            ),
            settings.groq_model,
        )

    if settings.featherless_api_key and settings.featherless_model:
        return (
            OpenAI(
                base_url=settings.featherless_base_url,
                api_key=settings.featherless_api_key,
                timeout=8.0,
                max_retries=1,
            ),
            settings.featherless_model,
        )

    return None


def llm_provider() -> str:
    if not settings.llm_enabled:
        return "disabled"
    if settings.groq_api_key and settings.groq_model:
        return "groq"
    if settings.featherless_api_key and settings.featherless_model:
        return "featherless"
    return "disabled"


def llm_model_name() -> str | None:
    if not settings.llm_enabled:
        return None
    if settings.groq_api_key and settings.groq_model:
        return settings.groq_model
    if settings.featherless_api_key and settings.featherless_model:
        return settings.featherless_model
    return None


def classify_intent_with_llm(message: str) -> dict[str, Any] | None:
    if not llm_enabled():
        return None

    try:
        client_cfg = _llm_client()
        if client_cfg is None:
            return None
        client, model = client_cfg
        system = (
            "Kamu adalah classifier intent untuk chatbot jadwal kereta. "
            "Balas JSON saja tanpa markdown. "
            "Schema: {\"intent\": \"station_query|train_query|city_to_city_query|unknown\", "
            "\"station_code\": string|null, \"station_query\": string|null, "
            "\"train_no\": string|null, \"train_query\": string|null}."
        )
        user = f"Message: {message}"
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception:
        return None

def run_llm_extraction(system_prompt: str, user_prompt: str) -> str | None:
    if not llm_enabled():
        return None
    try:
        client_cfg = _llm_client()
        if client_cfg is None:
            return None
        client, model = client_cfg
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def format_reply_with_llm(payload: dict[str, Any]) -> str | None:
    if not llm_enabled():
        return None
    if not settings.llm_response_formatting:
        return None

    try:
        client_cfg = _llm_client()
        if client_cfg is None:
            return None
        client, model = client_cfg

        system = (
            "Kamu adalah asisten chatbot jadwal kereta. "
            "Tugasmu: menyusun jawaban yang terdengar natural seperti manusia (Bahasa Indonesia) "
            "berdasarkan data terstruktur yang diberikan. "
            "Aturan wajib: "
            "1) Jangan mengarang fakta; gunakan hanya data yang ada di payload. "
            "2) Jangan menyebut source_page, nomor halaman, sitasi halaman dokumen internal, atau kutipan dokumen. "
            "3) Selalu sertakan kalimat persis: 'Bersumber dari GAPEKA 2025' (taruh di akhir). "
            "4) Jika payload berisi clarification, fokus bertanya klarifikasi dan tampilkan opsi bila ada. "
            "5) Untuk hasil list, ringkas dan rapi (boleh bullet/nomor), jangan terlalu panjang. "
            "6) Jika payload berisi konteks web/Tavily, pisahkan jelas antara data internal jadwal dan info terbaru dari web. "
            "7) Jika payload berisi sources web, boleh menyebut 'sumber web' secara ringkas tetapi jangan dump URL panjang di body jawaban. "
            "8) Jika payload berisi data cuaca (weather), sertakan info cuaca di stasiun sebagai 'added value' yang ramah."
        )

        user = json.dumps(payload, ensure_ascii=False)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None
        if "Bersumber dari GAPEKA 2025" not in text:
            text = "\n".join([text, "Bersumber dari GAPEKA 2025"]).strip()
        return text
    except Exception:
        return None
