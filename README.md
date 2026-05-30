## Chatbot GAPEKA 2025 (Supabase + FastAPI + React)
Aplikasi chatbot untuk tanya jawab jadwal kereta berbasis query terstruktur ke Supabase. Semua jawaban wajib memuat kalimat “Bersumber dari GAPEKA 2025”.

### Struktur Project
- `backend/`: FastAPI API server
- `frontend/`: React + Vite web UI
- `parse_and_upload_jadwal.py`: pipeline ETL (sudah digunakan, tidak dipakai oleh aplikasi web)

## Konfigurasi Environment
Salin `.env.example` menjadi `.env` lalu isi nilai yang benar.

Backend menggunakan:
- `SUPABASE_URL` (base project URL, tanpa `/rest/v1/`)
- `SUPABASE_SERVICE_ROLE_KEY` (hanya di backend)
- MongoDB pendukung (opsional tapi direkomendasikan untuk history/logging/cache):
  - `MONGODB_URI`
  - `MONGODB_DB_NAME`
  - `ENABLE_MONGO_LOGGING`
  - `ENABLE_MONGO_CACHE`
  - `MONGO_CACHE_TTL_SECONDS`
- Tavily untuk konteks web terbaru (opsional):
  - `TAVILY_API_KEY`
  - `ENABLE_TAVILY_SEARCH`
  - `TAVILY_MAX_RESULTS`
  - `TAVILY_SEARCH_DEPTH`

LLM (opsional) untuk intent classification/formatting:
- Toggle global:
  - `LLM_ENABLED` (default `true`)
  - `LLM_RESPONSE_FORMATTING` (default `true`)
- `FEATHERLESS_API_KEY`
- `FEATHERLESS_BASE_URL` (default `https://api.featherless.ai/v1`)
- `FEATHERLESS_MODEL` (wajib jika ingin LLM aktif)
- Alternatif (recommended): Groq OpenAI-compatible
  - `GROQ_API_KEY`
  - `GROQ_BASE_URL` (default `https://api.groq.com/openai/v1`)
  - `GROQ_MODEL`

Frontend menggunakan:
- `frontend/.env` berisi `VITE_API_BASE_URL` mengarah ke backend.

## Orkestrasi Agent
- Backend `/chat` menggunakan LangGraph dengan satu graph utama railway assistant.
- Checkpointer development memakai `InMemorySaver`.
- `thread_id` LangGraph diset ke `session_id` agar state percakapan bisa dilanjutkan per sesi.
- LangGraph dipakai untuk orchestration state/checkpoint, bukan menggantikan MongoDB trace logs atau chat history permanen.

## Menjalankan Lokal

### Backend (FastAPI)
1. Buat virtualenv (opsional) dan install dependency:

```bash
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
```

2. Jalankan server:

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Cek health:

```bash
curl http://localhost:8000/health
```

### Frontend (React + Vite)
1. Install dependency:

```bash
cd frontend
npm install
```

2. Set env:
buat `frontend/.env` dari `frontend/.env.example`, lalu sesuaikan:

```env
VITE_API_BASE_URL=http://localhost:8000
```

3. Jalankan dev server:

```bash
npm run dev
```

## API Ringkas
- `GET /health`
- `POST /chat`
- `POST /feedback`
- `GET /stations/{station_code}/stops?limit=30`
- `GET /trains/{train_no}/stops?limit=100`
- `GET /search/stations?q=...`
- `GET /search/trains?q=...`

### Payload `/chat`
- Field tambahan yang didukung:
  - `session_id` (opsional, jika kosong akan digenerate)
  - `user_id` (opsional)
  - `ui_metadata` (opsional)
  - `use_llm` (opsional)
- Jika Tavily dipakai, response `/chat` juga mengandung:
  - `data.web_context_used`
  - `data.sources`
  - `metadata.tavily_used`
  - `metadata.tavily_query`
- Response metadata juga memuat `events` dari node LangGraph untuk progress/debugging backend.

### Payload `/feedback`
```json
{
  "session_id": "sess_abc123",
  "request_id": "req_abc123",
  "rating": "helpful",
  "score": 5,
  "correction": null,
  "reviewer_note": null,
  "is_golden_example": false,
  "corrected_intent": null
}
```

## Catatan Policy
- Semua query jadwal harus lewat backend (frontend tidak boleh mengakses Supabase langsung).
- Response chatbot dilarang menampilkan `source_page`, sitasi halaman, link dokumen, atau kutipan dokumen.
- Tavily hanya dipakai untuk konteks web terbaru/eksternal, bukan menggantikan query jadwal dari Supabase.
- Role:
  - `public`: output terbatas
  - `internal`: jika query kereta tanpa segmen, bot wajib minta klarifikasi

## Deployment

### Frontend ke Vercel
- Build output: `frontend/dist`
- Env di Vercel:
  - `VITE_API_BASE_URL=https://<backend-domain>`

### Backend ke Railway/Render
- Deploy folder `backend/`
- Start command (contoh):

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

- Env di Railway/Render:
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
  - `MONGODB_URI`
  - `MONGODB_DB_NAME`
  - `ENABLE_MONGO_LOGGING`
  - `ENABLE_MONGO_CACHE`
  - `MONGO_CACHE_TTL_SECONDS`
  - `TAVILY_API_KEY`
  - `ENABLE_TAVILY_SEARCH`
  - `TAVILY_MAX_RESULTS`
  - `TAVILY_SEARCH_DEPTH`
  - `CORS_ALLOW_ORIGINS` (origin Vercel frontend)
  - opsional `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL`
  - opsional `FEATHERLESS_API_KEY`, `FEATHERLESS_BASE_URL`, `FEATHERLESS_MODEL`

### Opsi: Backend FastAPI di Vercel
- Konfigurasi minimal sudah disiapkan di `backend/vercel.json` dan entrypoint `backend/api/index.py`.
- Untuk memakai opsi ini, deploy folder `backend/` sebagai project Vercel terpisah dan set env vars backend di Vercel.

## Asumsi
- City-to-city masih skeleton karena belum ada tabel mapping kota→stasiun. Untuk full implementation dibutuhkan dataset mapping kota dan daftar stasiun yang relevan.
- MongoDB bersifat best-effort: jika koneksi gagal, endpoint utama tetap jalan dan tetap memakai Supabase sebagai source of truth.
- Tavily juga bersifat best-effort: jika timeout/error/tidak relevan, jawaban tetap dikembalikan dari Supabase atau fallback internal tanpa membuat `/chat` crash.
