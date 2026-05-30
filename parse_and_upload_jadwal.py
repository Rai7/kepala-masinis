import argparse
import os
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import fitz
import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client
from tqdm import tqdm

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional fallback
    pdfplumber = None


PDF_PATH = Path("DAFTAR JADWAL KERETA API GAPEKA 2025.pdf")
SQL_PATH = Path("jadwal.sql")
OUTPUT_PATH = Path("output/jadwal_kereta.csv")
EXPECTED_COLUMNS = [
    "train_no",
    "train_name",
    "route",
    "station_order",
    "station_name",
    "station_code",
    "arrival_time",
    "departure_time",
    "note",
    "source_page",
]
TRAIN_HEADER_RE = re.compile(
    r"^KA\s+([A-Z0-9./-]+)\s+\((.*?)\)\s+Lintas(?:\s+Pelayanan\s+([A-Z0-9-]+))?\s*$",
    re.IGNORECASE,
)
ROW_INLINE_RE = re.compile(r"^(\d+)\s+(.+)$")
STATION_CODE_RE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*$")
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
ADMIN_SUBSTRINGS = (
    "PT. KERETA API INDONESIA",
    "KANTOR PUSAT",
    "DAFTAR WAKTU",
    "Berlaku Pada GAPEKA",
    "Toka:",
    "Fax:",
)
PAGE_SKIP_MARKERS = (
    "DAFTAR ISI",
    "NOMOR JAM",
    "KELAS KERETA API",
    "URT KA BER DAT",
)
TABLE_HEADER_LINES = {
    "No.",
    "Stasiun dan Perhentian",
    "Datang Berangkat",
    "Keterangan",
    "Datang",
    "Berangkat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse jadwal kereta dari PDF GAPEKA, validasi, export CSV, dan upload ke Supabase."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, cleaning, validasi, dan export CSV tanpa upload ke Supabase.",
    )
    mode.add_argument(
        "--upload",
        action="store_true",
        help="Parse, cleaning, validasi, export CSV, lalu upload ke Supabase.",
    )
    parser.add_argument(
        "--pdf-path",
        default=str(PDF_PATH),
        help="Path ke file PDF GAPEKA.",
    )
    parser.add_argument(
        "--sql-path",
        default=str(SQL_PATH),
        help="Path ke file schema SQL target Supabase.",
    )
    parser.add_argument(
        "--output-path",
        default=str(OUTPUT_PATH),
        help="Path output CSV hasil parsing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Jumlah row per batch upload ke Supabase.",
    )
    args = parser.parse_args()
    if not args.dry_run and not args.upload:
        args.dry_run = True
    return args


def analyze_schema(sql_path: Path) -> Dict[str, object]:
    sql_text = sql_path.read_text(encoding="utf-8")
    match = re.search(
        r"create\s+table\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\);",
        sql_text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Tidak dapat menemukan CREATE TABLE pada {sql_path}")

    table_name = match.group(1)
    body = match.group(2)
    columns: List[Dict[str, str]] = []
    primary_key: Optional[str] = None

    for raw_line in body.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue

        column_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z0-9_]+)", line)
        if not column_match:
            continue

        name = column_match.group(1)
        data_type = column_match.group(2)
        columns.append({"name": name, "type": data_type})

        if re.search(r"\bprimary\s+key\b", line, re.IGNORECASE):
            primary_key = name

    if not columns:
        raise ValueError(f"Schema pada {sql_path} tidak memiliki kolom yang dapat diparse")

    return {
        "table_name": table_name,
        "primary_key": primary_key,
        "columns": columns,
    }


def create_supabase_client() -> Client:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL atau SUPABASE_SERVICE_ROLE_KEY belum tersedia di .env")
    return create_client(url, key)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_train_header(line: str) -> bool:
    return bool(TRAIN_HEADER_RE.match(line))


def parse_train_header(line: str) -> Optional[Dict[str, str]]:
    match = TRAIN_HEADER_RE.match(line)
    if not match:
        return None
    return {
        "train_no": match.group(1).strip(),
        "train_name": normalize_line(match.group(2)),
        "route": match.group(3).strip() if match.group(3) else None,
    }


def is_table_header_line(line: str) -> bool:
    return line in TABLE_HEADER_LINES or line.startswith(
        "No. Stasiun dan Perhentian Datang Berangkat Keterangan"
    )


def is_admin_line(line: str) -> bool:
    if not line:
        return True
    if any(fragment in line for fragment in ADMIN_SUBSTRINGS):
        return True
    if line in {"ISI:"}:
        return True
    return False


def looks_like_time_token(line: str) -> bool:
    return line in {"Ls", "-"} or bool(TIME_RE.match(line))


def starts_new_station(line: str) -> bool:
    return line.isdigit() or bool(ROW_INLINE_RE.match(line))


def normalize_time(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = normalize_line(value)
    if cleaned in {"", "-", "Ls"}:
        return None
    if TIME_RE.match(cleaned):
        return cleaned
    return None


def split_station_name_and_code(station_raw: str) -> Tuple[str, Optional[str]]:
    cleaned = normalize_line(station_raw)
    match = STATION_CODE_RE.match(cleaned)
    if match:
        station_name = normalize_line(match.group(1))
        station_code = normalize_line(match.group(2))
        return station_name, station_code
    return cleaned, None


def split_station_and_tail(content: str) -> Tuple[str, str]:
    match = re.match(r"^(?P<station>.+?\([^)]+\))\s*(?P<tail>.*)$", content)
    if match:
        return normalize_line(match.group("station")), normalize_line(match.group("tail"))

    fallback = re.match(
        r"^(?P<station>.*?)(?P<tail>(?:Ls|-|\d{2}:\d{2}:\d{2}|Berh\b).*)$",
        content,
    )
    if fallback:
        return normalize_line(fallback.group("station")), normalize_line(fallback.group("tail"))

    return normalize_line(content), ""


def extract_time_tokens_and_note(tail: str) -> Tuple[List[str], Optional[str]]:
    if not tail:
        return [], None

    parts = tail.split()
    time_tokens: List[str] = []
    idx = 0

    while idx < len(parts):
        token = parts[idx]
        if token in {"Ls", "-"} or TIME_RE.match(token):
            time_tokens.append(token)
            idx += 1
            continue
        break

    note = normalize_line(" ".join(parts[idx:])) or None
    return time_tokens, note


def interpret_time_tokens(tokens: List[str], station_order: int) -> Tuple[Optional[str], Optional[str]]:
    cleaned = [normalize_line(token) for token in tokens if normalize_line(token)]
    if not cleaned:
        return None, None

    if len(cleaned) == 1:
        token = cleaned[0]
        if token in {"Ls", "-"}:
            return None, None
        if station_order == 1:
            return None, normalize_time(token)
        return normalize_time(token), None

    first, second = cleaned[0], cleaned[1]
    arrival = normalize_time(first)
    departure = normalize_time(second)
    return arrival, departure


def get_page_text(doc: fitz.Document, page_index: int, plumber_doc=None) -> str:
    page = doc.load_page(page_index)
    text = page.get_text("text", sort=True)
    if text.strip():
        return text

    if plumber_doc is not None:
        fallback_text = plumber_doc.pages[page_index].extract_text() or ""
        if fallback_text.strip():
            return fallback_text

    return text


def extract_lines_from_page(text: str) -> List[str]:
    return [normalize_line(line) for line in text.splitlines() if normalize_line(line)]


def parse_station_entry(
    lines: List[str],
    start_idx: int,
    current_train: Optional[Dict[str, str]],
    page_number: int,
) -> Tuple[Optional[Dict[str, object]], int]:
    if current_train is None:
        return None, start_idx + 1

    line = lines[start_idx]
    station_order: Optional[int] = None
    station_parts: List[str] = []
    idx = start_idx

    inline_match = ROW_INLINE_RE.match(line)
    if inline_match:
        station_order = int(inline_match.group(1))
        station_inline = normalize_line(inline_match.group(2))
        station_raw, tail = split_station_and_tail(station_inline)

        if not station_raw or looks_like_time_token(station_raw) or TIME_RE.match(station_raw):
            station_parts_from_context: List[str] = []
            if start_idx > 0:
                previous_line = lines[start_idx - 1]
                if (
                    not is_train_header(previous_line)
                    and not is_table_header_line(previous_line)
                    and not starts_new_station(previous_line)
                    and not is_admin_line(previous_line)
                ):
                    station_parts_from_context.append(previous_line)
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                if (
                    not is_train_header(next_line)
                    and not is_table_header_line(next_line)
                    and not starts_new_station(next_line)
                    and not is_admin_line(next_line)
                ):
                    station_parts_from_context.append(next_line)
                    idx += 1
            station_raw = normalize_line(" ".join(station_parts_from_context))

        time_tokens, inline_note = extract_time_tokens_and_note(tail)
        station_name, station_code = split_station_name_and_code(station_raw)
        arrival_time, departure_time = interpret_time_tokens(time_tokens, station_order)
        note = inline_note
        idx += 1

        while idx < len(lines):
            candidate = lines[idx]
            if (
                is_train_header(candidate)
                or is_table_header_line(candidate)
                or starts_new_station(candidate)
                or is_admin_line(candidate)
            ):
                break
            note = normalize_line(" ".join([part for part in [note, candidate] if part])) or None
            idx += 1

        row = {
            "train_no": current_train["train_no"],
            "train_name": current_train["train_name"],
            "route": current_train["route"],
            "station_order": station_order,
            "station_name": station_name,
            "station_code": station_code,
            "arrival_time": arrival_time,
            "departure_time": departure_time,
            "note": note,
            "source_page": page_number,
        }
        return row, idx
    elif line.isdigit():
        station_order = int(line)
        idx += 1
        if idx >= len(lines):
            return None, idx
        next_line = lines[idx]
        if (
            is_train_header(next_line)
            or is_table_header_line(next_line)
            or is_admin_line(next_line)
            or next_line.isdigit()
        ):
            return None, idx
        station_parts.append(next_line)
        idx += 1
    else:
        return None, start_idx + 1

    while idx < len(lines):
        candidate = lines[idx]
        if (
            looks_like_time_token(candidate)
            or is_train_header(candidate)
            or is_table_header_line(candidate)
            or starts_new_station(candidate)
            or is_admin_line(candidate)
        ):
            break
        station_parts.append(candidate)
        idx += 1

    if not station_parts:
        return None, idx

    time_tokens: List[str] = []
    while idx < len(lines) and len(time_tokens) < 2:
        candidate = lines[idx]
        if looks_like_time_token(candidate):
            time_tokens.append(candidate)
            idx += 1
            continue
        break

    notes: List[str] = []
    while idx < len(lines):
        candidate = lines[idx]
        if (
            is_train_header(candidate)
            or is_table_header_line(candidate)
            or starts_new_station(candidate)
            or is_admin_line(candidate)
        ):
            break
        notes.append(candidate)
        idx += 1

    station_raw = normalize_line(" ".join(station_parts))
    station_name, station_code = split_station_name_and_code(station_raw)
    arrival_time, departure_time = interpret_time_tokens(time_tokens, station_order)
    note = normalize_line(" ".join(notes)) or None

    row = {
        "train_no": current_train["train_no"],
        "train_name": current_train["train_name"],
        "route": current_train["route"],
        "station_order": station_order,
        "station_name": station_name,
        "station_code": station_code,
        "arrival_time": arrival_time,
        "departure_time": departure_time,
        "note": note,
        "source_page": page_number,
    }
    return row, idx


def iter_schedule_rows(pdf_path: Path) -> Generator[Dict[str, object], None, None]:
    plumber_doc = None
    current_train: Optional[Dict[str, str]] = None
    doc = fitz.open(pdf_path)

    try:
        if pdfplumber is not None:
            plumber_doc = pdfplumber.open(pdf_path)

        for page_index in tqdm(range(doc.page_count), desc="Parsing PDF", unit="page"):
            page_number = page_index + 1
            try:
                text = get_page_text(doc, page_index, plumber_doc)
                lines = extract_lines_from_page(text)
                first_lines = " ".join(lines[:5])
                if any(marker in first_lines for marker in PAGE_SKIP_MARKERS):
                    continue
                idx = 0

                while idx < len(lines):
                    line = lines[idx]

                    header = parse_train_header(line)
                    if header:
                        current_train = header
                        idx += 1
                        continue

                    if is_table_header_line(line) or is_admin_line(line):
                        idx += 1
                        continue

                    row, next_idx = parse_station_entry(lines, idx, current_train, page_number)
                    if row:
                        yield row
                    idx = next_idx if next_idx > idx else idx + 1
            except Exception as exc:
                print(f"[WARN] Gagal memproses page {page_number}: {exc}")
                continue
    finally:
        doc.close()
        if plumber_doc is not None:
            plumber_doc.close()


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            df[column] = df[column].apply(
                lambda value: normalize_line(value) if isinstance(value, str) else value
            )

    df["arrival_time"] = df["arrival_time"].apply(normalize_time)
    df["departure_time"] = df["departure_time"].apply(normalize_time)
    df["note"] = df["note"].replace({"": None})
    df["station_name"] = df["station_name"].replace({"": None})
    df["station_code"] = df["station_code"].replace({"": None})

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicate_count = before - len(df)
    return df, duplicate_count


def filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    invalid_mask = df["station_code"].isna() & (
        df["station_name"].isna()
        | df["station_name"].astype(str).str.contains(r"^//\)?$", regex=True, na=False)
        | df["station_name"].astype(str).str.contains("Lintas Pelayanan", na=False)
        | df["train_name"].astype(str).str.contains("Lintas Pelayanan", na=False)
    )
    return df.loc[~invalid_mask].reset_index(drop=True)


def fill_missing_routes(df: pd.DataFrame) -> pd.DataFrame:
    for train_no, group in df.groupby("train_no"):
        route_values = group["route"].dropna().astype(str)
        route = route_values.iloc[0] if not route_values.empty else None

        if not route:
            ordered = group.sort_values("station_order")
            codes = ordered["station_code"].dropna().astype(str)
            if len(codes) >= 2:
                route = f"{codes.iloc[0]}-{codes.iloc[-1]}"

        if route:
            df.loc[df["train_no"] == train_no, "route"] = route

    return df


def ensure_dataframe_columns(df: pd.DataFrame, schema: Dict[str, object]) -> pd.DataFrame:
    schema_columns = [column["name"] for column in schema["columns"]]
    target_columns = [column for column in schema_columns if column != schema["primary_key"]]

    missing_columns = [column for column in target_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataframe belum memiliki kolom schema: {missing_columns}")

    return df[target_columns]


def print_schema_summary(schema: Dict[str, object]) -> None:
    print("\n=== ANALISIS SCHEMA ===")
    print(f"Table          : {schema['table_name']}")
    print(f"Primary Key    : {schema['primary_key']}")
    print("Kolom:")
    for column in schema["columns"]:
        print(f"- {column['name']}: {column['type']}")


def print_validation_summary(df: pd.DataFrame, duplicate_count: int) -> None:
    print("\n=== VALIDATION REPORT ===")
    print(f"Total row                : {len(df)}")
    print(f"Total unique train_no    : {df['train_no'].nunique(dropna=True)}")
    print(f"Total unique station_code: {df['station_code'].nunique(dropna=True)}")
    print(f"Total duplicate row      : {duplicate_count}")
    print("Missing value count per kolom:")
    for column, count in df.isna().sum().items():
        print(f"- {column}: {count}")

    print("\n=== DF.HEAD(20) ===")
    print(df.head(20).to_string(index=False))

    sample_size = min(20, len(df))
    print("\n=== SAMPLE 20 ROW RANDOM ===")
    if sample_size == 0:
        print("(data kosong)")
    else:
        print(df.sample(sample_size, random_state=42).to_string(index=False))


def print_quality_check(df: pd.DataFrame, schema: Dict[str, object]) -> None:
    print("\n=== QUALITY CHECK ===")
    print("Struktur dataframe final:")
    print(df.dtypes.to_string())
    print(f"\nJumlah record final: {len(df)}")

    print("\nContoh 20 record pertama:")
    print(df.head(20).to_string(index=False))

    print("\nMapping kolom dataframe ke kolom Supabase:")
    for column in schema["columns"]:
        if column["name"] == schema["primary_key"]:
            print(f"- {column['name']} ({column['type']}): diisi otomatis oleh database")
        else:
            print(f"- {column['name']} ({column['type']}): dataframe['{column['name']}']")

    issues: List[str] = []
    if df["station_code"].isna().any():
        issues.append(
            f"Masih ada {int(df['station_code'].isna().sum())} row tanpa station_code yang mungkin butuh aturan parsing tambahan."
        )
    if (df["arrival_time"].isna() & df["departure_time"].isna()).any():
        issues.append(
            f"Ada {int((df['arrival_time'].isna() & df['departure_time'].isna()).sum())} row tanpa arrival dan departure."
        )
    if df["note"].notna().any():
        issues.append(
            "Kolom note masih berisi annotation lintasan/persilangan dari PDF; validasikan apakah seluruh note memang perlu disimpan."
        )
    if not issues:
        issues.append("Tidak ada anomali besar yang terdeteksi secara otomatis dari hasil dry-run.")

    print("\nPotensi masalah parsing yang masih perlu diperbaiki:")
    for issue in issues:
        print(f"- {issue}")


def export_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def upload_to_supabase(
    df: pd.DataFrame,
    schema: Dict[str, object],
    batch_size: int,
) -> None:
    client = create_supabase_client()
    table_name = schema["table_name"]
    upload_df = df.where(pd.notna(df), None)
    records = upload_df.to_dict(orient="records")

    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index in range(total_batches):
        start = batch_index * batch_size
        end = start + batch_size
        batch = records[start:end]
        print(f"Uploading batch {batch_index + 1}/{total_batches}")
        try:
            client.table(table_name).insert(batch).execute()
        except Exception as exc:
            raise RuntimeError(
                f"Gagal insert batch {batch_index + 1}/{total_batches}: {exc}"
            ) from exc


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path)
    sql_path = Path(args.sql_path)
    output_path = Path(args.output_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {pdf_path}")
    if not sql_path.exists():
        raise FileNotFoundError(f"File SQL tidak ditemukan: {sql_path}")

    schema = analyze_schema(sql_path)
    print_schema_summary(schema)

    try:
        rows = list(iter_schedule_rows(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Terjadi error saat parsing PDF: {exc}") from exc

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    df, duplicate_count = clean_dataframe(df)
    df = filter_invalid_rows(df)
    df = fill_missing_routes(df)
    df = ensure_dataframe_columns(df, schema)
    print_validation_summary(df, duplicate_count)

    try:
        export_csv(df, output_path)
        print(f"\nCSV berhasil disimpan ke: {output_path}")
    except Exception as exc:
        raise RuntimeError(f"Gagal export CSV ke {output_path}: {exc}") from exc

    print_quality_check(df, schema)

    if args.upload:
        try:
            upload_to_supabase(df, schema, args.batch_size)
            print("\nUpload ke Supabase selesai.")
        except Exception as exc:
            raise RuntimeError(f"Supabase error: {exc}") from exc
    else:
        print("\nDry-run selesai. Upload dilewati.")


if __name__ == "__main__":
    main()
