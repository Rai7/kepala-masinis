import re
import fitz
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
import os

load_dotenv()

PDF_PATH = "DAFTAR JADWAL KERETA API GAPEKA 2025.pdf"

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

doc = fitz.open(PDF_PATH)

rows = []
current_train_no = None
current_train_name = None
current_route = None

header_pattern = re.compile(
    r"KA\s+(\w+)\s+\((.*?)\)\s+Lintas Pelayanan\s+([A-Z\-]+)",
    re.IGNORECASE
)

row_pattern = re.compile(
    r"^(\d+)\s+(.+?)\s+(Ls|\d{2}:\d{2}:\d{2})?\s*(\d{2}:\d{2}:\d{2})?\s*(.*)$"
)

for page_idx, page in enumerate(doc, start=1):
    text = page.get_text("text")
    
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        header_match = header_pattern.search(line)
        if header_match:
            current_train_no = header_match.group(1)
            current_train_name = header_match.group(2).strip()
            current_route = header_match.group(3).strip()
            continue

        match = row_pattern.match(line)
        if match and current_train_no:
            station_order = int(match.group(1))
            station_raw = match.group(2).strip()
            arrival = match.group(3)
            departure = match.group(4)
            note = match.group(5).strip() if match.group(5) else None

            code_match = re.search(r"\(([^)]+)\)", station_raw)
            station_code = code_match.group(1) if code_match else None
            station_name = re.sub(r"\s*\([^)]+\)", "", station_raw).strip()

            rows.append({
                "train_no": current_train_no,
                "train_name": current_train_name,
                "route": current_route,
                "station_order": station_order,
                "station_name": station_name,
                "station_code": station_code,
                "arrival_time": None if arrival == "Ls" else arrival,
                "departure_time": departure,
                "note": note,
                "source_page": page_idx
            })

df = pd.DataFrame(rows)
df = df.drop_duplicates()

records = df.to_dict("records")

batch_size = 500
for i in range(0, len(records), batch_size):
    supabase.table("train_schedules").insert(records[i:i+batch_size]).execute()

print(f"Inserted {len(records)} rows")