import os
import time
import psycopg2
import requests
import schedule
from datetime import datetime
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
DB_URL = os.getenv("DATABASE_URL")
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "vi",
    "device-id": "ADF1E947-BFD7-47BA-AA97-27531F3CC595",
    "origin": "https://iboard.ssi.com.vn",
    "referer": "https://iboard.ssi.com.vn/",
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}
INDEXES = ["VNINDEX", "HNXIndex"]

# ---------------- DB ----------------
def get_db_connection():
    return psycopg2.connect(DB_URL)

def save_rrg_data(conn, records):
    if not records:
        print("⚠️  No records to save.")
        return
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO rrg_index_data (symbol, date, value)
            VALUES %s
            ON CONFLICT (symbol, date) DO UPDATE
            SET value = EXCLUDED.value
        """, records)
    conn.commit()
    print(f"✅ Saved {len(records)} records.")


# ---------------- FETCH FUNCTIONS ----------------
def fetch_index_data(symbol):
    """Fetch full history data"""
    url = f"https://iboard-query.ssi.com.vn/exchange-index/{symbol}?hasHistory=true"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    if not data or "history" not in data:
        print(f"⚠️ No history for {symbol}")
        return []
    records = []
    for h in data["history"]:
        t = datetime.fromtimestamp(h["time"] / 1000.0)
        v = h.get("indexValue")
        if v:
            records.append((symbol, t, v))
    print(f"✅ Fetched {len(records)} history records for {symbol}")
    return records


def fetch_daily_index(symbol):
    """Fetch latest single snapshot"""
    url = f"https://iboard-query.ssi.com.vn/exchange-index/{symbol}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data", {})
    if not data or "indexValue" not in data:
        print(f"⚠️ No latest data for {symbol}")
        return []
    t = datetime.fromtimestamp(data["time"] / 1000.0)
    v = data["indexValue"]
    print(f"✅ Latest {symbol} = {v} at {t}")
    return [(symbol, t, v)]


# ---------------- MAIN LOGIC ----------------
def update_history():
    conn = get_db_connection()
    for s in INDEXES:
        print(f"🔹 Fetching {s} historical data ...")
        data = fetch_index_data(s)
        save_rrg_data(conn, data)
    conn.close()

def update_latest():
    conn = get_db_connection()
    for s in INDEXES:
        print(f"🔹 Fetching {s} latest data ...")
        data = fetch_daily_index(s)
        save_rrg_data(conn, data)
    conn.close()

def update_both():
    update_history()
    update_latest()


# ---------------- SCHEDULER ----------------
def schedule_daily_update():
    """Auto run every day at 17:00 VN time"""
    schedule.every().day.at("17:00").do(update_latest)
    print("⏰ Scheduler started. Running every day at 17:00 VN time.")
    while True:
        schedule.run_pending()
        time.sleep(60)


# ---------------- CLI ENTRY ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Update SSI index data to Neon DB")
    parser.add_argument("--mode", choices=["history", "latest", "both", "auto"], default="latest",
                        help="Choose mode: 'history' for all data, 'latest' for today, 'both' for both, 'auto' for daily scheduler")
    args = parser.parse_args()

    if args.mode == "history":
        update_history()
    elif args.mode == "latest":
        update_latest()
    elif args.mode == "both":
        update_both()
    elif args.mode == "auto":
        schedule_daily_update()
