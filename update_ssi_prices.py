import os
import psycopg2
import requests
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv
from psycopg2.extras import execute_values
import schedule
import time
import pytz
import page.filter_manager as filter_manager
import page.generate_charts as generate_charts
import db_connector

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

headers = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "vi",
    "device-id": "ADF1E947-BFD7-47BA-AA97-27531F3CC595",
    "origin": "https://iboard.ssi.com.vn",
    "referer": "https://iboard.ssi.com.vn/",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}


def get_db_connection():
    return psycopg2.connect(DB_URL)

# ===============================
# Fetch market index (VNINDEX, HNXIndex)
# ===============================


def fetch_index_data(symbol, has_history=True):
    # url = f"https://iboard-query.ssi.com.vn/exchange-index/{symbol}"
    # if has_history:
    #     url += "?hasHistory=true"

    HEADERS = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "vi",
        "device-id": "ADF1E947-BFD7-47BA-AA97-27531F3CC595",
        "origin": "https://iboard.ssi.com.vn",
        "referer": "https://iboard.ssi.com.vn/",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    }

    base_url = f"https://iboard-query.ssi.com.vn/exchange-index/{symbol}"
    if has_history:
        base_url += "?hasHistory=true"

    print(f"🔹 Fetching {symbol} data (history={has_history}) ...")

    try:
        resp = requests.get(base_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", {})
    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return []

    records = []

    # Nếu có history

    timestamp = data.get("time")
    if timestamp:
        date = datetime.fromtimestamp(timestamp / 1000)
        price = data.get("indexValue")
        volume = data.get("totalQtty", 0)
        # records.append((symbol, date, price, None, None, volume))

        # date = datetime.strptime(trading_date, "%Y%m%d")
        close_price = data.get("indexValue")
        open_price = data.get("chartOpen")
        high = data.get("chartHigh")
        low = data.get("chartLow")
        volume = data.get("totalQtty", 0)
        records.append((symbol, date, open_price, high, low,
                       close_price, volume, symbol.upper()))

    print(f"✅ Fetched {len(records)} records for {symbol}")
    return records

# ===============================
# Fetch stock prices for HOSE and HNX
# ===============================


def fetch_stock_data(exchange):
    url = f"https://iboard-query.ssi.com.vn/stock/exchange/{exchange}?boardId=MAIN"

    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json().get("data", [])

    records = []

    def div_thousand(value):
        if value is None:
            return None
        try:
            return value / 1000
        except Exception:
            return None
    for item in data:
        symbol = item.get("stockSymbol")
        trading_date = item.get("tradingDate")
        if not trading_date:
            # Skip malformed items without date
            continue
        date = datetime.strptime(trading_date, "%Y%m%d")
        close_price = div_thousand(item.get("matchedPrice"))
        open_price = div_thousand(item.get("openPrice"))
        high = div_thousand(item.get("highest"))
        low = div_thousand(item.get("lowest"))
        volume = item.get("nmTotalTradedQty")
        records.append((symbol, date, open_price, high, low,
                       close_price, volume, exchange.upper()))

    print(f"✅ Fetched {len(records)} stocks from {exchange.upper()}")
    return records


def fetch_stock_list(exchange: str):
    """Lấy danh sách mã cổ phiếu từ sàn HOSE hoặc HNX"""
    url = f"https://iboard.ssi.com.vn/dchart/api/stock/exchange/{exchange}"
    resp = requests.get(url, headers=headers, timeout=10)
    data = resp.json()
    if not data:
        print(f"❌ Không lấy được danh sách sàn {exchange}")
        return []
    symbols = [item["symbol"] for item in data if "symbol" in item]
    print(f"✅ Fetched {len(symbols)} stocks from {exchange}")
    return symbols


def fetch_stock_history(symbol: str, days: int = 100):
    """Lấy lịch sử giá 100 ngày gần nhất cho 1 mã cổ phiếu"""
    end_ts = int(datetime.now().timestamp())
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp())

    url = f"https://iboard-api.ssi.com.vn/statistics/charts/history?resolution=1D&symbol={symbol}&from={start_ts}&to={end_ts}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
    except Exception as e:
        print(f"⚠️ {symbol}: lỗi khi gọi API ({e})")
        return []

    if data.get("code") != "SUCCESS":
        print(f"⚠️ {symbol}: API trả về lỗi {data.get('message')}")
        return []

    hist = data.get("data", {})
    if not hist or "t" not in hist:
        return []

    records = []
    for i in range(len(hist["t"])):
        ts = hist["t"][i]
        date = datetime.fromtimestamp(ts)
        o = hist["o"][i]
        h = hist["h"][i]
        l = hist["l"][i]
        c = hist["c"][i]
        v = hist["v"][i]
        records.append((symbol, date, o, h, l, c, v))
    return records


def save_stock_history(conn, records, exchange):
    """Lưu dữ liệu lịch sử vào bảng stock_prices"""
    if not records:
        return

    # Deduplicate records based on (symbol, date) to avoid ON CONFLICT issues
    unique_records = {}
    for r in records:
        key = (r[0], r[1])  # symbol, date
        unique_records[key] = r

    records = list(unique_records.values())

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO stock_prices (symbol, date, open, high, low, close, volume, exchange)
            VALUES %s
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                exchange = EXCLUDED.exchange;
        """, [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], exchange) for r in records])
    conn.commit()


def update_exchange(conn, exchange: str, days: int = 100):
    """Cập nhật lịch sử giá cho toàn bộ mã cổ phiếu trong 1 sàn"""
    symbols = fetch_stock_data(exchange)
    for symbol in [s[0] for s in symbols]:
        time.sleep(0.2)  # tránh bị chặn API
        records = fetch_stock_history(symbol, days)
        if records:
            save_stock_history(conn, records, exchange)
            print(f"✅ {symbol} ({exchange}): {len(records)} ngày")
        else:
            print(f"⚠️ {symbol} ({exchange}): không có dữ liệu")

    symbol = "VNINDEX"
    records = fetch_stock_history(symbol, days)
    if records:
        save_stock_history(conn, records, exchange)
        print(f"✅ {symbol} ({exchange}): {len(records)} ngày")
    else:
        print(f"⚠️ {symbol} ({exchange}): không có dữ liệu")


def update_all(conn, days: int = 100):
    """Quét cả HOSE và HNX"""
    for ex in ["hose", "hnx"]:
        print(f"\n=== 🔄 Bắt đầu cập nhật {ex} ===")
        update_exchange(conn, ex, days)
    print("🎯 Hoàn tất cập nhật toàn bộ!")

# ===============================
# Save to DB
# ===============================


def save_rrg_data(conn, records, table="rrg_index_data"):
    """
    Lưu danh sách records vào bảng rrg_index_data.
    Mỗi record có dạng:
        (symbol, date, price, price_change, price_change_percent, volume)
    """

    if not records:
        print("⚠️ No records to save.")
        return

    with conn.cursor() as cur:
        # Loại bỏ trùng (symbol, date) theo ngày (DB thường dùng DATE, không phải TIMESTAMP)
        normalized_records = []
        for r in records:
            symbol = r[0]
            dt = r[1]
            date_only = dt.date() if isinstance(dt, datetime) else dt
            normalized_records.append(
                (symbol, date_only, r[2], r[3], r[4], r[5]))

        unique_records = {(r[0], r[1]): r for r in normalized_records}
        records = list(unique_records.values())

        # Chèn hoặc cập nhật dữ liệu
        insert_query = f"""
            INSERT INTO {table} 
                (symbol, date, price, price_change, price_change_percent, volume)
            VALUES %s
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                price = EXCLUDED.price,
                price_change = EXCLUDED.price_change,
                price_change_percent = EXCLUDED.price_change_percent,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP;
        """

        execute_values(cur, insert_query, records)
        conn.commit()

    print(f"✅ Saved {len(records)} records into {table}")


def save_stock_data(conn, records, table="stock_prices"):
    if not records:
        print("⚠️ No stock records to save.")
        return
    with conn.cursor() as cur:
        # Chuẩn hóa ngày (DATE) để tránh trùng trong cùng lệnh INSERT ... ON CONFLICT
        normalized_records = []
        for r in records:
            symbol = r[0]
            dt = r[1]
            date_only = dt.date() if isinstance(dt, datetime) else dt
            # (symbol, date, open, high, low, close, volume, exchange)
            normalized_records.append(
                (symbol, date_only, r[2], r[3], r[4], r[5], r[6], r[7]))

        unique_records = {(r[0], r[1]): r for r in normalized_records}
        records = list(unique_records.values())

        execute_values(cur, f"""
            INSERT INTO {table} (symbol, date, open, high, low, close, volume, exchange)
            VALUES %s
            ON CONFLICT (symbol, date)
            DO UPDATE SET close = EXCLUDED.close,
                          open = EXCLUDED.open,
                          high = EXCLUDED.high,
                          low = EXCLUDED.low,
                          volume = EXCLUDED.volume,
                          exchange = EXCLUDED.exchange
        """, records)
        conn.commit()
    print(f"✅ Saved {len(records)} stock records into {table}")

# ===============================
# Update modes
# ===============================


def update_history():
    conn = get_db_connection()
    indexes = ["VNINDEX", "HNXIndex"]
    # for idx in indexes:
    #     records = fetch_index_data(idx, has_history=True)
    #     save_rrg_data(conn, records)
    # for exch in ["hose", "hnx"]:
    #     stock_records = fetch_stock_data(exch)
    #     save_stock_data(conn, stock_records)
    update_all(conn, days=700)
    conn.close()


def update_latest():
    print("🔹 Updating latest market & stock data ...")
    conn = get_db_connection()
    indexes = ["VNINDEX", "HNXIndex"]
    for idx in indexes:
        records = fetch_index_data(idx, has_history=False)

        save_stock_data(conn, records)
        # save_rrg_data(conn, records)
    for exch in ["hose", "hnx"]:
        stock_records = fetch_stock_data(exch)
        print("Sample record:", stock_records[0])
        print("Record length:", len(stock_records[0]))
        save_stock_data(conn, stock_records)
    conn.close()

def calculate_and_save_rrg_data(conn):
    """
    Fetch symbols from stock_prices, calculate RRG data, and save to the database.
    """
    print("🔹 Starting RRG data calculation and saving...")

    # Fetch symbols for today
    today = date.today()
    start_date = today - timedelta(days=365)  # Get 1 year of data
    end_date = today

    try:
        # Fetch symbols with margin for today
        stock_data = filter_manager.get_stock_symbolsge_with_margin_for_today(conn)
        if not stock_data:
            print("⚠️ No stock data found for today.")
            return

        symbols_to_process = [symbol for symbol, _ in stock_data]
        print(f"🔹 Found {len(symbols_to_process)} symbols to process.")

        success_count = 0
        failed_symbols = []

        for idx, symbol in enumerate(symbols_to_process):
            try:
                print(f"🔄 Processing symbol {symbol} ({idx + 1}/{len(symbols_to_process)})...")

                # Fetch price data for symbol and benchmark
                price_df = db_connector.fetch_price_data(
                    conn, [symbol, generate_charts.BENCHMARK_SYMBOL], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                )

                if price_df.empty:
                    failed_symbols.append(f"{symbol} (no price data)")
                    continue

                # Calculate RRG data
                rrg_df = generate_charts.calculate_rrg_data(price_df, generate_charts.BENCHMARK_SYMBOL, generate_charts.RRG_PERIOD, generate_charts.SCALE_FACTOR)

                if rrg_df.empty:
                    failed_symbols.append(f"{symbol} (RRG calculation failed)")
                    continue

                # Save to database
                save_success = filter_manager.save_rrg_data_to_db(conn, rrg_df)
                if save_success:
                    success_count += 1
                else:
                    failed_symbols.append(f"{symbol} (failed to save)")

            except Exception as exc:
                failed_symbols.append(f"{symbol} (error: {str(exc)[:50]})")

        print(f"✅ Successfully processed {success_count} symbols.")
        if failed_symbols:
            print(f"⚠️ Failed to process {len(failed_symbols)} symbols: {', '.join(failed_symbols[:10])}...")

    except Exception as exc:
        print(f"❌ Error during RRG calculation: {exc}")

def update_latest_and_calculate_rrg():
    """
    Update the latest market & stock data, then calculate and save RRG data.
    """
    vn_tz = timezone(timedelta(hours=7))
    print("🔹 Updating latest market & stock data...", datetime.now(tz=vn_tz))
    update_latest()

    print("🔹 Calculating and saving RRG data...")
    conn = get_db_connection()
    calculate_and_save_rrg_data(conn)
    conn.close()
# ===============================
# Main
# ===============================
def launch_daily_scheduler(update_latest_and_calculate_rrg):
    VN_TZ = pytz.timezone("Asia/Ho_Chi_Minh")
    print("🕒 Auto mode started (market-hours every 30 minutes from 09:30 to 15:00 VN time)")

        # Schedule `update_latest` every 30 minutes from 09:30 to 15:00
    start_time = datetime.strptime("09:30", "%H:%M")
    end_time = datetime.strptime("16:00", "%H:%M")
    exclusive_start_time = datetime.strptime("11:33", "%H:%M")
    exclusive_end_time = datetime.strptime("13:31", "%H:%M")
    current_time = start_time
    print("Scheduling updates between", start_time.strftime("%H:%M"),
              "and", end_time.strftime("%H:%M"))
    scheduled_times = []

    while current_time <= end_time:
        if exclusive_start_time <= current_time < exclusive_end_time:
            current_time += timedelta(minutes=120)
            continue
        vn_time = VN_TZ.localize(current_time)
        utc_time = vn_time.astimezone(pytz.utc)
        time_str = utc_time.strftime("%H:%M")
        schedule.every().day.at(time_str).do(update_latest_and_calculate_rrg)
        scheduled_times.append(f"{current_time.strftime('%H:%M')} (VN time)")
        current_time += timedelta(minutes=120)

    print("Scheduled update_latest at:", ", ".join(scheduled_times))

        # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Update SSI market & stock data")
    parser.add_argument(
        "--mode", choices=["history", "latest", "auto"], default="latest")
    args = parser.parse_args()

    if args.mode == "history":
        print("🔹 Updating historical data ...")
        update_history()
    elif args.mode == "latest":
        vn_tz = timezone(timedelta(hours=7))
        print("🔹 Updating latest data ... at ", datetime.now(tz=vn_tz))
        update_latest()
    elif args.mode == "auto":
        launch_daily_scheduler(update_latest_and_calculate_rrg)
