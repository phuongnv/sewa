import os
import datetime
import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================
VN_SYMBOLS = [
    "VNM.VN", "FPT.VN", "HPG.VN", "VCB.VN", "BID.VN",
    "CTG.VN", "SSI.VN", "MWG.VN", "GAS.VN", "VCI.VN", '^VNINDEX'
]


DAYS_BACK = 100
# ===================================================

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode="require"
    )

def get_data_yahoo(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()
    
    df = df.reset_index()
    df["symbol"] = symbol

    # Chỉ giữ đúng các cột cần thiết
    cols = ["symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[cols]
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # Loại bỏ NaN
    df = df.fillna(0)

    # print("Sample row:", df.head(2))
    # print("Length of row:", len(df.head(1)))

    # remove the first row
    df = df.iloc[1:]

    return df


def insert_data(conn, df):
    df = df[["symbol", "date", "open", "high", "low", "close", "volume"]]
    df = df.fillna(0)  # hoặc df.dropna()
    
    with conn.cursor() as cur:
        rows = [tuple(x) for x in df.to_numpy()]
        execute_values(cur, """
            INSERT INTO stock_prices (symbol, date, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, date)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP;
        """, rows)
    conn.commit()


def main():
    conn = get_connection()
    today = datetime.date.today()
    start = today - datetime.timedelta(days=DAYS_BACK)
    
    all_data = []
    for symbol in VN_SYMBOLS:
        print(f"⬇️  Fetching {symbol} ...")
        df = get_data_yahoo(symbol, start, today)
        print("Sample row:", df.head(1))
        if not df.empty:
            all_data.append(df)
        else:
            print(f"⚠️  No data for {symbol}")
    
    if all_data:
        df_all = pd.concat(all_data)
        print(f"✅ Downloaded {len(df_all)} records total.")
        insert_data(conn, df_all)
    else:
        print("❌ No data fetched.")
    conn.close()

if __name__ == "__main__":
    main()
