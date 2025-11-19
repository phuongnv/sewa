import requests
import streamlit as st
import pandas as pd
from datetime import datetime

import db_connector

API_URL = "https://scanner.tradingview.com/vietnam/scan?label-product=screener-stock"
API_HEADERS = {"Content-Type": "application/json"}
API_PAYLOAD = {
    "columns": [
        "name",
        "description",
        "logoid",
        "update_mode",
        "type",
        "typespecs",
        "TechRating_1D",
        "TechRating_1D.tr",
        "MARating_1D",
        "MARating_1D.tr",
        "OsRating_1D",
        "OsRating_1D.tr",
        "RSI",
        "Mom",
        "pricescale",
        "minmov",
        "fractional",
        "minmove2",
        "AO",
        "CCI20",
        "Stoch.K",
        "Stoch.D",
        "Candle.3BlackCrows",
        "Candle.3WhiteSoldiers",
        "Candle.AbandonedBaby.Bearish",
        "Candle.AbandonedBaby.Bullish",
        "Candle.Doji",
        "Candle.Doji.Dragonfly",
        "Candle.Doji.Gravestone",
        "Candle.Engulfing.Bearish",
        "Candle.Engulfing.Bullish",
        "Candle.EveningStar",
        "Candle.Hammer",
        "Candle.HangingMan",
        "Candle.Harami.Bearish",
        "Candle.Harami.Bullish",
        "Candle.InvertedHammer",
        "Candle.Kicking.Bearish",
        "Candle.Kicking.Bullish",
        "Candle.LongShadow.Lower",
        "Candle.LongShadow.Upper",
        "Candle.Marubozu.Black",
        "Candle.Marubozu.White",
        "Candle.MorningStar",
        "Candle.ShootingStar",
        "Candle.SpinningTop.Black",
        "Candle.SpinningTop.White",
        "Candle.TriStar.Bearish",
        "Candle.TriStar.Bullish",
        "exchange",
    ],
    "filter": [
        {"left": "AnalystRating", "operation": "in_range", "right": ["StrongBuy", "Buy"]},
        {"left": "average_volume_10d_calc", "operation": "greater", "right": 1_000_000},
        {"left": "is_primary", "operation": "equal", "right": True},
    ],
    "ignore_unknown_fields": False,
    "options": {"lang": "vi"},
    "range": [0, 100],
    "sort": {"sortBy": "name", "sortOrder": "asc"},
    "symbols": {},
    "markets": ["vietnam"],
    "filter2": {
        "operator": "and",
        "operands": [
            {
                "operation": {
                    "operator": "or",
                    "operands": [
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                                    {"expression": {"left": "typespecs", "operation": "has", "right": ["common"]}},
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {"expression": {"left": "type", "operation": "equal", "right": "stock"}},
                                    {"expression": {"left": "typespecs", "operation": "has", "right": ["preferred"]}},
                                ],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [{"expression": {"left": "type", "operation": "equal", "right": "dr"}}],
                            }
                        },
                        {
                            "operation": {
                                "operator": "and",
                                "operands": [
                                    {"expression": {"left": "type", "operation": "equal", "right": "fund"}},
                                    {
                                        "expression": {
                                            "left": "typespecs",
                                            "operation": "has_none_of",
                                            "right": ["etf"],
                                        }
                                    },
                                ],
                            }
                        },
                    ],
                }
            },
            {"expression": {"left": "typespecs", "operation": "has_none_of", "right": ["pre-ipo"]}},
        ],
    },
}


def fetch_tradingview_symbols():
    """Call TradingView API and return list of ticker symbols (data.d[0])."""
    try:
        response = requests.post(API_URL, headers=API_HEADERS, json=API_PAYLOAD, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.error(f"Lỗi khi gọi TradingView API: {exc}")
        return []

    symbols = []
    for item in payload.get("data", []):
        data_fields = item.get("d", [])
        if data_fields:
            symbols.append(data_fields[0])
    return symbols


def update_recommend_column(conn, symbols: list[str]):
    """Reset recommend to 0 for all rows, then set to 1 for provided symbols."""
    if not conn:
        return False

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return False

    try:
        cur = conn.cursor()
        cur.execute("UPDATE stock_info SET recommend = '0'")

        now = datetime.now()
        for sym in symbols:
            cur.execute(
                """
                INSERT INTO stock_info (symbol, recommend, last_updated)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET recommend = EXCLUDED.recommend,
                    last_updated = EXCLUDED.last_updated;
                """,
                (sym, "1", now),
            )

        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        st.error(f"Lỗi khi cập nhật bảng stock_info: {exc}")
        return False


def render(conn):
    st.header("Filter manager")
    if not conn:
        st.error("Không thể kết nối cơ sở dữ liệu.")
        return

    st.write("Quản lý các bộ lọc và đồng bộ dữ liệu vào bảng `stock_info`.")

    st.subheader("Section 1 — TradingView Strong Buy / Buy Screener")
    st.markdown(
        """
        - Lấy danh sách cổ phiếu từ TradingView API (sàn Việt Nam) thỏa điều kiện AnalystRating ∈ {StrongBuy, Buy}.
        - Tự động đặt `recommend = 1` cho các mã xuất hiện trong API, và các mã còn lại được reset về `0`.
        """
    )

    if st.button("Fetch & Update recommendations", type="primary"):
        with st.spinner("Đang gọi TradingView API..."):
            symbols = fetch_tradingview_symbols()

        if not symbols:
            st.warning("Không có dữ liệu trả về từ API.")
            return

        with st.spinner("Đang cập nhật bảng stock_info..."):
            success = update_recommend_column(conn, symbols)

        if success:
            st.success(f"Đã cập nhật recommend = 1 cho {len(symbols)} mã.")
            st.dataframe(pd.DataFrame({"symbol": symbols}).sort_values("symbol").reset_index(drop=True))
        else:
            st.error("Không thể cập nhật dữ liệu. Vui lòng kiểm tra log.")
