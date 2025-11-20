import requests
import streamlit as st
import pandas as pd
from datetime import datetime, date

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


def get_stock_symbols_for_today(conn):
    """Get list of (symbol, close_price) from stock_prices where date is today."""
    if not conn:
        return []

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return []

    try:
        today = date.today()
        query = """
            SELECT symbol, close
            FROM stock_prices
            WHERE date = %s
            ORDER BY symbol ASC;
        """
        df = pd.read_sql(query, conn, params=(today,))
        if df.empty:
            return []
        # Filter out rows with NaN or null close prices
        df = df.dropna(subset=["close"])
        if df.empty:
            return []
        return list(zip(df["symbol"].tolist(), df["close"].tolist()))
    except Exception as exc:
        st.error(f"Lỗi khi lấy danh sách mã từ stock_prices: {exc}")
        return []


def fetch_margin_from_ssi_api(symbol: str, price: float, bearer_token: str, account: str = "1862556"):
    """Call SSI API to get margin ratio for a stock symbol."""
    # Check for NaN or invalid price
    if pd.isna(price) or price <= 0:
        st.warning(f"Giá không hợp lệ cho {symbol}: {price}")
        return None
    
    url = "https://iboard-tapi.ssi.com.vn/trading/max-buy-sell"
    params = {
        "stockSymbol": symbol,
        "account": account,
        "price": int(price),
        "type": "A",
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "vi",
        "authorization": f"Bearer {bearer_token}",
        "device-id": "93EB98BF-6691-47FB-A86E-1B3133215792",
        "origin": "https://iboard.ssi.com.vn",
        "priority": "u=1, i",
        "referer": "https://iboard.ssi.com.vn/",
        "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") == "SUCCESS" and "data" in payload:
            margin_ratio_str = payload["data"].get("marginRatio")
            # Check if marginRatio exists and is not None
            if margin_ratio_str is None or not isinstance(margin_ratio_str, str):
                return None
            # Remove '%' and convert to float
            margin_value = float(margin_ratio_str.replace("%", "").strip())
            return margin_value
        else:
            return None
    except requests.RequestException as exc:
        # Silently return None - errors will be logged in the loop
        return None
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        # Silently return None - errors will be logged in the loop
        return None
    except Exception as exc:
        # Catch any other unexpected errors
        return None


def reset_margin_column(conn):
    """Reset all margin values to NULL in stock_info table."""
    if not conn:
        return False

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return False

    try:
        cur = conn.cursor()
        cur.execute("UPDATE stock_info SET margin = 0")
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        st.error(f"Lỗi khi reset cột margin: {exc}")
        return False


def update_margin_column(conn, symbol_margin_map: dict[str, float]):
    """Update margin column in stock_info table for provided symbols."""
    if not conn:
        return False

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return False

    try:
        cur = conn.cursor()
        now = datetime.now()

        for symbol, margin_value in symbol_margin_map.items():
            cur.execute(
                """
                INSERT INTO stock_info (symbol, margin, last_updated)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET margin = EXCLUDED.margin,
                    last_updated = EXCLUDED.last_updated;
                """,
                (symbol, margin_value, now),
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

    st.subheader("Section 2 — SSI Margin Ratio")
    st.markdown(
        """
        - Lấy danh sách mã cổ phiếu từ bảng `stock_prices` với `date` là ngày hiện tại.
        - Gọi SSI API để lấy `marginRatio` cho từng mã.
        - Cập nhật cột `margin` trong bảng `stock_info` với giá trị số (đã loại bỏ ký tự '%').
        """
    )

    bearer_token = st.text_input(
        "Bearer Token",
        type="password",
        help="Nhập Bearer token để xác thực với SSI API",
    )

    if st.button("Fetch & Update margin ratios", type="primary"):
        if not bearer_token:
            st.error("Vui lòng nhập Bearer token.")
            return

        with st.spinner("Đang lấy danh sách mã từ stock_prices..."):
            stock_data = get_stock_symbols_for_today(conn)

        if not stock_data:
            st.warning("Không có dữ liệu trong stock_prices cho ngày hôm nay.")
            return

        # Reset all margin values to NULL before starting
        with st.spinner("Đang reset tất cả giá trị margin về NULL..."):
            reset_success = reset_margin_column(conn)
            if not reset_success:
                st.error("Không thể reset cột margin. Vui lòng kiểm tra log.")
                return

        st.info(f"Tìm thấy {len(stock_data)} mã. Đang gọi API cho từng mã...")

        symbol_margin_map = {}
        failed_symbols = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (symbol, price) in enumerate(stock_data):
            try:
                status_text.text(f"Đang xử lý {symbol} ({idx + 1}/{len(stock_data)})...")
                margin_value = fetch_margin_from_ssi_api(symbol, price, bearer_token)

                if margin_value is not None:
                    symbol_margin_map[symbol] = margin_value
                else:
                    failed_symbols.append(symbol)
            except Exception as exc:
                # Catch any unexpected errors and continue with next symbol
                failed_symbols.append(symbol)
                # Optionally log the error (commented out to avoid cluttering UI)
                # st.warning(f"Lỗi không mong đợi khi xử lý {symbol}: {exc}")

            progress_bar.progress((idx + 1) / len(stock_data))

        status_text.empty()
        progress_bar.empty()

        if not symbol_margin_map:
            st.warning("Không có dữ liệu margin nào được lấy về từ API.")
            if failed_symbols:
                st.info(f"Không thể lấy margin cho {len(failed_symbols)} mã: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")
            return

        if failed_symbols:
            st.info(f"Đã bỏ qua {len(failed_symbols)} mã không có dữ liệu hoặc lỗi.")

        with st.spinner("Đang cập nhật bảng stock_info..."):
            success = update_margin_column(conn, symbol_margin_map)

        if success:
            st.success(f"Đã cập nhật margin cho {len(symbol_margin_map)} mã.")
            df_result = pd.DataFrame(
                {"symbol": list(symbol_margin_map.keys()), "margin": list(symbol_margin_map.values())}
            ).sort_values("symbol")
            st.dataframe(df_result.reset_index(drop=True))
        else:
            st.error("Không thể cập nhật dữ liệu. Vui lòng kiểm tra log.")
