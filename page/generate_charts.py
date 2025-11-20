import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import db_connector

BENCHMARK_SYMBOL = "VNINDEX"
RRG_PERIOD = 50
DAYS_FOR_CHART = 30
SCALE_FACTOR = 4.0
CUSTOM_SYMBOL_HISTORY_DAYS = 365

RRG_FILTER_OPTIONS = {
    "none": "Không áp dụng",
    "rs_gt_100": "RS-Ratio > 100 (giá trị mới nhất)",
    "avg_up": "Trung bình 3 ngày gần nhất tăng so với 3 ngày trước",
}



def normalize_data(data: pd.Series) -> pd.Series:
    """Chuẩn hóa Z-Score cho một Series."""
    if data.std() == 0:
        return pd.Series(0, index=data.index)
    return (data - data.mean()) / data.std()

def wma_func(x: pd.Series, period: int) -> float:
    """Hàm tính Weighted Moving Average (WMA) cho cửa sổ lăn."""
    weights = np.arange(1, len(x) + 1)
    weights = weights[len(weights) - period:] if len(weights) > period else weights
    
    if len(x) < len(weights):
        return np.nan

    return np.sum(x.values[-len(weights):] * weights) / np.sum(weights)

# @st.cache_data
def calculate_rrg_data(df: pd.DataFrame, benchmark_symbol: str, period: int, scale_factor: float) -> pd.DataFrame:
    """
    Tính toán chỉ số RRG (RS-Ratio và RS-Momentum) bằng WMA, 
    Chuẩn hóa Z-Score, và Dịch chuyển về tâm 100.
    """
    df = df.copy()
    if df.empty: return pd.DataFrame()

    close_prices = df.pivot(index='date', columns='symbol', values='close')
    if benchmark_symbol not in close_prices.columns: return df
    benchmark = close_prices[benchmark_symbol]
    
    # --- A. Tính toán RS và RM (Chưa chuẩn hóa) ---
    rs_line = close_prices.div(benchmark, axis=0)
    
    # Tính WMA của RS Line
    wma_rs_line = rs_line.rolling(window=period, min_periods=period).apply(lambda x: wma_func(x, period), raw=False)
    rs_ratio_wide = (rs_line / wma_rs_line) * 100

    # Tính WMA của RS-Ratio
    wma_rs_ratio = rs_ratio_wide.rolling(window=period, min_periods=period).apply(lambda x: wma_func(x, period), raw=False)
    rs_momentum_wide = (rs_ratio_wide / wma_rs_ratio) * 100

    # 2. Chuyển kết quả về dạng dài (Long format)
    rrg_results = []
    for symbol in rs_ratio_wide.columns:
        if symbol == benchmark_symbol:
            continue
        temp_df = pd.DataFrame(
            {
                "date": rs_ratio_wide.index,
                "symbol": symbol,
                "rs_ratio": rs_ratio_wide[symbol].values,
                "rs_momentum": rs_momentum_wide[symbol].values,
            }
        )
        rrg_results.append(temp_df)

    if not rrg_results:
        return pd.DataFrame()

    rrg_results_long = pd.concat(rrg_results, ignore_index=True)

    # 3. CHUẨN HÓA VÀ DỊCH CHUYỂN TÂM 100
    rrg_results_long['rs_ratio_z'] = normalize_data(rrg_results_long['rs_ratio'])
    rrg_results_long['rs_momentum_z'] = normalize_data(rrg_results_long['rs_momentum'])
    
    rrg_results_long['rs_ratio_scaled'] = 100 + rrg_results_long['rs_ratio_z'] * scale_factor
    rrg_results_long['rs_momentum_scaled'] = 100 + rrg_results_long['rs_momentum_z'] * scale_factor
    
    # 4. Merge kết quả với DataFrame gốc
    rrg_results_long = rrg_results_long.reset_index(drop=True) 

    df = df.merge(
        rrg_results_long[['date', 'symbol', 'rs_ratio_scaled', 'rs_momentum_scaled']], 
        on=['date', 'symbol'], 
        how='left'
    )
    
    # 5. Loại bỏ các dòng có giá trị NaN
    df = df.dropna(subset=['rs_ratio_scaled', 'rs_momentum_scaled'])
    
    return df

# =====================
# RRG Chart Plotting
# =====================

def fetch_rrg_data_from_db(conn, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch RRG data from rrg_data table."""
    if not conn or not symbols:
        return pd.DataFrame()

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return pd.DataFrame()

    try:
        symbol_list = tuple(symbols)
        if len(symbols) == 1:
            symbol_list = (symbols[0],)

        query = """
            SELECT symbol, date, rs_ratio_scaled, rs_momentum_scaled, close
            FROM rrg_data
            WHERE symbol IN %s AND date BETWEEN %s AND %s
            ORDER BY date, symbol;
        """

        df = pd.read_sql(query, conn, params=(symbol_list, start_date, end_date))

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        # Rename columns to match expected format
        df = df.rename(columns={
            'rs_ratio_scaled': 'rs_ratio_scaled',
            'rs_momentum_scaled': 'rs_momentum_scaled'
        })
        return df

    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu RRG từ DB: {e}")
        return pd.DataFrame()


def filter_symbols_by_rrg_condition(conn, symbols: list[str], end_date, condition: str) -> list[str]:
    """Filter symbols based on stored RRG data conditions."""
    if condition == "none" or not symbols:
        return symbols

    conn = db_connector.ensure_connection(conn)
    if not conn:
        return symbols

    try:
        symbol_array = symbols

        if condition == "rs_gt_100":
            query = """
                WITH latest AS (
                    SELECT symbol, rs_ratio_scaled, date,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
                    FROM rrg_data
                    WHERE symbol = ANY(%s) AND date <= %s
                )
                SELECT symbol
                FROM latest
                WHERE rn = 1 AND rs_ratio_scaled > 100;
            """
            df = pd.read_sql(query, conn, params=(symbol_array, end_date))
            if df.empty:
                return []
            return df["symbol"].tolist()

        if condition == "avg_up":
            query = """
                WITH ranked AS (
                    SELECT symbol, rs_ratio_scaled, date,
                           ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
                    FROM rrg_data
                    WHERE symbol = ANY(%s) AND date <= %s
                )
                SELECT symbol, rs_ratio_scaled, rn
                FROM ranked
                WHERE rn <= 6
                ORDER BY symbol, rn;
            """
            df = pd.read_sql(query, conn, params=(symbol_array, end_date))
            if df.empty:
                return []

            valid_symbols = []
            for symbol, group in df.groupby("symbol"):
                recent = group[group["rn"] <= 3]["rs_ratio_scaled"]
                previous = group[(group["rn"] > 3) & (group["rn"] <= 6)]["rs_ratio_scaled"]
                if len(recent) == 3 and len(previous) == 3:
                    if recent.mean() > previous.mean():
                        valid_symbols.append(symbol)
            return valid_symbols

        return symbols

    except Exception as e:
        st.error(f"Lỗi khi áp dụng bộ lọc RRG: {e}")
        return symbols


def plot_rrg_time_series(rrg_df: pd.DataFrame, symbol: str, benchmark: str, period: int):
    """Vẽ biểu đồ RRG Time Series (Tâm 100)."""
    if rrg_df.empty:
        st.info("Không có dữ liệu RRG để vẽ biểu đồ.")
        return

    rs = rrg_df[rrg_df['symbol'] == symbol]['rs_ratio_scaled']
    rm = rrg_df[rrg_df['symbol'] == symbol]['rs_momentum_scaled']

    if rs.empty:
        st.warning(f"Không tìm thấy dữ liệu RRG đã tính toán cho mã {symbol}.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Định nghĩa màu sắc 4 góc phần tư
    quadrant_colors = {'Leading': 'green', 'Weakening': '#ffc000', 'Lagging': 'red', 'Improving': 'blue'}

    # Vẽ các đường ngang và dọc chuẩn (TÂM 100)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)

    # Đặt giới hạn trục X và Y
    rm_min_val = min( rm.min(), 98)
    rm_max_val = max( rm.max(), 102)
    rs_min_val = min(rs.min(), 98)
    rs_max_val = max(rs.max(), 102)
    rm_padding = (rm_max_val - rm_min_val) * 0.1
    rs_padding = (rs_max_val - rs_min_val) * 0.1
    ax.set_xlim(rs_min_val - rs_padding, rs_max_val + rs_padding)
    ax.set_ylim(rm_min_val - rm_padding, rm_max_val + rm_padding)
    
    # Xác định quadrant
    quadrants = pd.Series(index=rs.index, dtype=str)
    quadrants[(rs >= 100) & (rm >= 100)] = 'Leading'
    quadrants[(rs >= 100) & (rm < 100)] = 'Weakening'
    quadrants[(rs < 100) & (rm < 100)] = 'Lagging'
    quadrants[(rs < 100) & (rm >= 100)] = 'Improving'

    # Vẽ đường RRG Time Series
    for i in range(1, len(rs)):
        current_quadrant = quadrants.iloc[i]
        color = quadrant_colors.get(current_quadrant, 'black')
        ax.plot(
            [rs.iloc[i-1], rs.iloc[i]],
            [rm.iloc[i-1], rm.iloc[i]],
            color=color,
            linewidth=2,
            alpha=0.7,
            zorder=3
        )

    # Điểm cuối cùng (Hiện tại)
    ax.scatter(rs.iloc[-1], rm.iloc[-1], color='black', s=150, zorder=5) 
    ax.text(rs.iloc[-1], rm.iloc[-1], symbol, fontsize=12, ha='right', va='bottom', zorder=6) 

    # Điểm đầu tiên
    ax.scatter(rs.iloc[0], rm.iloc[0], color='gray', s=50, marker='o', zorder=5)

    # Thêm nhãn góc phần tư
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95, 'Leading', fontsize=12, color='green', ha='right', va='top')
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] * 1.05, 'Weakening', fontsize=12, color='red', ha='right', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[0] * 1.05, 'Lagging', fontsize=12, color='blue', ha='left', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[1] * 0.95, 'Improving', fontsize=12, color='#ffc000', ha='left', va='top')

    # Get last price for the current symbol
    symbol_df = rrg_df[rrg_df['symbol'] == symbol].sort_values('date')
    if not symbol_df.empty and 'close' in symbol_df.columns:
        last_price = symbol_df.iloc[-1]['close']
    else:
        last_price = None

    ax.set_title(f'RRG Time Series Chart: {symbol} vs {benchmark} (last Price: {last_price})', fontsize=14)
    ax.set_xlabel('Relative Strength (RS Ratio)')
    ax.set_ylabel('Relative Momentum (RM Momentum)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') 

    st.pyplot(fig)






def render(conn):
    st.header("Generate charts")
    st.write("Tự động tạo biểu đồ RRG cho các mã có cờ `margin = 1` trong bảng `stock_info`.")

    if not conn:
        st.error("Không thể kết nối cơ sở dữ liệu. Vui lòng kiểm tra cấu hình.")
        return

    today = datetime.now().date()
    default_start = today - timedelta(days=DAYS_FOR_CHART)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Từ ngày",
            value=default_start,
            max_value=today - timedelta(days=1),
            help="Khoảng thời gian càng dài sẽ cho đường RRG mượt hơn.",
        )
    with col2:
        end_date = st.date_input(
            "Đến ngày",
            value=today,
            max_value=today,
        )

    st.subheader("Symbol filters")
    
    # Add symbol text input filter
    symbol_filter_input = st.text_input(
        "Danh sách mã cần vẽ biểu đồ (tùy chọn)",
        value="",
        help="Nhập các mã phân cách bằng dấu phẩy (ví dụ: ACB,VCB,TCB). Để trống để sử dụng bộ lọc margin/recommend.",
    )
    
    use_custom_symbols = bool(symbol_filter_input and symbol_filter_input.strip())
    
    if not use_custom_symbols:
        filter_margin = st.checkbox("Filter theo Margin (>= giá trị)", value=True)
        if filter_margin:
            margin_threshold = st.number_input(
                "Giá trị margin tối thiểu",
                min_value=0,
                max_value=100,
                value=40,
                step=1,
                help="Chỉ lấy các mã có margin lớn hơn hoặc bằng giá trị này.",
            )
        else:
            margin_threshold = 0

        filter_recommend = st.checkbox("Filter theo Recommend", value=False)
        if filter_recommend:
            recommend_value = st.selectbox(
                "Giá trị recommend cần lấy",
                options=["1", "0"],
                index=0,
                help="Chỉ lấy các mã có recommend đúng bằng giá trị này.",
            )
        else:
            recommend_value = "1"

        rrg_filter_option = st.selectbox(
            "Bộ lọc bổ sung theo dữ liệu RRG (từ DB)",
            options=list(RRG_FILTER_OPTIONS.keys()),
            index=0,
            format_func=lambda key: RRG_FILTER_OPTIONS[key],
            help="Áp dụng thêm điều kiện dựa trên dữ liệu đã lưu trong bảng rrg_data.",
        )

        margin_symbols = db_connector.fetch_symbols_by_filters(
            conn,
            use_margin=filter_margin,
            margin_value=margin_threshold,
            use_recommend=filter_recommend,
            recommend_value=recommend_value,
        )

        if margin_symbols and rrg_filter_option != "none":
            filtered_symbols = filter_symbols_by_rrg_condition(conn, margin_symbols, end_date, rrg_filter_option)
            removed = len(margin_symbols) - len(filtered_symbols)
            margin_symbols = filtered_symbols
            if removed > 0:
                st.info(f"Đã loại {removed} mã không đạt bộ lọc RRG ({RRG_FILTER_OPTIONS[rrg_filter_option]}).")
    else:
        rrg_filter_option = "none"
        # Parse custom symbols
        margin_symbols = [s.strip().upper() for s in symbol_filter_input.split(",") if s.strip()]
        filter_margin = False
        filter_recommend = False

    if margin_symbols:
        active_filters = []
        if filter_margin:
            active_filters.append(f"margin ≥ {margin_threshold}")
        if filter_recommend:
            active_filters.append(f"recommend = {recommend_value}")
        if rrg_filter_option != "none":
            active_filters.append(RRG_FILTER_OPTIONS[rrg_filter_option])
        filter_text = ", ".join(active_filters) if active_filters else "Không áp dụng bộ lọc"
        st.info(f"Đang áp dụng bộ lọc: {filter_text}. Tìm thấy **{len(margin_symbols)}** mã.")
    else:
        st.warning("Không tìm thấy mã nào theo bộ lọc đã chọn.")

    charts_per_line = st.number_input(
        "Số biểu đồ mỗi dòng",
        min_value=1,
        max_value=6,
        value=5,
        help="Chọn số lượng biểu đồ hiển thị trên mỗi dòng (1-6)."
    )

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        return

    if st.button("Generate charts", type="primary"):
        if not margin_symbols:
            st.info("Không có mã nào để tạo biểu đồ. Vui lòng cập nhật cờ margin trước.")
            return

        str_start = start_date.strftime("%Y-%m-%d")
        str_end = end_date.strftime("%Y-%m-%d")

        # Group symbols into chunks for displaying charts per row
        def chunk_list(lst, chunk_size):
            """Split list into chunks of specified size."""
            for i in range(0, len(lst), chunk_size):
                yield lst[i:i + chunk_size]

        # Process all symbols and prepare data
        symbol_data = {}
        with st.spinner("Đang tải dữ liệu RRG..."):
            if use_custom_symbols:
                # Always recalculate using fresh 365-day price data
                calc_start_date = end_date - timedelta(days=CUSTOM_SYMBOL_HISTORY_DAYS)
                str_calc_start = calc_start_date.strftime("%Y-%m-%d")
                for symbol in margin_symbols:
                    price_df = db_connector.fetch_price_data(conn, [symbol, BENCHMARK_SYMBOL], str_calc_start, str_end)
                    if price_df.empty:
                        symbol_data[symbol] = None
                        continue

                    rrg_df = calculate_rrg_data(price_df, BENCHMARK_SYMBOL, RRG_PERIOD, SCALE_FACTOR)
                    if rrg_df.empty:
                        symbol_data[symbol] = None
                        continue

                    rrg_df["date"] = pd.to_datetime(rrg_df["date"])
                    rrg_filtered = rrg_df[
                        (rrg_df["date"] >= pd.to_datetime(str_start)) & (rrg_df["date"] <= pd.to_datetime(str_end))
                    ]
                    if rrg_filtered.empty:
                        symbol_data[symbol] = None
                    else:
                        symbol_data[symbol] = rrg_filtered
            else:
                # Fetch from DB, fallback to calculation if missing
                rrg_df_db = fetch_rrg_data_from_db(conn, margin_symbols, str_start, str_end)
                
                if not rrg_df_db.empty:
                    for symbol in margin_symbols:
                        symbol_rrg = rrg_df_db[rrg_df_db['symbol'] == symbol]
                        if not symbol_rrg.empty:
                            symbol_data[symbol] = symbol_rrg
                        else:
                            symbol_data[symbol] = None
                else:
                    st.info("Không tìm thấy dữ liệu RRG trong database. Đang tính toán trực tiếp...")
                    for symbol in margin_symbols:
                        price_df = db_connector.fetch_price_data(conn, [symbol, BENCHMARK_SYMBOL], str_start, str_end)
                        if price_df.empty:
                            symbol_data[symbol] = None
                            continue

                        rrg_df = calculate_rrg_data(price_df, BENCHMARK_SYMBOL, RRG_PERIOD, SCALE_FACTOR)
                        if rrg_df.empty:
                            symbol_data[symbol] = None
                            continue

                        symbol_data[symbol] = rrg_df

        # Display charts in rows based on charts_per_line setting
        for row_symbols in chunk_list(margin_symbols, charts_per_line):
            cols = st.columns(charts_per_line)
            for idx, symbol in enumerate(row_symbols):
                with cols[idx]:
                    st.markdown(f"### {symbol}")
                    if symbol_data.get(symbol) is None:
                        st.warning(f"Không có dữ liệu cho {symbol}.")
                    else:
                        plot_rrg_time_series(symbol_data[symbol], symbol, BENCHMARK_SYMBOL, RRG_PERIOD)
                        


