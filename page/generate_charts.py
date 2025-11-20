import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import db_connector

BENCHMARK_SYMBOL = "VNINDEX"
RRG_PERIOD = 50
DAYS_FOR_CHART = 365
SCALE_FACTOR = 4.0



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
    filter_margin = st.checkbox("Filter theo Margin (>= giá trị)", value=True)
    if filter_margin:
        margin_threshold = st.number_input(
            "Giá trị margin tối thiểu",
            min_value=0,
            max_value=100,
            value=1,
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

    margin_symbols = db_connector.fetch_symbols_by_filters(
        conn,
        use_margin=filter_margin,
        margin_value=margin_threshold,
        use_recommend=filter_recommend,
        recommend_value=recommend_value,
    )

    if margin_symbols:
        active_filters = []
        if filter_margin:
            active_filters.append(f"margin ≥ {margin_threshold}")
        if filter_recommend:
            active_filters.append(f"recommend = {recommend_value}")
        filter_text = ", ".join(active_filters) if active_filters else "Không áp dụng bộ lọc"
        st.info(f"Đang áp dụng bộ lọc: {filter_text}. Tìm thấy **{len(margin_symbols)}** mã.")
    else:
        st.warning("Không tìm thấy mã nào theo bộ lọc đã chọn.")

    charts_per_line = st.number_input(
        "Số biểu đồ mỗi dòng",
        min_value=1,
        max_value=6,
        value=3,
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
        with st.spinner("Đang tải dữ liệu cho tất cả các mã..."):
            for symbol in margin_symbols:
                price_df = db_connector.fetch_price_data(conn, [symbol, BENCHMARK_SYMBOL], str_start, str_end)
                # if price_df.empty or symbol not in price_df.columns:
                #     symbol_data[symbol] = None
                #     continue

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
                        


