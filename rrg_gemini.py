import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import db_connector # Import DB utility
from datetime import timedelta

# =====================
# CẤU HÌNH CỐ ĐỊNH & HẰNG SỐ (RRG-specific)
# =====================
BENCHMARK_SYMBOL = 'VNINDEX'
RRG_PERIOD = 50              # Chu kỳ WMA cho RRG
DAYS_FOR_CHART = 365         # Số ngày dữ liệu tối đa
SCALE_FACTOR = 4.0           # Hệ số tỉ lệ cho RRG để nén dữ liệu


# =====================
# DATA SOURCE (POSTGRESQL)
# =====================

def get_data_from_db(conn, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Tải dữ liệu giá từ PostgreSQL."""
    if not conn:
        return pd.DataFrame()
    
    # Thêm benchmark vào danh sách yêu cầu
    all_symbols_to_fetch = list(set(symbols + [BENCHMARK_SYMBOL]))
    
    # Lấy dữ liệu từ DB
    return db_connector.fetch_price_data(conn, all_symbols_to_fetch, start_date, end_date)


# =====================
# RRG CALCULATION (Không đổi)
# =====================

@st.cache_data(ttl=3600)
def calculate_rrg_data(price_df: pd.DataFrame, benchmark_symbol: str, period: int, scale_factor: float) -> pd.DataFrame:
    """Tính toán RRG: Relative Strength (RS) và Relative Momentum (RM)."""
    
    if price_df.empty or benchmark_symbol not in price_df.columns:
        return pd.DataFrame()

    df = price_df.copy()
    symbols = [col for col in df.columns if col != benchmark_symbol]

    rrg_results = []

    for symbol in symbols:
        # 1. Tính toán Relative Strength (RS) - J.W.T.
        rs_ratio = (df[symbol] / df[benchmark_symbol])
        
        # 2. Làm mượt bằng WMA (Weighted Moving Average)
        rs_ratio_wma = rs_ratio.ewm(span=period, adjust=False).mean()

        # 3. Tính toán Relative Momentum (RM)
        rs_momentum = rs_ratio_wma.pct_change(periods=period) 
        
        # 4. Chuẩn hóa Z-Score cho RS và RM
        rs_ratio_z = (rs_ratio_wma - rs_ratio_wma.mean()) / rs_ratio_wma.std()
        rs_momentum_z = (rs_momentum - rs_momentum.mean()) / rs_momentum.std()
        
        # 5. Scaling (Đưa về tâm 100)
        rs_ratio_scaled = 100 + rs_ratio_z * scale_factor
        rs_momentum_scaled = 100 + rs_momentum_z * scale_factor

        temp_df = pd.DataFrame({
            'date': df.index,
            'symbol': symbol,
            'rs_ratio_scaled': rs_ratio_scaled,
            'rs_momentum_scaled': rs_momentum_scaled
        }).dropna()
        
        rrg_results.append(temp_df)

    if not rrg_results:
        return pd.DataFrame()

    rrg_df = pd.concat(rrg_results, ignore_index=True)
    return rrg_df

# =====================
# RRG CHART PLOTTING (Không đổi)
# =====================

def plot_rrg_time_series(rrg_df: pd.DataFrame, symbol: str, benchmark: str, period: int):
    """
    Vẽ biểu đồ RRG Continuous Arrow Trail (Tâm 100).
    (Code vẽ biểu đồ giữ nguyên)
    """
    if rrg_df.empty:
        st.info("Không có dữ liệu RRG để vẽ biểu đồ.")
        return

    df_symbol = rrg_df[rrg_df['symbol'] == symbol].copy().reset_index(drop=True)
    rs = df_symbol['rs_ratio_scaled']
    rm = df_symbol['rs_momentum_scaled']

    if rs.empty:
        st.warning(f"Không tìm thấy dữ liệu RRG đã tính toán cho mã {symbol}.")
        return

    U = np.diff(rs) 
    V = np.diff(rm) 
    X = rs.iloc[:-1]
    Y = rm.iloc[:-1]

    fig, ax = plt.subplots(figsize=(10, 10))

    quadrant_colors = {'Leading': 'green', 'Weakening': '#ffc000', 'Lagging': 'red', 'Improving': 'blue'}
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)
    
    min_val = min(rs.min(), rm.min(), 98)
    max_val = max(rs.max(), rm.max(), 102)
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    # 3. VẼ MŨI TÊN (QUIVER PLOT)
    quadrants_series = pd.Series(index=X.index, dtype=str)
    quadrants_series[(X >= 100) & (Y >= 100)] = 'Leading'
    quadrants_series[(X >= 100) & (Y < 100)] = 'Weakening'
    quadrants_series[(X < 100) & (Y < 100)] = 'Lagging'
    quadrants_series[(X < 100) & (Y >= 100)] = 'Improving'

    colors_quiver = [quadrant_colors.get(q, 'gray') for q in quadrants_series]
    
    # Vẽ đường dẫn (tạo cảm giác đường nét liền mạch)
    ax.quiver(
        X, Y, U, V, 
        color=colors_quiver,
        scale_units='xy', 
        scale=1, 
        width=0.008, 
        headwidth=0, 
        headlength=0,
        zorder=3
    )
    
    # Vẽ mũi tên cuối cùng (chỉ hướng hiện tại)
    if len(rs) >= 2:
        ax.quiver(
            rs.iloc[-2], rm.iloc[-2], rs.iloc[-1] - rs.iloc[-2], rm.iloc[-1] - rm.iloc[-2],
            color='black',
            scale_units='xy', 
            scale=1, 
            width=0.015,
            headwidth=7, 
            headlength=10,
            zorder=5 
        )

    # 4. ĐIỂM HIỆN TẠI (CUỐI CÙNG)
    ax.scatter(rs.iloc[-1], rm.iloc[-1], color='black', s=150, zorder=6) 
    ax.text(rs.iloc[-1], rm.iloc[-1], symbol, fontsize=12, ha='right', va='bottom', zorder=7) 

    # 5. Các nhãn và cấu hình khác 
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95, 'Leading', fontsize=12, color='green', ha='right', va='top')
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] * 1.05, 'Weakening', fontsize=12, color='#ffc000', ha='right', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[0] * 1.05, 'Lagging', fontsize=12, color='red', ha='left', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[1] * 0.95, 'Improving', fontsize=12, color='blue', ha='left', va='top')

    ax.set_title(f'RRG Relative Rotation Graph: {symbol} vs {BENCHMARK_SYMBOL} (Period: {RRG_PERIOD})', fontsize=14)
    ax.set_xlabel('Relative Strength (RS Ratio)')
    ax.set_ylabel('Relative Momentum (RM Momentum)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') 

    st.pyplot(fig)


# =====================
# RRG ANALYZER PAGE FUNCTION
# =====================

def rrg_analyzer_page(conn):
    """Trang phân tích RRG chính."""
    
    # Giả lập danh sách mã (Trong thực tế, bạn sẽ lấy danh sách từ DB)
    all_symbols = ['FPT', 'HPG', 'SSI', 'ACB', 'VPB', 'VNDIRECT', 'GAS', BENCHMARK_SYMBOL]
    all_available_symbols = sorted([s for s in all_symbols if s != BENCHMARK_SYMBOL])

    st.title("📈 Phân Tích Biểu Đồ Relative Rotation Graph (RRG)")
    st.markdown("Dữ liệu giá được lấy từ PostgreSQL (Bảng `price_data`).")
    
    # Cấu hình sidebar
    with st.sidebar:
        st.header("⚙️ Cấu hình Chart")
        
        if not all_available_symbols:
            st.error("Không thể tải danh sách mã chứng khoán.")
            return

        selected_symbol = st.selectbox(
            "Chọn Mã Chứng Khoán", 
            options=all_available_symbols,
            index=all_available_symbols.index('FPT') if 'FPT' in all_available_symbols else 0,
            key='selected_symbol'
        )

        end_date = pd.Timestamp.today().date()
        start_date = end_date - timedelta(days=DAYS_FOR_CHART)
        
        st.info("Đang tải dữ liệu giá từ PostgreSQL...")

    if not conn:
        st.warning("Không có kết nối DB. Vui lòng kiểm tra cấu hình.")
        return

    # Tải và Tính toán Dữ liệu
    price_df = get_data_from_db(conn, [selected_symbol], str(start_date), str(end_date))

    if price_df.empty or selected_symbol not in price_df.columns or BENCHMARK_SYMBOL not in price_df.columns:
        st.error(f"Không tìm thấy dữ liệu giá cho {selected_symbol} hoặc {BENCHMARK_SYMBOL} trong DB. Vui lòng kiểm tra bảng `price_data`.")
        return

    # Tính toán RRG
    with st.spinner("Đang tính toán tọa độ RRG..."):
        rrg_df = calculate_rrg_data(price_df, BENCHMARK_SYMBOL, RRG_PERIOD, SCALE_FACTOR)
    
    if rrg_df.empty:
        st.warning("Lỗi tính toán RRG. Vui lòng kiểm tra dữ liệu.")
        return

    # Vẽ biểu đồ
    plot_rrg_time_series(rrg_df, selected_symbol, BENCHMARK_SYMBOL, RRG_PERIOD)

# --- END OF rrg_gemini.py ---