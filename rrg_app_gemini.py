import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import sqlalchemy
from dotenv import load_dotenv
import os

# =====================
# LOAD .ENV CONFIG
# =====================
load_dotenv()

# Sử dụng giá trị mặc định an toàn cho các biến môi trường
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")

# Chỉ tạo DB_CONN nếu có đủ thông tin cần thiết
if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    DB_CONN = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_CONN = None
    st.error("Thiếu cấu hình biến môi trường DB (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME). Vui lòng kiểm tra file .env.")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="RRG Chart — Fast & Smooth", layout="wide")

# =====================
# ABSTRACT DATA SOURCE
# =====================
class DataSource(ABC):
    @abstractmethod
    def get_data(self, symbols, start_date, end_date) -> pd.DataFrame:
        pass

# =====================
# CUSTOM DB SOURCE
# =====================
class CustomDBSource(DataSource):
    """
    Lấy dữ liệu giá cổ phiếu từ database.
    """

    def __init__(self, connection_string=None):
        self.connection_string = connection_string
        self.engine = None
        if connection_string:
            try:
                self.engine = sqlalchemy.create_engine(connection_string)
            except Exception as e:
                st.warning(f"Không thể khởi tạo kết nối DB: {e}")

    def get_data(self, symbols=None, start_date=None, end_date=None):
        if self.engine is None:
            return pd.DataFrame()

        all_symbols = list(symbols) if symbols else []
        benchmark_symbol = st.session_state.get('benchmark', 'VNINDEX') 
        if benchmark_symbol not in all_symbols:
            all_symbols.append(benchmark_symbol)

        where_clause = "1=1"
        if all_symbols:
            placeholders = ",".join([f"'{s}'" for s in all_symbols])
            where_clause += f" AND symbol IN ({placeholders})"
        
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            where_clause += f" AND date BETWEEN '{start_str}' AND '{end_str}'"
        else:
            return pd.DataFrame()

        query = f"""
            SELECT symbol, date, close
            FROM stock_prices
            WHERE {where_clause}
            ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, self.engine)
            if df.empty:
                st.warning("Không tìm thấy dữ liệu cho các mã và khoảng thời gian đã chọn.")
                return pd.DataFrame()
                
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            st.error(f"Lỗi khi truy vấn database: {e}")
            return pd.DataFrame()

# =====================
# RRG UTILITY FUNCTIONS
# =====================

def normalize_data(data: pd.Series) -> pd.Series:
    """Chuẩn hóa Z-Score cho một Series."""
    if data.std() == 0:
        return pd.Series(0, index=data.index)
    return (data - data.mean()) / data.std()

def wma_func(x: pd.Series, period: int) -> float:
    """Hàm tính Weighted Moving Average (WMA) cho cửa sổ lăn."""
    weights = np.arange(1, len(x) + 1)
    weights = weights[len(weights) - period:] if len(weights) > period else weights
    
    # Chỉ tính WMA nếu có đủ dữ liệu bằng với số lượng trọng số
    if len(x) < len(weights):
        return np.nan

    return np.sum(x.values[-len(weights):] * weights) / np.sum(weights)


@st.cache_data
def calculate_rrg_data(df: pd.DataFrame, benchmark_symbol: str, period: int = 14) -> pd.DataFrame:
    """
    Tính toán chỉ số RRG (RS-Ratio và RS-Momentum) bằng WMA và 
    Chuẩn hóa Z-Score, sau đó Dịch chuyển về tâm 100.
    """
    df = df.copy()
    
    if df.empty:
        return pd.DataFrame()

    close_prices = df.pivot(index='date', columns='symbol', values='close')

    if benchmark_symbol not in close_prices.columns:
        return df

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
    rrg_results_long = pd.DataFrame(index=rs_ratio_wide.index)
    
    for symbol in rs_ratio_wide.columns:
        if symbol != benchmark_symbol:
            temp_df = pd.DataFrame({
                'date': rs_ratio_wide.index,
                'symbol': symbol,
                'rs_ratio': rs_ratio_wide[symbol].values,
                'rs_momentum': rs_momentum_wide[symbol].values
            })
            rrg_results_long = pd.concat([rrg_results_long, temp_df])

    # 3. CHUẨN HÓA VÀ DỊCH CHUYỂN TÂM 100
    
    # Chuẩn hóa Z-Score: (data - mean) / std (Tâm 0)
    rrg_results_long['rs_ratio_z'] = normalize_data(rrg_results_long['rs_ratio'])
    rrg_results_long['rs_momentum_z'] = normalize_data(rrg_results_long['rs_momentum'])

    # Dịch chuyển tâm về 100 (Scaling):
    # Công thức: Z-Score * Độ lệch chuẩn mục tiêu + 100
    # Ta sử dụng hệ số 5.5 hoặc 6.5 cho độ lệch chuẩn mục tiêu để tạo độ "lan" hợp lý, 
    # tương tự cách các nền tảng RRG thương mại sử dụng.
    SCALE_FACTOR = 5.5
    
    rrg_results_long['rs_ratio_scaled'] = 100 + rrg_results_long['rs_ratio_z'] * SCALE_FACTOR
    rrg_results_long['rs_momentum_scaled'] = 100 + rrg_results_long['rs_momentum_z'] * SCALE_FACTOR
    
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
    """
    Vẽ biểu đồ RRG Time Series cho một mã chứng khoán sử dụng dữ liệu scaled (tâm 100).
    """
    if rrg_df.empty:
        st.info("Không có dữ liệu RRG để vẽ biểu đồ.")
        return

    # Lấy dữ liệu đã được SCALED VỀ TÂM 100
    rs = rrg_df[rrg_df['symbol'] == symbol]['rs_ratio_scaled']
    rm = rrg_df[rrg_df['symbol'] == symbol]['rs_momentum_scaled']

    if rs.empty:
        st.warning(f"Không tìm thấy dữ liệu RRG đã tính toán cho mã {symbol}.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Định nghĩa 4 góc phần tư
    quadrant_colors = {
        'Leading': 'green',
        'Weakening': '#ffc000',  # Màu vàng/cam
        'Lagging': 'red',
        'Improving': 'blue'
    }

    # Vẽ các đường ngang và dọc chuẩn (TÂM 100)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)

    # Đặt giới hạn trục X và Y
    min_val = min(rs.min(), rm.min(), 98)
    max_val = max(rs.max(), rm.max(), 102)
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    
    # Xác định quadrant cho từng điểm dữ liệu
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

    # Điểm cuối cùng (Điểm RRG hiện tại)
    ax.scatter(rs.iloc[-1], rm.iloc[-1], color='black', s=150, zorder=5) 
    ax.text(rs.iloc[-1], rm.iloc[-1], symbol, fontsize=12, ha='right', va='bottom', zorder=6) 

    # Điểm đầu tiên
    ax.scatter(rs.iloc[0], rm.iloc[0], color='gray', s=50, marker='o', zorder=5)

    # Thêm nhãn góc phần tư
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95, 'Leading', fontsize=12, color='green', ha='right', va='top')
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] * 1.05, 'Weakening', fontsize=12, color='red', ha='right', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[0] * 1.05, 'Lagging', fontsize=12, color='blue', ha='left', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[1] * 0.95, 'Improving', fontsize=12, color='#ffc000', ha='left', va='top')


    ax.set_title(f'RRG Time Series Chart: {symbol} vs {benchmark} (Period: {period} ngày)', fontsize=14)
    ax.set_xlabel('Relative Strength (RS Ratio)')
    ax.set_ylabel('Relative Momentum (RM Momentum)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # Bắt buộc tỷ lệ 1:1

    st.pyplot(fig)


# =====================
# STREAMLIT APP
# =====================
def main():
    st.title("📈 RRG Time Series Chart (Chuẩn hóa Tâm 100)")

    # Khởi tạo nguồn dữ liệu
    data_source = CustomDBSource(DB_CONN)
    
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("⚙️ Cấu hình Chart")
        
        # 1. Mã chuẩn (Benchmark)
        available_benchmarks = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX', 'VN30'] 
        benchmark_default = 'VNINDEX'
        
        benchmark_input = st.selectbox(
            "Chọn mã chuẩn (Benchmark)",
            options=available_benchmarks,
            index=available_benchmarks.index(benchmark_default) if benchmark_default in available_benchmarks else 0,
            key='benchmark'
        )

        # 2. Mã chứng khoán cần vẽ (Autocomplete)
        # Giả định danh sách mã để demo autocomplete
        all_available_symbols = ['FPT', 'HPG', 'VCB', 'ACB', 'VND', 'SSI', 'GAS', 'MWG', 'MSN']
        
        selected_symbol = st.selectbox(
            "Nhập Mã chứng khoán cần vẽ (Ví dụ: FPT)",
            options=all_available_symbols,
            index=all_available_symbols.index('FPT') if 'FPT' in all_available_symbols else 0,
            key='selected_symbol'
        )

        # 3. Date Pickers
        today = datetime.now().date()
        default_start_date = today - timedelta(days=365)
        
        date_from = st.date_input("Ngày Bắt đầu", value=default_start_date, max_value=today - timedelta(days=1))
        date_to = st.date_input("Ngày Kết thúc", value=today, max_value=today)

        # 4. RRG Period
        rrg_period = st.slider("Chu kỳ RRG (Ngày - cho WMA)", min_value=1, max_value=50, value=14, step=1, key='rrg_period')
        
        st.info("Chu kỳ RRG thường dùng 10 hoặc 14 ngày.")


    # --- Main App Logic ---
    if not selected_symbol:
        st.warning("Vui lòng chọn Mã chứng khoán cần vẽ.")
        return

    # Lấy dữ liệu
    all_symbols_to_fetch = [selected_symbol, benchmark_input]
    
    with st.spinner(f"Đang tải dữ liệu cho {', '.join(all_symbols_to_fetch)}..."):
        data_df = data_source.get_data(
            symbols=all_symbols_to_fetch, 
            start_date=date_from, 
            end_date=date_to
        )

    if data_df.empty:
        st.warning("Không có dữ liệu để tính toán RRG.")
        return

    # Tính toán RRG (Đã được SCALED về tâm 100)
    with st.spinner("Đang tính toán chỉ số RRG và Chuẩn hóa (Tâm 100)..."):
        # Lỗi Scope đã được xử lý: biến được truyền vào hàm
        rrg_df = calculate_rrg_data(
            data_df, 
            benchmark_symbol=benchmark_input, 
            period=rrg_period
        )

    if rrg_df.empty or 'rs_ratio_scaled' not in rrg_df.columns:
        st.error(f"Không thể tính RRG cho mã {selected_symbol}. Vui lòng kiểm tra dữ liệu.")
        return

    # Vẽ biểu đồ
    st.subheader(f"Biểu đồ RRG Time Series: **{selected_symbol}**")
    plot_rrg_time_series(rrg_df, selected_symbol, benchmark_input, rrg_period)
    
    st.markdown("---")
    st.subheader("Dữ liệu RRG đã tính toán và Chuẩn hóa (Top 5)")
    st.dataframe(rrg_df[rrg_df['symbol'] == selected_symbol].tail())
    

if __name__ == "__main__":
    main()