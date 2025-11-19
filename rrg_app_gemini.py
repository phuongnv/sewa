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
# CẤU HÌNH CỐ ĐỊNH
# =====================
BENCHMARK_SYMBOL = 'VNINDEX' # Mã chuẩn cố định
RRG_PERIOD = 50              # Chu kỳ RRG cố định (WMA length)
DAYS_FOR_CHART = 365         # Số ngày mặc định để vẽ biểu đồ (1 năm)
SCALE_FACTOR = 4.0           # Hệ số scale để dịch chuyển Z-Score về tâm 100

# =====================
# LOAD .ENV CONFIG
# =====================
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "stock_db")

if all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    DB_CONN = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_CONN = None
    st.error("Thiếu cấu hình biến môi trường DB.")

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="RRG Chart — VN Index", layout="wide")

# =====================
# ABSTRACT DATA SOURCE
# =====================
class DataSource(ABC):
    @abstractmethod
    def get_data(self, symbols, start_date, end_date) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> list:
        pass

# =====================
# CUSTOM DB SOURCE
# =====================
class CustomDBSource(DataSource):

    def __init__(self, connection_string=None):
        self.connection_string = connection_string
        self.engine = None
        if connection_string:
            try:
                self.engine = sqlalchemy.create_engine(connection_string)
            except Exception as e:
                st.warning(f"Không thể khởi tạo kết nối DB: {e}")
    
    # @st.cache_data(ttl=3600) # Cache danh sách mã 1 tiếng
    def get_available_symbols(self) -> list:
        """Lấy danh sách các mã chứng khoán có sẵn trong DB."""
        if self.engine is None:
            return []
        
        query = "SELECT DISTINCT symbol FROM stock_prices ORDER BY symbol ASC"
        try:
            df = pd.read_sql(query, self.engine)
            # Thêm VNINDEX nếu chưa có (để đảm bảo lấy được dữ liệu benchmark)
            symbols = df['symbol'].tolist()
            if BENCHMARK_SYMBOL not in symbols:
                symbols.insert(0, BENCHMARK_SYMBOL)
            return symbols
        except Exception as e:
            st.error(f"Lỗi khi truy vấn danh sách mã: {e}")
            return []

    def get_data(self, symbols=None, start_date=None, end_date=None):
        if self.engine is None:
            return pd.DataFrame()

        all_symbols = list(symbols) if symbols else []
        if BENCHMARK_SYMBOL not in all_symbols:
            all_symbols.append(BENCHMARK_SYMBOL)

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
    
    if len(x) < len(weights):
        return np.nan

    return np.sum(x.values[-len(weights):] * weights) / np.sum(weights)


@st.cache_data
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

    # --- Cấu hình Chart (Giữ nguyên) ---
    quadrant_colors = {'Leading': 'green', 'Weakening': '#ffc000', 'Lagging': 'red', 'Improving': 'blue'}
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)
    
    # Đặt giới hạn trục X và Y
    min_val = min(rs.min(), rm.min(), 98)
    max_val = max(rs.max(), rm.max(), 102)
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    
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



    ax.set_title(f'RRG Time Series Chart: {symbol} vs {benchmark} (Chu kỳ: {period} ngày)', fontsize=14)
    ax.set_xlabel('Relative Strength (RS Ratio)')
    ax.set_ylabel('Relative Momentum (RM Momentum)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') 

    st.pyplot(fig)

# =====================
# STREAMLIT APP
# =====================
def main():
    st.title("📈 RRG Time Series Chart (VNINDEX - P50)")

    # Khởi tạo nguồn dữ liệu
    data_source = CustomDBSource(DB_CONN)
    
    # Lấy danh sách mã chứng khoán từ DB
    # 1. Định nghĩa khóa cache
    CACHE_KEY_SYMBOLS = "cached_symbols_list"
    
    # 2. Kiểm tra nếu danh sách mã chưa có trong cache
    if CACHE_KEY_SYMBOLS not in st.session_state:
        with st.spinner("Đang tải danh sách mã chứng khoán lần đầu..."):
            # Gọi phương thức lấy dữ liệu từ DB (không có @st.cache_data)
            symbols_list = data_source.get_available_symbols() 
            
            # Lưu kết quả vào session_state
            st.session_state[CACHE_KEY_SYMBOLS] = symbols_list
            
    all_available_symbols = st.session_state[CACHE_KEY_SYMBOLS]

    if not all_available_symbols:
        st.error("Không thể tải danh sách mã chứng khoán từ database.")
        return

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("⚙️ Cấu hình Chart")
        
        # 1. Mã chuẩn (Cố định)
        st.info(f"Mã chuẩn: **{BENCHMARK_SYMBOL}** (Cố định)")

        # 2. Chu kỳ RRG (Cố định)
        st.info(f"Chu kỳ RRG: **{RRG_PERIOD} ngày** (Cố định)")

        # 3. Mã chứng khoán cần vẽ (Lấy từ DB)
        selected_symbol = st.selectbox(
            "Nhập Mã chứng khoán cần vẽ",
            options=all_available_symbols,
            index=all_available_symbols.index('FPT') if 'FPT' in all_available_symbols else (0 if all_available_symbols else None),
            key='selected_symbol'
        )

        # 4. Date Pickers (Tự động tính ngày bắt đầu)
        today = datetime.now().date()
        date_to = st.date_input("Ngày Kết thúc", value=today, max_value=today)

        # Ngày bắt đầu phải đủ xa để tính WMA 50 ngày và vẽ 1 năm dữ liệu
        # Ta cần ít nhất 50 ngày trước ngày bắt đầu để tính RRG đầu tiên.
        min_start_date_needed = date_to - timedelta(days=DAYS_FOR_CHART + RRG_PERIOD * 2)
        
        # Ngày bắt đầu mặc định cho người dùng thấy
        default_date_from = date_to - timedelta(days=DAYS_FOR_CHART)
        
        date_from = st.date_input("Ngày Bắt đầu", 
                                value=default_date_from, 
                                max_value=date_to - timedelta(days=1),
                                help=f"Hệ thống sẽ lấy dữ liệu từ ngày {min_start_date_needed.strftime('%Y-%m-%d')} để đảm bảo tính toán đủ 50 ngày WMA."
                            )
        
        # NEW: Limit number of last points to draw on RRG Time Series chart
        limit_points = st.slider(
            "Số điểm cuối cùng để vẽ (last N points)",
            min_value=5,
            max_value=DAYS_FOR_CHART,
            value=min(20, DAYS_FOR_CHART),
            step=1,
            help="Giới hạn số điểm gần nhất để vẽ trên biểu đồ RRG Time Series."
        )

    # --- Main App Logic ---
    if not selected_symbol:
        st.warning("Vui lòng chọn Mã chứng khoán cần vẽ.")
        return

    # Tính toán ngày cần thiết để lấy dữ liệu thô (để có đủ 50 ngày WMA trước ngày date_from)
    fetch_start_date = date_from - timedelta(days=RRG_PERIOD * 2) 
    
    # Lấy dữ liệu
    all_symbols_to_fetch = [selected_symbol, BENCHMARK_SYMBOL]
    
    with st.spinner(f"Đang tải dữ liệu cho {', '.join(all_symbols_to_fetch)} từ {fetch_start_date} đến {date_to}..."):
        data_df = data_source.get_data(
            symbols=all_symbols_to_fetch, 
            start_date=fetch_start_date, # Dùng ngày bắt đầu mở rộng
            end_date=date_to
        )

    if data_df.empty:
        st.error(f"❌ Không có dữ liệu để tính RRG cho {selected_symbol} hoặc {BENCHMARK_SYMBOL} trong khoảng thời gian yêu cầu.")
        return

    # Tính toán RRG (Đã được SCALED về tâm 100)
    with st.spinner(f"Đang tính toán chỉ số RRG (P={RRG_PERIOD}) và Chuẩn hóa (Tâm 100)..."):
        rrg_df_raw = calculate_rrg_data(
            data_df, 
            benchmark_symbol=BENCHMARK_SYMBOL, 
            period=RRG_PERIOD,
            scale_factor=SCALE_FACTOR
        )

    if rrg_df_raw.empty or 'rs_ratio_scaled' not in rrg_df_raw.columns:
        st.error(f"❌ Không thể tính RRG. Có thể dữ liệu không đủ {RRG_PERIOD} ngày liên tiếp.")
        return
        
    # Lọc lại dữ liệu để chỉ hiển thị trên biểu đồ trong khoảng ngày mà người dùng đã chọn
    rrg_df = rrg_df_raw[rrg_df_raw['date'] >= pd.to_datetime(date_from)]

    if rrg_df.empty:
        st.warning("Dữ liệu sau khi tính RRG không còn điểm nào trong khoảng ngày bạn chọn.")
        return
    
    # APPLY LIMIT: keep only last `limit_points` per symbol (show recent points only)
    try:
        limit_points = int(limit_points)
        if limit_points <= 0:
            limit_points = min(20, DAYS_FOR_CHART)
    except Exception:
        limit_points = min(20, DAYS_FOR_CHART)

    # Keep the last N rows per symbol sorted by date
    rrg_df = (
        rrg_df.sort_values(['symbol', 'date'])
              .groupby('symbol', group_keys=False)
              .apply(lambda g: g.tail(limit_points))
              .reset_index(drop=True)
    )
        
    # Vẽ biểu đồ
    plot_rrg_time_series(rrg_df, selected_symbol, BENCHMARK_SYMBOL, RRG_PERIOD)
    
    st.markdown("---")
    st.subheader("Dữ liệu RRG đã tính toán và Chuẩn hóa (Top 5)")
    st.dataframe(rrg_df[rrg_df['symbol'] == selected_symbol].tail())
    

if __name__ == "__main__":
    main()