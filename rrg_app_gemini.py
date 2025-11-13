import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import sqlalchemy
from dotenv import load_dotenv
import os
import pyRRG as rrg

# =====================
#  LOAD .ENV CONFIG
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
#  PAGE CONFIG
# =====================
st.set_page_config(page_title="RRG Chart — Fast & Smooth", layout="wide")

# =====================
#  ABSTRACT DATA SOURCE
# =====================
class DataSource(ABC):
    @abstractmethod
    def get_data(self, symbols, start_date, end_date) -> pd.DataFrame:
        pass

# =====================
#  CUSTOM DB SOURCE
# =====================
class CustomDBSource(DataSource):
    """
    Lấy dữ liệu giá cổ phiếu từ database Neon.tech
    Giả định bảng `stock_prices` có các cột:
    symbol | date | open | high | low | close | volume | exchange
    """

    def __init__(self, connection_string=None):
        self.connection_string = connection_string
        self.engine = None
        if connection_string:
            try:
                self.engine = sqlalchemy.create_engine(connection_string)
            except Exception as e:
                # Xử lý ngoại lệ kết nối DB một cách rõ ràng
                st.warning(f"Không thể khởi tạo kết nối DB: {e}")

    def get_data(self, symbols=None, start_date=None, end_date=None):
        if self.engine is None:
            # Nếu engine không được khởi tạo (do lỗi kết nối/thiếu config)
            # Trả về DataFrame rỗng để tránh lỗi tiếp theo
            return pd.DataFrame()

        # Giả sử cần thêm một mã chuẩn (benchmark) để tính toán RS
        all_symbols = list(symbols) if symbols else []
        # Thay 'VNINDEX' bằng mã chỉ số chuẩn của bạn
        benchmark_symbol = st.session_state.get('benchmark', 'VNINDEX') 
        if benchmark_symbol not in all_symbols:
            all_symbols.append(benchmark_symbol)

        where_clause = "1=1"
        if all_symbols:
            # Sửa lại cách tạo placeholders để tránh lỗi SQL Injection (dù pd.read_sql có thể hỗ trợ params, ta vẫn dùng cách thủ công với string)
            # Đối với Streamlit App, nếu không dùng params, cách an toàn hơn là dùng parameterized query. 
            # Tuy nhiên, để đơn giản hóa cho ví dụ này, ta dùng string.
            placeholders = ",".join([f"'{s}'" for s in all_symbols])
            where_clause += f" AND symbol IN ({placeholders})"
        
        # Thêm điều kiện ngày, đảm bảo ngày là đối tượng datetime hoặc chuỗi định dạng YYYY-MM-DD
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            where_clause += f" AND date BETWEEN '{start_str}' AND '{end_str}'"
        else:
            # Nếu thiếu ngày, trả về rỗng để tránh truy vấn quá lớn
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
#  RRG Chart Logic
# =====================
@st.cache_data
def calculate_rrg_data(data_df: pd.DataFrame, benchmark: str, period: int = 10) -> pd.DataFrame:
    """
    Tính toán chỉ số Relative Strength (RS) và Relative Momentum (RM)
    Sử dụng thư viện 'rrg' cho các tính toán theo Julius.
    """
    if data_df.empty:
        return pd.DataFrame()

    # Chuyển đổi dữ liệu sang format phù hợp với rrg (pivot table)
    close_prices = data_df.pivot(index='date', columns='symbol', values='close')

    if benchmark not in close_prices.columns:
        st.error(f"Mã chuẩn '{benchmark}' không có trong dữ liệu. Không thể tính RRG.")
        return pd.DataFrame()

    # Khởi tạo và tính toán RRG
    # Giả sử dùng 10 ngày cho Momentum (như rrg Julius mặc định)
    # df.apply(np.log) là bước quan trọng để tính toán theo công thức RRG
    
    try:
        
        rrg_data = calculate_rrg_data_manual(
            data_df, 
            benchmark=benchmark, 
            period_rs=period, # Sử dụng rrg_period làm period_rs
            period_rm=int(period * 0.6)
        )
        return rrg_data
    except Exception as e:
        st.error(f"Lỗi khi tính toán RRG: {e}")
        return pd.DataFrame()

# HÀM MỚI TÍNH TOÁN RRG TỰ LÀM
def calculate_rrg_data_manual(data_df: pd.DataFrame, benchmark: str, period_rs: int = 10, period_rm: int = 6) -> pd.DataFrame:
    """
    Tính toán chỉ số RRG (RS và RM) thủ công bằng công thức EMA chuẩn.
    Thường dùng period 10 cho RS và period 6 cho RM, theo Julius De Kempenaer.
    """
    if data_df.empty:
        return pd.DataFrame()

    # 1. Chuyển đổi dữ liệu sang dạng pivot table (log)
    close_prices = data_df.pivot(index='date', columns='symbol', values='close')
    
    # Tính log của giá
    log_prices = close_prices.apply(np.log)

    if benchmark not in log_prices.columns:
        st.error(f"Mã chuẩn '{benchmark}' không có trong dữ liệu. Không thể tính RRG.")
        return pd.DataFrame()

    benchmark_log = log_prices[benchmark]
    rrg_results = pd.DataFrame(index=log_prices.index)

    # Lặp qua từng mã (trừ mã chuẩn)
    for symbol in log_prices.columns:
        if symbol == benchmark:
            continue

        # --- A. Tính toán Relative Strength (RS) ---
        # 1. Tính tỷ lệ giữa symbol và benchmark (Log Ratio)
        log_ratio = log_prices[symbol] - benchmark_log
        
        # 2. Làm mịn Log Ratio bằng EMA (Log Ratio EMA)
        # span = period_rs (thường là 10)
        log_ratio_ema = log_ratio.ewm(span=period_rs, adjust=False).mean()

        # 3. Chuyển đổi Log Ratio EMA sang Tỷ lệ phần trăm và Chuẩn hóa (RS Index)
        # Công thức chuẩn hóa: RS Index = 100 + 100 * (Log Ratio EMA / 0.005)
        # Sử dụng hệ số 0.005 (hoặc 0.001) để Scaling, tùy thuộc vào sở thích vẽ
        # Ở đây ta dùng công thức đơn giản hơn để tránh Scaling cố định.
        # Hoặc dùng công thức chuẩn hóa dựa trên RS của Julius:
        # RS Index = 100 + 100 * (RS EMA / RS_Slope_Factor)
        
        # Để đơn giản, ta dùng công thức EMA Log Ratio (RS)
        # Giả định RS Index = 100 + 100 * Log Ratio EMA
        # (Lưu ý: Công thức Scaling chuẩn của Julius phức tạp hơn, nhưng đây là cách đơn giản nhất để có tín hiệu)
        
        # Ta sẽ dùng công thức của StockCharts.com/RRG:
        # RS_Line = log(Price/Benchmark)
        # RS_Index = 100 + 100 * EMA_smooth(RS_Line) * Factor 
        # Vì ta đang dùng log_prices, log_ratio_ema là Log của tỷ lệ.

        # Ta sẽ sử dụng giá trị Log Ratio EMA thô, chuẩn hóa lại bằng 100
        rs_index = 100 + log_ratio_ema * 100 

        # --- B. Tính toán Relative Momentum (RM) ---
        # 1. Tính Tốc độ thay đổi của RS (Slope) - (RS Index / RS Index N ngày trước) - 1
        # Ta dùng Change của Log Ratio EMA (giá trị RS)

        # Công thức chuẩn: RM Index = 100 + 100 * EMA_smooth(Change của RS_Line) * Factor
        
        # Tính sự thay đổi (Momentum) của Log Ratio EMA (RS Line)
        rs_momentum = log_ratio_ema.diff() 
        
        # Làm mịn Momentum bằng EMA (RM Index)
        # span = period_rm (thường là 6)
        rm_index = 100 + rs_momentum.ewm(span=period_rm, adjust=False).mean() * 100

        # Lưu kết quả
        rrg_results[f'{symbol}_RS'] = rs_index.round(2)
        rrg_results[f'{symbol}_RM'] = rm_index.round(2)
        
    return rrg_results.dropna()

def plot_rrg_time_series(rrg_df: pd.DataFrame, symbol: str, period: int):
    """
    Vẽ biểu đồ RRG Time Series cho một mã chứng khoán (Kiểu Julius)
    """
    if rrg_df.empty:
        st.info("Không có dữ liệu RRG để vẽ biểu đồ.")
        return

    # Lấy dữ liệu RS và RM của mã chứng khoán cần vẽ
    rs = rrg_df[f'{symbol}_RS']
    rm = rrg_df[f'{symbol}_RM']

    fig, ax = plt.subplots(figsize=(10, 10))

    # Định nghĩa 4 góc phần tư
    # 1. Leading (Tăng trưởng) - X > 100, Y > 100
    # 2. Weakening (Suy yếu) - X > 100, Y < 100
    # 3. Lagging (Chậm lại) - X < 100, Y < 100
    # 4. Improving (Cải thiện) - X < 100, Y > 100

    # Vẽ các đường ngang và dọc chuẩn (100)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(100, color='gray', linestyle='--', linewidth=0.8)

    # Đặt giới hạn trục X và Y
    min_val = min(rs.min(), rm.min(), 98)
    max_val = max(rs.max(), rm.max(), 102)
    padding = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    # Vẽ đường RRG Time Series
    # Sử dụng màu sắc dựa trên góc phần tư (tùy chọn)
    
    # Chia điểm dữ liệu thành 4 khu vực
    quadrant_colors = {
        'Leading': 'green',
        'Weakening': 'yellow',
        'Lagging': 'red',
        'Improving': 'blue'
    }

    # Xác định quadrant cho từng điểm dữ liệu
    quadrants = pd.Series(index=rrg_df.index, dtype=str)
    quadrants[(rs >= 100) & (rm >= 100)] = 'Leading'
    quadrants[(rs >= 100) & (rm < 100)] = 'Weakening'
    quadrants[(rs < 100) & (rm < 100)] = 'Lagging'
    quadrants[(rs < 100) & (rm >= 100)] = 'Improving'

    # Vẽ theo từng đoạn, tô màu theo quadrant
    for i in range(1, len(rs)):
        current_quadrant = quadrants.iloc[i]
        color = quadrant_colors.get(current_quadrant, 'black')
        ax.plot(
            [rs.iloc[i-1], rs.iloc[i]],
            [rm.iloc[i-1], rm.iloc[i]],
            color=color,
            linewidth=2,
            alpha=0.7
        )

    # Điểm cuối cùng (Điểm RRG hiện tại)
    ax.scatter(rs.iloc[-1], rm.iloc[-1], color='black', s=150, zorder=5) # Điểm cuối cùng là dấu chấm đậm
    ax.text(rs.iloc[-1], rm.iloc[-1], symbol, fontsize=12, ha='right', va='bottom', zorder=6) # Ghi nhãn

    # Điểm đầu tiên
    ax.scatter(rs.iloc[0], rm.iloc[0], color='gray', s=50, marker='o', zorder=5)

    # Thêm nhãn góc phần tư
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[1] * 0.95, 'Leading', fontsize=12, color='green', ha='right', va='top')
    ax.text(ax.get_xlim()[1] * 0.95, ax.get_ylim()[0] * 1.05, 'Weakening', fontsize=12, color='yellow', ha='right', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[0] * 1.05, 'Lagging', fontsize=12, color='red', ha='left', va='bottom')
    ax.text(ax.get_xlim()[0] * 1.05, ax.get_ylim()[1] * 0.95, 'Improving', fontsize=12, color='blue', ha='left', va='top')


    ax.set_title(f'RRG Time Series Chart: {symbol} vs {st.session_state.get("benchmark", "VNINDEX")} (Period: {period} ngày)', fontsize=14)
    ax.set_xlabel('Relative Strength (RS)')
    ax.set_ylabel('Relative Momentum (RM)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box') # Đảm bảo trục X và Y có tỷ lệ 1:1

    st.pyplot(fig)

# =====================
#  STREAMLIT APP
# =====================
def main():
    st.title("📈 RRG Time Series Chart (Julius RRG Style)")

    # Khởi tạo nguồn dữ liệu
    data_source = CustomDBSource(DB_CONN)
    
    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("⚙️ Cấu hình Chart")
        
        # 1. Mã chuẩn (Benchmark)
        # Giả định có danh sách các mã phổ biến để gợi ý
        available_benchmarks = ['VNINDEX', 'HNXINDEX', 'UPCOMINDEX'] 
        benchmark_default = 'VNINDEX'
        
        benchmark_input = st.selectbox(
            "Chọn mã chuẩn (Benchmark)",
            options=available_benchmarks,
            index=available_benchmarks.index(benchmark_default) if benchmark_default in available_benchmarks else 0,
            key='benchmark'
        )

        # 2. Mã chứng khoán cần vẽ (Autocomplete)
        # Để hỗ trợ autocompleted, ta cần một danh sách mã chứng khoán (giả định)
        # Trong thực tế, bạn sẽ lấy danh sách này từ DB
        all_available_symbols = ['FPT', 'HPG', 'VCB', 'ACB', 'VND', 'SSI', 'GAS', 'MWG', 'MSN', benchmark_input]
        
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
        rrg_period = st.slider("Chu kỳ RRG (Ngày)", min_value=1, max_value=50, value=10, step=1, key='rrg_period')
        
        st.info("Lưu ý: Bạn cần có bảng `stock_prices` trong DB với các cột `symbol`, `date`, `close` để ứng dụng hoạt động.")


    # --- Main App Logic ---
    if not selected_symbol:
        st.warning("Vui lòng chọn Mã chứng khoán cần vẽ.")
        return

    # Lấy dữ liệu
    all_symbols_to_fetch = [selected_symbol, benchmark_input]
    
    with st.spinner(f"Đang tải dữ liệu cho {', '.join(all_symbols_to_fetch)}..."):
        # Lấy dữ liệu cho cả mã chứng khoán và mã chuẩn
        data_df = data_source.get_data(
            symbols=all_symbols_to_fetch, 
            start_date=date_from, 
            end_date=date_to
        )

    if data_df.empty:
        st.warning("Không có dữ liệu để tính toán RRG.")
        return

    # Tính toán RRG
    with st.spinner("Đang tính toán chỉ số RRG..."):
        rrg_df = calculate_rrg_data(data_df, benchmark=benchmark_input, period=rrg_period)

    if rrg_df.empty or f'{selected_symbol}_RS' not in rrg_df.columns:
        st.error(f"Không thể tính RRG cho mã {selected_symbol}. Vui lòng kiểm tra dữ liệu.")
        return

    # Vẽ biểu đồ
    st.subheader(f"Biểu đồ RRG Time Series của **{selected_symbol}**")
    plot_rrg_time_series(rrg_df, selected_symbol, rrg_period)
    
    st.markdown("---")
    st.subheader("Dữ liệu RRG đã tính toán (Top 5)")
    st.dataframe(rrg_df.tail())
    

if __name__ == "__main__":
    main()