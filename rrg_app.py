import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import sqlalchemy  # dùng cho DB sau này (MySQL, PostgreSQL...)

# =====================
#  CONFIG
# =====================
st.set_page_config(page_title="RRG Chart", layout="wide")


# =====================
#  ABSTRACT DATA SOURCE
# =====================
class DataSource(ABC):
    """Lớp trừu tượng cho mọi nguồn dữ liệu (Yahoo, DB, API...)"""
    
    @abstractmethod
    def get_data(self, symbols, start_date, end_date) -> pd.DataFrame:
        pass


# =====================
#  YAHOO DATA SOURCE
# =====================
class YahooFinanceSource(DataSource):
    """Nguồn dữ liệu từ Yahoo Finance"""
    
    def get_data(self, symbols, start_date, end_date):
        # data = yf.download(symbols, start=start_date, end=end_date)["Adj Close"]
        data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]

        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data


# =====================
#  CUSTOM DB SOURCE
# =====================
class CustomDBSource(DataSource):
    """
    Nguồn dữ liệu từ Database hoặc module realtime của bạn.
    Bạn có thể thay đổi kết nối & query theo hệ thống riêng.
    """
    
    def __init__(self, connection_string=None):
        """
        connection_string: ví dụ cho MySQL
        "mysql+pymysql://user:password@localhost:3306/stock_data"
        """
        self.connection_string = connection_string
        self.engine = None
        if connection_string:
            try:
                self.engine = sqlalchemy.create_engine(connection_string)
            except Exception as e:
                st.warning(f"Không thể khởi tạo kết nối DB: {e}")
    
    def get_data(self, symbols, start_date, end_date):
        """
        Lấy dữ liệu giá từ DB.
        Giả định bạn có bảng `prices` với schema:
        symbol | date | close_price
        """
        if self.engine is None:
            raise ConnectionError("Chưa có connection_string hợp lệ cho database.")
        
        query = f"""
            SELECT symbol, date, close_price
            FROM prices
            WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
              AND date BETWEEN '{start_date}' AND '{end_date}'
        """
        df = pd.read_sql(query, self.engine)
        
        # Pivot về dạng wide: mỗi cột là 1 symbol
        df_pivot = df.pivot(index="date", columns="symbol", values="close_price")
        df_pivot.index = pd.to_datetime(df_pivot.index)
        return df_pivot.sort_index()


# =====================
#  FACTORY FUNCTION
# =====================
def get_data(source_type, symbols, start_date, end_date, db_conn=None):
    """
    Gateway chọn nguồn dữ liệu
    """
    if source_type == "Yahoo":
        source = YahooFinanceSource()
    elif source_type == "Custom":
        source = CustomDBSource(connection_string=db_conn)
    else:
        raise ValueError("Nguồn dữ liệu không hợp lệ.")
    
    return source.get_data(symbols, start_date, end_date)


# =====================
#  RRG COMPUTATION
# =====================
def compute_rrg(df, benchmark_symbol):
    if benchmark_symbol not in df.columns:
        st.error(f"Không tìm thấy mã chuẩn '{benchmark_symbol}' trong dữ liệu.")
        return None

    benchmark = df[benchmark_symbol]
    rel_strength = df.divide(benchmark, axis=0)
    rs_ratio = 100 * rel_strength.divide(rel_strength.rolling(50).mean())
    rs_momentum = rs_ratio.pct_change(10) * 100

    last = pd.DataFrame({
        'RS-Ratio': rs_ratio.iloc[-1],
        'RS-Momentum': rs_momentum.iloc[-1]
    })
    return last.dropna()


def compute_rrg_series(df, benchmark_symbol, trail_days=20):
    """Tính toàn bộ chuỗi thời gian RS-Ratio và RS-Momentum"""
    if benchmark_symbol not in df.columns:
        st.error(f"Không tìm thấy mã chuẩn '{benchmark_symbol}' trong dữ liệu.")
        return None

    benchmark = df[benchmark_symbol]
    rel_strength = df.divide(benchmark, axis=0)
    rs_ratio = 100 * rel_strength.divide(rel_strength.rolling(50).mean())
    rs_momentum = rs_ratio.pct_change(10) * 100

    # Ghép 2 DataFrame để có dạng [date, symbol, RS-Ratio, RS-Momentum]
    long_df = []
    for sym in df.columns:
        if sym == benchmark_symbol:
            continue
        tmp = pd.DataFrame({
            "date": df.index,
            "symbol": sym,
            "RS-Ratio": rs_ratio[sym],
            "RS-Momentum": rs_momentum[sym]
        })
        long_df.append(tmp)
    rrg_df = pd.concat(long_df)
    
    # Giới hạn trail để dễ xem (ví dụ 20 ngày cuối)
    rrg_df = rrg_df.groupby("symbol").tail(trail_days)
    return rrg_df.dropna()


# =====================
#  STREAMLIT UI
# =====================
st.title("📊 Relative Rotation Graph (RRG)")

col1, col2 = st.columns(2)
with col1:
    symbols_input = st.text_input(
        "Danh sách mã cổ phiếu (phân tách bởi dấu phẩy)",
        value="AAPL,MSFT,AMZN,GOOG,META"
    )
with col2:
    benchmark_symbol = st.text_input("Mã chuẩn (benchmark)", value="^GSPC")

source_type = st.radio("Nguồn dữ liệu", ["Yahoo", "Custom"], horizontal=True)
db_conn = st.text_input(
    "Database connection string (nếu dùng Custom)",
    value="mysql+pymysql://user:password@localhost:3306/stock_data"
)
start_date = st.date_input("Ngày bắt đầu", datetime.today() - timedelta(days=365))
end_date = st.date_input("Ngày kết thúc", datetime.today())

if st.button("Tải & Tính RRG"):
    symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
    
    with st.spinner("Đang tải dữ liệu..."):
        try:
            df = get_data(source_type, symbols + [benchmark_symbol], start_date, end_date, db_conn=db_conn)
            st.success("✅ Dữ liệu tải thành công!")
        except Exception as e:
            st.error(f"Lỗi khi tải dữ liệu: {e}")
            st.stop()

    st.write("### Dữ liệu giá cuối cùng:")
    st.dataframe(df.tail())

    st.write("### Biểu đồ RRG:")
    rrg_df = compute_rrg_series(df, benchmark_symbol, trail_days=30)
    if rrg_df is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(100, color='gray', linestyle='--')
        ax.axvline(100, color='gray', linestyle='--')

        for symbol, data in rrg_df.groupby("symbol"):
            ax.plot(data["RS-Ratio"], data["RS-Momentum"], marker='o', markersize=4, label=symbol)
            ax.text(data["RS-Ratio"].iloc[-1] + 0.5, data["RS-Momentum"].iloc[-1], symbol)

        ax.set_xlabel("RS-Ratio (Relative Strength)")
        ax.set_ylabel("RS-Momentum (Momentum of RS)")
        ax.set_title("Relative Rotation Graph (RRG) - Trailing 30 days")
        ax.legend()
        st.pyplot(fig)

