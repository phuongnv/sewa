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
#  LOAD .ENV CONFIG
# =====================
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

DB_CONN = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# =====================
#  PAGE CONFIG
# =====================
st.set_page_config(page_title="RRG Chart — Fast & Smooth", layout="wide")

# =====================
#  ABSTRACT DATA SOURCE
# =====================
class DataSource(ABC):
    @abstractmethod
    def get_data(self, symbols, start_date, end_date) -> pd.DataFrame:
        pass

# =====================
#  CUSTOM DB SOURCE
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
                st.warning(f"Không thể khởi tạo kết nối DB: {e}")

    def get_data(self, symbols, start_date, end_date):
        if self.engine is None:
            raise ConnectionError("❌ Chưa có connection_string hợp lệ cho database.")

        placeholders = ",".join([f"'{s}'" for s in symbols])
        query = f"""
            SELECT symbol, date, close
            FROM stock_prices
            WHERE symbol IN ({placeholders})
              AND date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date ASC
        """

        df = pd.read_sql(query, self.engine)
        if df.empty:
            st.warning("⚠️ Không có dữ liệu trong khoảng thời gian này.")
            return pd.DataFrame()

        df_pivot = df.pivot(index="date", columns="symbol", values="close")
        df_pivot.index = pd.to_datetime(df_pivot.index)
        return df_pivot.sort_index()

# =====================
#  RRG COMPUTATION
# =====================
def compute_rrg_series(df, benchmark_symbol, n=10, m=10, trail_days=30):
    if benchmark_symbol not in df.columns:
        st.error(f"Không tìm thấy mã chuẩn '{benchmark_symbol}' trong dữ liệu.")
        return None

    benchmark = df[benchmark_symbol]
    rs = df.divide(benchmark, axis=0)

    # 🔸 Theo chuẩn FireAnt / StockCharts
    rs_ema_n = rs.ewm(span=n).mean()
    rs_ema_2n = rs.ewm(span=2 * n).mean()
    rs_ratio = 100 + 10 * (rs_ema_n - rs_ema_2n) / rs_ema_2n

    rs_mom_n = rs_ratio.ewm(span=m).mean()
    rs_mom_2m = rs_ratio.ewm(span=2 * m).mean()
    rs_momentum = 100 + 10 * (rs_mom_n - rs_mom_2m) / rs_mom_2m

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
    return rrg_df.groupby("symbol").tail(trail_days).dropna()

# =====================
#  DRAW RRG
# =====================
def draw_rrg(rrg_df, title="RRG Chart", figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(100, color="gray", linestyle="--")
    ax.axvline(100, color="gray", linestyle="--")

    for symbol, data in rrg_df.groupby("symbol"):
        ax.plot(data["RS-Ratio"], data["RS-Momentum"], marker="o", markersize=3, label=symbol)
        ax.text(data["RS-Ratio"].iloc[-1] + 0.5, data["RS-Momentum"].iloc[-1], symbol, fontsize=9)

    ax.set_xlabel("RS-Ratio (Relative Strength)")
    ax.set_ylabel("RS-Momentum (Momentum of RS)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig

# =====================
#  STREAMLIT UI
# =====================
st.title("📊 Relative Rotation Graph (RRG) — Fast vs Smooth")

col1, col2 = st.columns(2)
with col1:
    symbols_input = st.text_input(
        "Danh sách mã cổ phiếu (phân tách bởi dấu phẩy)",
        value="AAA,ACB,VCB,VNM,FPT"
    )
with col2:
    benchmark_symbol = st.text_input("Mã chuẩn (benchmark)", value="VNINDEX")

start_date = st.date_input("Ngày bắt đầu", datetime.today() - timedelta(days=180))
end_date = st.date_input("Ngày kết thúc", datetime.today())

# Tham số điều chỉnh
st.markdown("### ⚙️ Tham số điều chỉnh")
col_fast, col_smooth = st.columns(2)

with col_fast:
    st.subheader("🚀 Fast RRG")
    fast_n = st.slider("RS-Ratio (n)", 5, 40, 10, step=1, key="fast_n")
    fast_m = st.slider("RS-Momentum (m)", 5, 40, 10, step=1, key="fast_m")
    fast_trail = st.slider("Số ngày hiển thị trail", 10, 90, 30, step=5, key="fast_trail")

with col_smooth:
    st.subheader("🌊 Smooth RRG")
    smooth_n = st.slider("RS-Ratio (n)", 5, 60, 20, step=1, key="smooth_n")
    smooth_m = st.slider("RS-Momentum (m)", 5, 60, 20, step=1, key="smooth_m")
    smooth_trail = st.slider("Số ngày hiển thị trail", 10, 90, 30, step=5, key="smooth_trail")

if st.button("📈 Tải & Tính RRG"):
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

    with st.spinner("🔹 Đang tải dữ liệu từ database Neon.tech ..."):
        try:
            source = CustomDBSource(DB_CONN)
            df = source.get_data(symbols + [benchmark_symbol], start_date, end_date)
            if df.empty:
                st.stop()
            st.success("✅ Dữ liệu tải thành công!")
        except Exception as e:
            st.error(f"Lỗi khi truy vấn database: {e}")
            st.stop()

    st.write("### 📋 Dữ liệu mẫu:")
    st.dataframe(df.tail())

    # Hai RRG song song
    rrg_fast = compute_rrg_series(df, benchmark_symbol, n=fast_n, m=fast_m, trail_days=fast_trail)
    rrg_smooth = compute_rrg_series(df, benchmark_symbol, n=smooth_n, m=smooth_m, trail_days=smooth_trail)

    colA, colB = st.columns(2)
    with colA:
        if rrg_fast is not None:
            st.pyplot(draw_rrg(rrg_fast, title=f"🚀 Fast RRG (n={fast_n}, m={fast_m})"))
    with colB:
        if rrg_smooth is not None:
            st.pyplot(draw_rrg(rrg_smooth, title=f"🌊 Smooth RRG (n={smooth_n}, m={smooth_m})"))
