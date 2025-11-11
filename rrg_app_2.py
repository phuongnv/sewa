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

    def get_data(self, symbols=None, start_date=None, end_date=None):
        if self.engine is None:
            raise ConnectionError("❌ Chưa có connection_string hợp lệ cho database.")

        where_clause = "1=1"
        if symbols:
            placeholders = ",".join([f"'{s}'" for s in symbols])
            where_clause += f" AND symbol IN ({placeholders})"
        if start_date and end_date:
            where_clause += f" AND date BETWEEN '{start_date}' AND '{end_date}'"

        query = f"""
            SELECT symbol, date, close, volume
            FROM stock_prices
            WHERE {where_clause}
            ORDER BY date ASC
        """

        df = pd.read_sql(query, self.engine)
        df["date"] = pd.to_datetime(df["date"])
        return df

# =====================
#  RRG COMPUTATION
# =====================
def compute_rrg_series(df, benchmark_symbol, n=10, m=10, trail_days=30):
    pivot = df.pivot(index="date", columns="symbol", values="close").sort_index()
    if benchmark_symbol not in pivot.columns:
        st.error(f"Không tìm thấy mã chuẩn '{benchmark_symbol}' trong dữ liệu.")
        return None

    benchmark = pivot[benchmark_symbol]
    rs = pivot.divide(benchmark, axis=0)

    rs_ema_n = rs.ewm(span=n).mean()
    rs_ema_2n = rs.ewm(span=2 * n).mean()
    rs_ratio = 100 + 10 * (rs_ema_n - rs_ema_2n) / rs_ema_2n

    rs_mom_n = rs_ratio.ewm(span=m).mean()
    rs_mom_2m = rs_ratio.ewm(span=2 * m).mean()
    rs_momentum = 100 + 10 * (rs_mom_n - rs_mom_2m) / rs_mom_2m

    long_df = []
    for sym in pivot.columns:
        if sym == benchmark_symbol:
            continue
        tmp = pd.DataFrame({
            "date": pivot.index,
            "symbol": sym,
            "RS-Momentum": rs_momentum[sym],
            "RS-Ratio": rs_ratio[sym]
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
        ax.text(data["RS-Ratio"].iloc[-1] + 0.01, data["RS-Momentum"].iloc[-1], symbol, fontsize=9)

        ax.scatter(data["RS-Ratio"].iloc[-1], data["RS-Momentum"].iloc[-1], s=80, color="red", edgecolors="black", zorder=6)

    ax.set_xlabel("RS-Ratio (Relative Strength)")
    ax.set_ylabel("RS-Momentum (Momentum of RS)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


# =====================
#  UI PHẦN 1 — INPUT TỰ NHẬP
# =====================
st.title("📊 Relative Rotation Graph (RRG) — Fast vs Smooth")

st.markdown("## 🔹 RRG tùy chỉnh bằng input thủ công")

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
df = None

with col_fast:
    st.subheader("🚀 Fast RRG")
    fast_n = st.slider("RS-Ratio (n)", 5, 40, 10, step=1)
    fast_m = st.slider("RS-Momentum (m)", 5, 40, 10, step=1)
    fast_trail = st.slider("Số ngày hiển thị trail", 10, 90, 30, step=5, key="fast_trail")
with col_smooth:
    st.subheader("🌊 Smooth RRG")
    smooth_n = st.slider("RS-Ratio (n)", 5, 60, 20, step=1)
    smooth_m = st.slider("RS-Momentum (m)", 5, 60, 20, step=1)
    smooth_trail = st.slider("Số ngày hiển thị trail", 10, 90, 30, step=5, key="smooth_trail")

if st.button("📈 Tải & Tính RRG (từ input thủ công)"):
    symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    source = CustomDBSource(DB_CONN)
    df = source.get_data(symbols + [benchmark_symbol], start_date, end_date)

    rrg_fast = compute_rrg_series(df, benchmark_symbol, n=fast_n, m=fast_m, trail_days=fast_trail)
    rrg_smooth = compute_rrg_series(df, benchmark_symbol, n=smooth_n, m=smooth_m, trail_days=smooth_trail)

    colA, colB = st.columns(2)
    with colA:
        if rrg_fast is not None:
            st.pyplot(draw_rrg(rrg_fast, title=f"🚀 Fast RRG (n={fast_n}, m={fast_m})"))
    with colB:
        if rrg_smooth is not None:
            st.pyplot(draw_rrg(rrg_smooth, title=f"🌊 Smooth RRG (n={smooth_n}, m={smooth_m})"))

st.write("### 📋 Dữ liệu mẫu:")
if df is not None:
    st.dataframe(df.tail())
else:
    st.write("❌ Không có dữ liệu")


# =====================
#  UI PHẦN 2 — DANH SÁCH MÃ TỪ DATABASE
# =====================
# st.markdown("---")
# st.markdown("## 🔹 Danh sách mã cổ phiếu từ Database (lọc theo Vol)")

# min_vol = st.slider("Lọc cổ phiếu có Vol trung bình > ", 
#                     min_value=0, max_value=2_000_000, value=500_000, step=50_000)

# try:
#     source = CustomDBSource(DB_CONN)
#     df_all = source.get_data(start_date=start_date, end_date=end_date)
#     avg_vol = df_all.groupby("symbol")["volume"].mean().sort_values(ascending=False)
#     filtered_symbols = avg_vol[avg_vol > min_vol].index.tolist()

#     if not filtered_symbols:
#         st.warning("⚠️ Không có mã nào đạt điều kiện khối lượng.")
#     else:
#         selected_symbol = st.selectbox("Chọn mã để hiển thị RRG:", filtered_symbols)
#         st.write(f"📊 Đang hiển thị dữ liệu cho: `{selected_symbol}`")

#         df_selected = df_all[df_all["symbol"].isin([selected_symbol, benchmark_symbol])]
#         rrg_fast = compute_rrg_series(df_selected, benchmark_symbol, n=fast_n, m=fast_m, trail_days=fast_trail)
#         rrg_smooth = compute_rrg_series(df_selected, benchmark_symbol, n=smooth_n, m=smooth_m, trail_days=smooth_trail)

#         colC, colD = st.columns(2)
#         with colC:
#             if rrg_fast is not None:
#                 st.pyplot(draw_rrg(rrg_fast, title=f"🚀 Fast RRG ({selected_symbol})"))
#         with colD:
#             if rrg_smooth is not None:
#                 st.pyplot(draw_rrg(rrg_smooth, title=f"🌊 Smooth RRG ({selected_symbol})"))

#         if (df_selected is not None):
#             st.write("### 📋 Dữ liệu mẫu:")
#             st.dataframe(df_selected.tail())
#         else:
#             st.write("❌ Không có dữ liệu")


# except Exception as e:
#     st.error(f"Lỗi khi truy vấn DB: {e}")

st.markdown("---")
st.markdown("## 🔹 Danh sách mã cổ phiếu từ Database (lọc theo Vol & tên)")

min_vol = st.slider(
    "Lọc cổ phiếu có Vol trung bình > ",
    min_value=0,
    max_value=2_000_000,
    value=500_000,
    step=50_000
)

try:
    source = CustomDBSource(DB_CONN)
    df_all = source.get_data(start_date=start_date, end_date=end_date)
    avg_vol = df_all.groupby("symbol")["volume"].mean().sort_values(ascending=False)
    filtered_symbols = avg_vol[avg_vol > min_vol].index.tolist()

    if not filtered_symbols:
        st.warning("⚠️ Không có mã nào đạt điều kiện khối lượng.")
    else:
        # --- Bộ lọc nhanh theo ký tự ---
        filter_text = st.text_input("🔍 Lọc mã theo ký tự (ví dụ: 'VN', 'ACB')", "").strip().upper()
        if filter_text:
            filtered_symbols = [s for s in filtered_symbols if filter_text in s]

        # Sắp xếp
        filtered_symbols = sorted(filtered_symbols)

        st.markdown("### 🏷️ Chọn mã để hiển thị RRG:")

        # --- Quản lý trạng thái ---
        if "selected_symbol" not in st.session_state:
            st.session_state["selected_symbol"] = filtered_symbols[0] if filtered_symbols else None

        selected_symbol = st.session_state["selected_symbol"]

        # --- Hiển thị danh sách mã theo hàng ngang ---
        num_per_row = 10  # số nút mỗi hàng
        rows = [filtered_symbols[i:i+num_per_row] for i in range(0, len(filtered_symbols), num_per_row)]

        for row_symbols in rows:
            cols = st.columns(len(row_symbols))
            for i, sym in enumerate(row_symbols):
                is_selected = sym == selected_symbol
                button_label = f"✅ {sym}" if is_selected else sym
                if cols[i].button(button_label, key=f"btn_{sym}"):
                    st.session_state["selected_symbol"] = sym
                    selected_symbol = sym

        st.write(f"📊 Đang hiển thị dữ liệu cho: `{selected_symbol}`")

        # --- Hiển thị chart ---
        df_selected = df_all[df_all["symbol"].isin([selected_symbol, benchmark_symbol])]
        rrg_fast = compute_rrg_series(df_selected, benchmark_symbol, n=fast_n, m=fast_m, trail_days=fast_trail)
        rrg_smooth = compute_rrg_series(df_selected, benchmark_symbol, n=smooth_n, m=smooth_m, trail_days=smooth_trail)

        colC, colD = st.columns(2)
        with colC:
            if rrg_fast is not None:
                st.pyplot(draw_rrg(rrg_fast, title=f"🚀 Fast RRG ({selected_symbol})"))
        with colD:
            if rrg_smooth is not None:
                st.pyplot(draw_rrg(rrg_smooth, title=f"🌊 Smooth RRG ({selected_symbol})"))

        if df_selected is not None:
            st.write("### 📋 Dữ liệu mẫu:")
            st.dataframe(df_selected.tail())
        else:
            st.write("❌ Không có dữ liệu")

except Exception as e:
    st.error(f"Lỗi khi truy vấn DB: {e}")
