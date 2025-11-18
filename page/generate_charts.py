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


def calculate_rrg_data(price_df: pd.DataFrame, benchmark_symbol: str, period: int, scale_factor: float) -> pd.DataFrame:
    """Tính toán RRG dựa trên thuật toán trong rrg_app_gemini."""
    if price_df.empty or benchmark_symbol not in price_df.columns:
        return pd.DataFrame()

    df = price_df.copy()
    symbols = [col for col in df.columns if col != benchmark_symbol]
    results = []

    for symbol in symbols:
        rs_ratio = df[symbol] / df[benchmark_symbol]
        rs_ratio_wma = rs_ratio.ewm(span=period, adjust=False).mean()
        rs_momentum = rs_ratio_wma.pct_change(periods=period)

        rs_ratio_std = rs_ratio_wma.std()
        rs_momentum_std = rs_momentum.std()

        if rs_ratio_std == 0 or rs_momentum_std == 0:
            continue

        rs_ratio_z = (rs_ratio_wma - rs_ratio_wma.mean()) / rs_ratio_std
        rs_momentum_z = (rs_momentum - rs_momentum.mean()) / rs_momentum_std

        temp_df = pd.DataFrame(
            {
                "date": df.index,
                "symbol": symbol,
                "rs_ratio_scaled": 100 + rs_ratio_z * scale_factor,
                "rs_momentum_scaled": 100 + rs_momentum_z * scale_factor,
            }
        ).dropna()

        results.append(temp_df)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def plot_rrg_time_series(rrg_df: pd.DataFrame, symbol: str, benchmark: str, period: int):
    """Vẽ biểu đồ RRG giống phiên bản trong rrg_app_gemini."""
    df_symbol = rrg_df[rrg_df["symbol"] == symbol].copy().reset_index(drop=True)
    if df_symbol.empty:
        st.warning(f"Không có dữ liệu RRG cho {symbol}.")
        return

    rs = df_symbol["rs_ratio_scaled"]
    rm = df_symbol["rs_momentum_scaled"]
    if rs.empty or rm.empty:
        st.warning(f"Dữ liệu RRG không đủ để vẽ biểu đồ cho {symbol}.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(100, color="gray", linestyle="--", linewidth=0.8)

    min_val = min(rs.min(), rm.min(), 98)
    max_val = max(rs.max(), rm.max(), 102)
    padding = max((max_val - min_val) * 0.1, 1)
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)

    ax.plot(rs, rm, color="#1f77b4", linewidth=2, alpha=0.8)
    ax.scatter(rs.iloc[0], rm.iloc[0], color="#1f77b4", s=80, alpha=0.7, marker="o")
    ax.scatter(rs.iloc[-1], rm.iloc[-1], color="black", s=120, marker=">", zorder=5)
    ax.text(rs.iloc[-1], rm.iloc[-1], symbol, fontsize=12, ha="left", va="bottom")

    ax.set_title(f"RRG Time Series: {symbol} vs {benchmark} (P={period})", fontsize=13)
    ax.set_xlabel("Relative Strength (RS Ratio)")
    ax.set_ylabel("Relative Momentum (RM Momentum)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

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

    margin_symbols = db_connector.fetch_symbols_by_margin(conn, margin_value=1)
    if margin_symbols:
        st.info(f"Hiện có **{len(margin_symbols)}** mã được gắn cờ margin = 1.")
    else:
        st.warning("Chưa có mã nào được gắn margin = 1.")

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        return

    if st.button("Generate charts", type="primary"):
        if not margin_symbols:
            st.info("Không có mã nào để tạo biểu đồ. Vui lòng cập nhật cờ margin trước.")
            return

        str_start = start_date.strftime("%Y-%m-%d")
        str_end = end_date.strftime("%Y-%m-%d")

        for symbol in margin_symbols:
            st.markdown(f"### {symbol}")
            with st.spinner(f"Đang tạo biểu đồ cho {symbol}..."):
                price_df = db_connector.fetch_price_data(conn, [symbol, BENCHMARK_SYMBOL], str_start, str_end)
                if price_df.empty or symbol not in price_df.columns:
                    st.warning(f"Không tìm thấy dữ liệu giá cho {symbol} hoặc {BENCHMARK_SYMBOL}.")
                    continue

                rrg_df = calculate_rrg_data(price_df, BENCHMARK_SYMBOL, RRG_PERIOD, SCALE_FACTOR)
                if rrg_df.empty:
                    st.warning(f"Dữ liệu RRG chưa đủ để vẽ biểu đồ cho {symbol}.")
                    continue

                plot_rrg_time_series(rrg_df, symbol, BENCHMARK_SYMBOL, RRG_PERIOD)

