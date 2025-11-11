import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import sqlalchemy
from dotenv import load_dotenv
import os
from scipy.interpolate import CubicSpline, make_interp_spline

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
st.set_page_config(page_title="RRG Chart — Volume Filter", layout="wide")

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
#  RRG CALCULATION
# =====================
def calculate_rrg_data(df, benchmark_symbol='VNINDEX', period=21):
    """
    Tính toán dữ liệu cho RRG chart
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])
    
    # Tính log return
    df['log_return'] = np.log(df['close'] / df.groupby('symbol')['close'].shift(1))
    
    # Tạo dataframe cho benchmark
    benchmark_df = df[df['symbol'] == benchmark_symbol][['date', 'close', 'log_return']].copy()
    benchmark_df = benchmark_df.rename(columns={'close': 'benchmark_close', 'log_return': 'benchmark_return'})
    
    # Merge với benchmark
    df = df.merge(benchmark_df[['date', 'benchmark_close', 'benchmark_return']], on='date', how='left')
    
    # Tính relative price ratio
    df['price_ratio'] = df['close'] / df['benchmark_close']
    
    # Tính JdK RS-Ratio và RS-Momentum
    df['rs_ratio'] = (df['price_ratio'] / df.groupby('symbol')['price_ratio'].transform(lambda x: x.rolling(period).mean())) * 100
    df['rs_momentum'] = (df['price_ratio'] / df.groupby('symbol')['price_ratio'].shift(period)) * 100
    
    # Loại bỏ các dòng có giá trị NaN
    df = df.dropna(subset=['rs_ratio', 'rs_momentum'])
    
    return df

# =====================
#  VOLUME FILTER FUNCTIONS
# =====================
def calculate_volume_stats(df, window=10):
    """
    Tính volume trung bình 10 ngày cho mỗi mã
    """
    # Lọc dữ liệu 10 ngày gần nhất
    latest_date = df['date'].max()
    start_date = latest_date - timedelta(days=window * 2)  # Lấy thêm dữ liệu để tính trung bình
    
    recent_data = df[df['date'] >= start_date].copy()
    
    # Tính volume trung bình 10 ngày cho mỗi symbol
    volume_stats = recent_data.groupby('symbol').apply(
        lambda x: x.nlargest(window, 'date')['volume'].mean()
    ).reset_index()
    volume_stats.columns = ['symbol', 'avg_volume_10d']
    
    return volume_stats

def filter_symbols_by_volume(df, min_volume):
    """
    Lọc các mã có volume trung bình 10 ngày >= min_volume
    """
    volume_stats = calculate_volume_stats(df)
    filtered_symbols = volume_stats[volume_stats['avg_volume_10d'] >= min_volume]['symbol'].tolist()
    
    # Loại bỏ VNINDEX khỏi danh sách filtered
    filtered_symbols = [s for s in filtered_symbols if s != 'VNINDEX']
    
    return filtered_symbols, volume_stats

# =====================
#  DYNAMIC RANGE CALCULATION
# =====================
def calculate_dynamic_limits(rrg_df, selected_symbols, days_back=30, padding_ratio=0.1):
    """
    Tính toán giới hạn trục động dựa trên dữ liệu thực tế
    """
    # Lấy ngày cuối cùng và tính ngày bắt đầu
    latest_date = rrg_df['date'].max()
    start_date = latest_date - timedelta(days=days_back)
    
    # Lọc dữ liệu trong khoảng thời gian
    time_filtered_data = rrg_df[
        (rrg_df['date'] >= start_date) & 
        (rrg_df['date'] <= latest_date) &
        (rrg_df['symbol'].isin(selected_symbols))
    ]
    
    if time_filtered_data.empty:
        return 80, 120, 80, 120  # Default limits
    
    # Tính min/max của dữ liệu
    min_ratio = time_filtered_data['rs_ratio'].min()
    max_ratio = time_filtered_data['rs_ratio'].max()
    min_momentum = time_filtered_data['rs_momentum'].min()
    max_momentum = time_filtered_data['rs_momentum'].max()
    
    # Tính range và thêm padding
    ratio_range = max_ratio - min_ratio
    momentum_range = max_momentum - min_momentum
    
    # Đảm bảo range tối thiểu
    min_range = 20  # Minimum range to display
    ratio_range = max(ratio_range, min_range)
    momentum_range = max(momentum_range, min_range)
    
    # Tính limits với padding
    padding_x = ratio_range * padding_ratio
    padding_y = momentum_range * padding_ratio
    
    x_min = min_ratio - padding_x
    x_max = max_ratio + padding_x
    y_min = min_momentum - padding_y
    y_max = max_momentum + padding_y
    
    # Đảm bảo giới hạn hợp lý
    x_min = max(x_min, 50)   # Không quá thấp
    x_max = min(x_max, 150)  # Không quá cao
    y_min = max(y_min, 50)
    y_max = min(y_max, 150)
    
    return x_min, x_max, y_min, y_max

def calculate_quadrant_positions(x_min, x_max, y_min, y_max):
    """
    Tính vị trí quadrant labels dựa trên giới hạn động
    """
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Vị trí quadrant labels (điều chỉnh theo kích thước chart)
    positions = {
        'leading': (x_min + x_range * 0.75, y_min + y_range * 0.75),
        'weakening': (x_min + x_range * 0.75, y_min + y_range * 0.25),
        'lagging': (x_min + x_range * 0.25, y_min + y_range * 0.25),
        'improving': (x_min + x_range * 0.25, y_min + y_range * 0.75)
    }
    
    return positions

# =====================
#  SMOOTHING FUNCTIONS
# =====================
def smooth_trajectory(x, y, method='cubic', num_points=100):
    """
    Làm mịn đường trajectory sử dụng spline interpolation
    """
    if len(x) < 3:
        return x, y
    
    try:
        # Tạo parameter t (cumulative distance)
        t = np.zeros(len(x))
        for i in range(1, len(x)):
            t[i] = t[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
        
        # Chuẩn hóa t về [0, 1]
        t = t / t[-1]
        
        if method == 'cubic':
            # Cubic spline interpolation
            cs_x = CubicSpline(t, x)
            cs_y = CubicSpline(t, y)
        else:
            # B-spline interpolation
            cs_x = make_interp_spline(t, x, k=min(3, len(x)-1))
            cs_y = make_interp_spline(t, y, k=min(3, len(x)-1))
        
        # Tạo points mới
        t_new = np.linspace(0, 1, num_points)
        x_smooth = cs_x(t_new)
        y_smooth = cs_y(t_new)
        
        return x_smooth, y_smooth
    
    except Exception as e:
        # Fallback: return original data if smoothing fails
        return x, y

# =====================
#  RRG CHART FUNCTIONS
# =====================
def create_rrg_timeseries_chart(rrg_df, selected_symbols, days_back=30, figsize=(12, 8)):
    """
    Vẽ RRG chart với đường nối các điểm theo thời gian (original)
    """
    # Tính giới hạn động
    x_min, x_max, y_min, y_max = calculate_dynamic_limits(rrg_df, selected_symbols, days_back)
    quadrant_positions = calculate_quadrant_positions(x_min, x_max, y_min, y_max)
    
    # Lấy ngày cuối cùng và tính ngày bắt đầu
    latest_date = rrg_df['date'].max()
    start_date = latest_date - timedelta(days=days_back)
    
    # Lọc dữ liệu trong khoảng thời gian
    time_filtered_data = rrg_df[
        (rrg_df['date'] >= start_date) & 
        (rrg_df['date'] <= latest_date) &
        (rrg_df['symbol'].isin(selected_symbols))
    ].copy()
    
    # Sắp xếp theo ngày
    time_filtered_data = time_filtered_data.sort_values('date')
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Vẽ quadrant lines (tại 100, 100)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Vẽ quadrant labels với vị trí động
    quadrants = ['Leading', 'Weakening', 'Lagging', 'Improving']
    quadrant_colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightblue']
    
    for quadrant, color, pos_key in zip(quadrants, quadrant_colors, ['leading', 'weakening', 'lagging', 'improving']):
        x_pos, y_pos = quadrant_positions[pos_key]
        ax.text(x_pos, y_pos, quadrant, fontsize=11, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # Màu sắc cho các symbol
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_symbols)))
    color_dict = {symbol: color for symbol, color in zip(selected_symbols, colors)}
    
    # Vẽ cho từng symbol
    for symbol in selected_symbols:
        symbol_data = time_filtered_data[time_filtered_data['symbol'] == symbol]
        
        if len(symbol_data) > 0:
            # Vẽ đường nối các điểm theo thời gian (original)
            ax.plot(symbol_data['rs_ratio'], symbol_data['rs_momentum'], 
                   color=color_dict[symbol], alpha=0.6, linewidth=2, 
                   label=symbol, marker='')
            
            # Điểm đầu
            first_point = symbol_data.iloc[0]
            ax.scatter(first_point['rs_ratio'], first_point['rs_momentum'], 
                      color=color_dict[symbol], s=80, alpha=0.8, marker='o')
            
            # Điểm cuối (ngày gần nhất)
            last_point = symbol_data.iloc[-1]
            ax.scatter(last_point['rs_ratio'], last_point['rs_momentum'], 
                      color=color_dict[symbol], s=120, alpha=1.0, marker='*', 
                      edgecolor='black', linewidth=1)
            ax.annotate(f"{symbol}", 
                       (last_point['rs_ratio'], last_point['rs_momentum']),
                       xytext=(10, 10), textcoords='offset points', fontsize=9,
                       alpha=1.0, weight='bold')
    
    # Thiết lập chart với giới hạn động
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('RS-Ratio (Relative Strength)', fontsize=12, weight='bold')
    ax.set_ylabel('RS-Momentum', fontsize=12, weight='bold')
    
    date_range_str = f"{start_date.strftime('%d/%m/%Y')} - {latest_date.strftime('%d/%m/%Y')}"
    ax.set_title(f'RRG Time Series - Original ({days_back} ngày)\nRange: RS-Ratio [{x_min:.1f}-{x_max:.1f}], RS-Momentum [{y_min:.1f}-{y_max:.1f}]', 
                 fontsize=12, weight='bold', pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    return fig, time_filtered_data

def create_smoothed_rrg_chart(rrg_df, selected_symbols, days_back=30, smoothing_method='cubic', figsize=(12, 8)):
    """
    Vẽ RRG chart với đường trajectory được làm mịn giống Julius RRG
    """
    # Tính giới hạn động
    x_min, x_max, y_min, y_max = calculate_dynamic_limits(rrg_df, selected_symbols, days_back)
    quadrant_positions = calculate_quadrant_positions(x_min, x_max, y_min, y_max)
    
    # Lấy ngày cuối cùng và tính ngày bắt đầu
    latest_date = rrg_df['date'].max()
    start_date = latest_date - timedelta(days=days_back)
    
    # Lọc dữ liệu trong khoảng thời gian
    time_filtered_data = rrg_df[
        (rrg_df['date'] >= start_date) & 
        (rrg_df['date'] <= latest_date) &
        (rrg_df['symbol'].isin(selected_symbols))
    ].copy()
    
    # Sắp xếp theo ngày
    time_filtered_data = time_filtered_data.sort_values('date')
    
    # Tạo figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Vẽ quadrant lines
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    
    # Vẽ quadrant labels với vị trí động
    quadrants = ['Leading', 'Weakening', 'Lagging', 'Improving']
    quadrant_colors = ['#90EE90', '#FFFACD', '#FFB6C1', '#ADD8E6']
    
    for quadrant, color, pos_key in zip(quadrants, quadrant_colors, ['leading', 'weakening', 'lagging', 'improving']):
        x_pos, y_pos = quadrant_positions[pos_key]
        ax.text(x_pos, y_pos, quadrant, fontsize=10, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8,
                         edgecolor='gray', linewidth=0.5))
    
    # Màu sắc cho các symbol
    colors = plt.cm.Dark2(np.linspace(0, 1, len(selected_symbols)))
    color_dict = {symbol: color for symbol, color in zip(selected_symbols, colors)}
    
    # Vẽ cho từng symbol với đường làm mịn
    for symbol in selected_symbols:
        symbol_data = time_filtered_data[time_filtered_data['symbol'] == symbol]
        
        if len(symbol_data) >= 3:  # Cần ít nhất 3 điểm để làm mịn
            x_original = symbol_data['rs_ratio'].values
            y_original = symbol_data['rs_momentum'].values
            
            # Làm mịn trajectory
            x_smooth, y_smooth = smooth_trajectory(x_original, y_original, 
                                                  method=smoothing_method, 
                                                  num_points=100)
            
            # Vẽ đường làm mịn
            ax.plot(x_smooth, y_smooth, 
                   color=color_dict[symbol], alpha=0.8, linewidth=3,
                   label=symbol, solid_capstyle='round')
            
            # Vẽ points gốc (nhỏ, mờ)
            ax.scatter(x_original, y_original, 
                      color=color_dict[symbol], s=30, alpha=0.3, marker='o')
            
            # Điểm đầu
            ax.scatter(x_smooth[0], y_smooth[0], 
                      color=color_dict[symbol], s=100, alpha=0.9, marker='o',
                      edgecolor='black', linewidth=1.5)
            
            # Điểm cuối với mũi tên
            ax.scatter(x_smooth[-1], y_smooth[-1], 
                      color=color_dict[symbol], s=150, alpha=1.0, marker='>',
                      edgecolor='black', linewidth=2)
            
            # Hiển thị tên ở điểm cuối
            ax.annotate(f"{symbol}", 
                       (x_smooth[-1], y_smooth[-1]),
                       xytext=(12, 12), textcoords='offset points', 
                       fontsize=10, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', 
                                alpha=0.8, edgecolor=color_dict[symbol]))
            
            # Thêm mũi tên chỉ hướng di chuyển
            if len(x_smooth) > 20:
                # Vị trí 1/3
                idx1 = len(x_smooth) // 3
                ax.annotate('', xy=(x_smooth[idx1+1], y_smooth[idx1+1]), 
                           xytext=(x_smooth[idx1], y_smooth[idx1]),
                           arrowprops=dict(arrowstyle='->', color=color_dict[symbol], 
                                         alpha=0.6, lw=1.5))
                
                # Vị trí 2/3
                idx2 = 2 * len(x_smooth) // 3
                ax.annotate('', xy=(x_smooth[idx2+1], y_smooth[idx2+1]), 
                           xytext=(x_smooth[idx2], y_smooth[idx2]),
                           arrowprops=dict(arrowstyle='->', color=color_dict[symbol], 
                                         alpha=0.6, lw=1.5))
        
        else:
            # Nếu không đủ điểm để làm mịn, vẽ đường thẳng bình thường
            ax.plot(symbol_data['rs_ratio'], symbol_data['rs_momentum'], 
                   color=color_dict[symbol], alpha=0.7, linewidth=2, 
                   label=symbol)
            
            first_point = symbol_data.iloc[0]
            last_point = symbol_data.iloc[-1]
            
            ax.scatter(first_point['rs_ratio'], first_point['rs_momentum'], 
                      color=color_dict[symbol], s=80, alpha=0.8, marker='o')
            ax.scatter(last_point['rs_ratio'], last_point['rs_momentum'], 
                      color=color_dict[symbol], s=100, alpha=1.0, marker='>')
    
    # Thiết lập chart với giới hạn động
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('RS-Ratio', fontsize=13, weight='bold', color='#333333')
    ax.set_ylabel('RS-Momentum', fontsize=13, weight='bold', color='#333333')
    
    date_range_str = f"{start_date.strftime('%d/%m/%Y')} - {latest_date.strftime('%d/%m/%Y')}"
    ax.set_title(f'Julius RRG Style - Smoothed Trajectory ({days_back} ngày)\nRange: RS-Ratio [{x_min:.1f}-{x_max:.1f}], RS-Momentum [{y_min:.1f}-{y_max:.1f}]', 
                 fontsize=12, weight='bold', pad=20, color='#333333')
    
    # Grid style
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f8f9fa')
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9, 
              edgecolor='gray', facecolor='white')
    
    plt.tight_layout()
    return fig, time_filtered_data

# =====================
#  STREAMLIT UI
# =====================
def main():
    st.title("📈 RRG Charts - Volume Filter & Interactive Selection")
    st.markdown("**Lọc cổ phiếu theo volume và chọn interactive để hiển thị trên biểu đồ**")
    
    # Initialize session state for selected symbols
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = []
    
    if 'volume_filtered_symbols' not in st.session_state:
        st.session_state.volume_filtered_symbols = []
    
    if 'rrg_data' not in st.session_state:
        st.session_state.rrg_data = None
    
    # Sidebar controls
    st.sidebar.header("⚙️ Cài đặt tham số")
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date_input = st.date_input("Từ ngày", start_date)
    with col2:
        end_date_input = st.date_input("Đến ngày", end_date)
    
    # Parameters
    period = st.sidebar.slider("Chu kỳ RRG (ngày)", min_value=5, max_value=50, value=21)
    days_back = st.sidebar.slider("Số ngày hiển thị", min_value=10, max_value=90, value=30)
    
    # Volume filter settings
    st.sidebar.markdown("### 🔍 Lọc theo Volume")
    min_volume = st.sidebar.slider(
        "Volume trung bình 10 ngày tối thiểu", 
        min_value=10000, 
        max_value=1000000, 
        value=100000,
        step=10000,
        help="Chỉ hiển thị các mã có volume trung bình 10 ngày lớn hơn giá trị này"
    )
    
    # Dynamic range settings
    padding_ratio = st.sidebar.slider("Padding (%)", min_value=5, max_value=30, value=10) / 100
    
    # Smoothing parameters
    smoothing_method = st.sidebar.selectbox(
        "Phương pháp làm mịn",
        ["cubic", "bspline"],
        index=0
    )
    
    # Load data button - chỉ load data, không reset selection
    if st.sidebar.button("🔄 Tải dữ liệu mới"):
        with st.spinner("Đang tải dữ liệu từ database..."):
            try:
                # Initialize data source
                data_source = CustomDBSource(DB_CONN)
                
                # Get all symbols for volume filtering
                all_df = data_source.get_data(
                    symbols=None,  # Lấy tất cả symbols
                    start_date=start_date_input.strftime('%Y-%m-%d'),
                    end_date=end_date_input.strftime('%Y-%m-%d')
                )
                
                if all_df.empty:
                    st.error("❌ Không có dữ liệu cho khoảng thời gian đã chọn.")
                    return
                
                if 'VNINDEX' not in all_df['symbol'].unique():
                    st.error("❌ Không tìm thấy dữ liệu VNINDEX trong database.")
                    return
                
                # Calculate volume stats and filter symbols
                filtered_symbols, volume_stats = filter_symbols_by_volume(all_df, min_volume)
                st.session_state.volume_filtered_symbols = filtered_symbols
                
                # Calculate RRG data for filtered symbols (including VNINDEX)
                symbols_for_rrg = filtered_symbols + ['VNINDEX']
                rrg_df = calculate_rrg_data(all_df[all_df['symbol'].isin(symbols_for_rrg)], 'VNINDEX', period)
                st.session_state.rrg_data = rrg_df
                
                st.success(f"✅ Đã tải {len(rrg_df)} dòng dữ liệu. Tìm thấy {len(filtered_symbols)} mã thoả điều kiện volume.")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    # Hiển thị volume filtered symbols và selection interface
    if st.session_state.volume_filtered_symbols:
        st.sidebar.markdown("### 📋 Chọn mã hiển thị")
        
        # Hiển thị số lượng mã được chọn
        st.sidebar.info(f"**{len(st.session_state.selected_symbols)}** mã đang được chọn")
        
        # Hiển thị các mã được lọc theo volume dưới dạng clickable tags
        st.sidebar.markdown("#### Các mã thoả điều kiện volume:")
        
        # Tạo columns để hiển thị các tag
        cols = st.sidebar.columns(3)
        for idx, symbol in enumerate(st.session_state.volume_filtered_symbols):
            col_idx = idx % 3
            with cols[col_idx]:
                # Hiển thị tag, nếu click sẽ thêm vào selected symbols
                if st.button(symbol, key=f"add_{symbol}", use_container_width=True):
                    if symbol not in st.session_state.selected_symbols:
                        st.session_state.selected_symbols.append(symbol)
                        st.rerun()
        
        # Hiển thị danh sách mã đã chọn với option để xoá
        if st.session_state.selected_symbols:
            st.sidebar.markdown("#### Mã đang hiển thị:")
            for symbol in st.session_state.selected_symbols[:]:  # Copy list để tránh modification during iteration
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.write(f"**{symbol}**")
                with col2:
                    if st.button("❌", key=f"remove_{symbol}"):
                        st.session_state.selected_symbols.remove(symbol)
                        st.rerun()
            
            # Nút xoá tất cả
            if st.sidebar.button("🗑️ Xoá tất cả"):
                st.session_state.selected_symbols = []
                st.rerun()
    
    # Render charts button - chỉ render lại chart, không load data mới
    if st.sidebar.button("🎨 Vẽ/Render lại Biểu đồ") and st.session_state.rrg_data is not None and st.session_state.selected_symbols:
        with st.spinner("Đang vẽ biểu đồ..."):
            try:
                rrg_df = st.session_state.rrg_data
                selected_symbols = st.session_state.selected_symbols
                
                # Hiển thị range info
                x_min, x_max, y_min, y_max = calculate_dynamic_limits(rrg_df, selected_symbols, days_back, padding_ratio)
                st.info(f"**Đang hiển thị {len(selected_symbols)} mã** | RS-Ratio: {x_min:.1f}-{x_max:.1f} | RS-Momentum: {y_min:.1f}-{y_max:.1f}")
                
                # Hiển thị cả hai chart
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 RRG Original")
                    fig_original, original_data = create_rrg_timeseries_chart(
                        rrg_df, selected_symbols, days_back, figsize=(10, 8)
                    )
                    st.pyplot(fig_original)
                
                with col2:
                    st.subheader("🎯 RRG Smoothed (Julius Style)")
                    fig_smoothed, smoothed_data = create_smoothed_rrg_chart(
                        rrg_df, selected_symbols, days_back, smoothing_method, figsize=(10, 8)
                    )
                    st.pyplot(fig_smoothed)
                
                # Data summary
                with st.expander("📈 Thống kê dữ liệu"):
                    if not original_data.empty:
                        st.write("**Phạm vi dữ liệu thực tế:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("RS-Ratio Min", f"{original_data['rs_ratio'].min():.2f}")
                        with col2:
                            st.metric("RS-Ratio Max", f"{original_data['rs_ratio'].max():.2f}")
                        with col3:
                            st.metric("RS-Momentum Min", f"{original_data['rs_momentum'].min():.2f}")
                        with col4:
                            st.metric("RS-Momentum Max", f"{original_data['rs_momentum'].max():.2f}")
                
            except Exception as e:
                st.error(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")
    
    elif st.session_state.rrg_data is None:
        st.warning("⚠️ Vui lòng nhấn 'Tải dữ liệu mới' trước")
    elif not st.session_state.selected_symbols:
        st.warning("⚠️ Vui lòng chọn ít nhất một mã cổ phiếu để hiển thị")
    
    # Default instructions
    if not st.session_state.volume_filtered_symbols:
        st.info("""
        👈 **Hướng dẫn sử dụng:**
        
        **Bước 1:** Nhấn **"Tải dữ liệu mới"** để lấy dữ liệu từ database
        **Bước 2:** Chọn các mã cổ phiếu từ danh sách được lọc theo volume
        **Bước 3:** Nhấn **"Vẽ/Render lại Biểu đồ"** để hiển thị biểu đồ
        
        **Tính năng mới:**
        - 🔍 Lọc theo volume trung bình 10 ngày
        - 📋 Chọn mã interactive bằng cách click
        - ❌ Xoá mã bằng nút delete
        - 🎨 Render lại biểu đồ mà không cần tải lại dữ liệu
        """)

if __name__ == "__main__":
    main()