import streamlit as st
import psycopg2
import os
import pandas as pd
from datetime import datetime

# =====================
# CẤU HÌNH VÀ KẾT NỐI DB
# =====================

def get_db_connection():
    """
    Thiết lập và trả về kết nối PostgreSQL.
    Sử dụng các biến môi trường để lấy chuỗi kết nối.
    """
    # Trong môi trường Canvas/Cloud, chuỗi kết nối thường được cung cấp qua
    # biến môi trường hoặc cấu hình bí mật. Ở đây ta giả lập một biến môi trường.
    # Vui lòng thay thế 'YOUR_NEON_POSTGRES_URL' bằng biến thực tế.
    
    # Giả định biến môi trường chứa chuỗi kết nối
    # Ví dụ: os.environ.get("DATABASE_URL")
    DB_URL = os.environ.get("DATABASE_URL") 
    
    if not DB_URL:
        st.error("Lỗi: Không tìm thấy chuỗi kết nối DB (DB_URL).")
        return None
    
    try:
        # Sử dụng st.cache_resource để chỉ kết nối một lần
        @st.cache_resource
        def connect_db(url):
            return psycopg2.connect(url)
        
        conn = connect_db(DB_URL)
        return conn
    
    except Exception as e:
        st.error(f"Lỗi kết nối PostgreSQL: {e}")
        return None

def setup_tables(conn):
    """Tạo các bảng cần thiết nếu chúng chưa tồn tại."""
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # 1. Bảng lưu trữ khuyến nghị từ TradingView
        # 'symbol' là khóa chính
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol VARCHAR(10) PRIMARY KEY,
                tradingview VARCHAR(50) NOT NULL,
                last_updated TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                margin INTEGER NOT NULL DEFAULT 0
            );
        """)
        cur.execute("""
            ALTER TABLE stock_info
            ADD COLUMN IF NOT EXISTS margin INTEGER NOT NULL DEFAULT 0;
        """)
        
        # 2. Bảng lưu trữ dữ liệu giá (thay thế yfinance)
        # Bảng này sẽ được dùng trong rrg_page.py
        
        
        conn.commit()
        # st.success("Cấu trúc bảng DB đã sẵn sàng.")
        
    except Exception as e:
        st.error(f"Lỗi thiết lập bảng DB: {e}")
        conn.rollback()


# =====================
# CHỨC NĂNG CẬP NHẬT (RECOMMENDATION)
# =====================

def update_stock_recommendations(conn, recommendations: dict):
    """
    Cập nhật trạng thái khuyến nghị vào bảng 'stock_info' bằng cách sử dụng UPSERT.
    Lưu ý: PostgreSQL 9.5+ hỗ trợ ON CONFLICT (UPSERT).
    """
    if not conn:
        return 0
    
    try:
        cur = conn.cursor()
        
        success_count = 0
        current_time = datetime.now()
        
        for symbol, data in recommendations.items():
            # Sử dụng ON CONFLICT (UPSERT) để cập nhật nếu đã tồn tại, hoặc chèn mới
            cur.execute(
                """
                INSERT INTO stock_info (symbol, tradingview, last_updated)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET tradingview = EXCLUDED.tradingview,
                    last_updated = EXCLUDED.last_updated;
                """,
                (symbol, data['tradingview'], current_time)
            )
            success_count += 1

        conn.commit()
        return success_count
    
    except Exception as e:
        st.error(f"Lỗi khi cập nhật khuyến nghị DB: {e}")
        conn.rollback()
        return 0

def fetch_all_recommendations(conn) -> pd.DataFrame:
    """Lấy tất cả dữ liệu khuyến nghị từ bảng 'stock_info'."""
    if not conn:
        return pd.DataFrame()
    
    try:
        query = "SELECT symbol AS \"Mã CK\", tradingview AS \"Trạng thái (TradingView)\", last_updated AS \"Cập nhật cuối\" FROM stock_info ORDER BY symbol ASC;"
        df = pd.read_sql(query, conn)
        # Chuyển đổi timestamp sang string cho hiển thị
        df['Cập nhật cuối'] = df['Cập nhật cuối'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu khuyến nghị DB: {e}")
        return pd.DataFrame()

def fetch_symbols_by_margin(conn, margin_value: int = 1) -> list:
    """Lấy danh sách symbol có margin bằng margin_value."""
    if not conn:
        return []
    try:
        query = """
            SELECT symbol
            FROM stock_info
            WHERE margin = %s
            ORDER BY symbol ASC;
        """
        df = pd.read_sql(query, conn, params=(margin_value,))
        if df.empty:
            return []
        return df['symbol'].tolist()
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách symbol theo margin: {e}")
        return []
# =====================
# CHỨC NĂNG DỮ LIỆU GIÁ (RRG)
# =====================

def fetch_price_data(conn, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Lấy dữ liệu giá Adj Close cho các mã chứng khoán trong một khoảng thời gian.
    (Giả định bạn đã nhập dữ liệu giá vào bảng price_data)
    """
    if not conn:
        return pd.DataFrame()

    try:
        # Chuẩn bị danh sách mã để đưa vào truy vấn SQL (tránh SQL Injection)
        symbol_list = tuple(symbols)
        
        # Nếu chỉ có 1 symbol, cần đảm bảo tuple có dấu phẩy: ('symbol',)
        if len(symbols) == 1:
            symbol_list = (symbols[0],)

        # Lấy dữ liệu giá
        query = f"""
            SELECT date, symbol, adj_close
            FROM price_data
            WHERE symbol IN %s AND date BETWEEN %s AND %s
            ORDER BY date, symbol;
        """
        
        df = pd.read_sql(query, conn, params=(symbol_list, start_date, end_date))
        
        if df.empty:
            return pd.DataFrame()

        # Chuyển đổi từ định dạng dài (long) sang định dạng rộng (wide)
        price_df = df.pivot(index='date', columns='symbol', values='adj_close')
        price_df.index = pd.to_datetime(price_df.index)
        
        return price_df
        
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu giá DB: {e}")
        return pd.DataFrame()
    
# --- END OF db_connector.py ---