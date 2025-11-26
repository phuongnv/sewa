import streamlit as st
import psycopg2
import os
import pandas as pd
from datetime import datetime

# =====================
# CẤU HÌNH VÀ KẾT NỐI DB
# =====================

def _is_connection_closed(conn):
    """Kiểm tra xem kết nối có bị đóng hay không."""
    if conn is None:
        return True
    try:
        # Kiểm tra trạng thái kết nối bằng cách thử thực hiện một truy vấn đơn giản
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return False
    except (psycopg2.InterfaceError, psycopg2.OperationalError, psycopg2.ProgrammingError):
        return True

def _create_new_connection():
    """Tạo kết nối mới đến database."""
    DB_URL = os.environ.get("DATABASE_URL") 
    
    if not DB_URL:
        DB_URL = st.secrets.get("DATABASE_URL") 
        if not DB_URL:
            st.error("Lỗi: Không tìm thấy chuỗi kết nối DB (DB_URL).")
            return None
    
    try:
        return psycopg2.connect(DB_URL)
    except Exception as e:
        st.error(f"Lỗi kết nối PostgreSQL: {e}")
        return None

def get_db_connection():
    """
    Thiết lập và trả về kết nối PostgreSQL.
    Sử dụng các biến môi trường để lấy chuỗi kết nối.
    Tự động reconnect nếu kết nối bị đóng.
    """
    # Sử dụng st.cache_resource để chỉ kết nối một lần
    # @st.cache_resource
    def connect_db():
        return _create_new_connection()
    
    conn = connect_db()
    
    # Kiểm tra và reconnect nếu cần
    if conn and _is_connection_closed(conn):
        # Xóa cache và tạo kết nối mới
        try:
            connect_db.clear()
        except Exception:
            pass  # Ignore errors when clearing cache
        conn = connect_db()
    
    return conn

def ensure_connection(conn):
    """
    Đảm bảo kết nối còn hoạt động, reconnect nếu cần.
    Trả về kết nối hợp lệ.
    """
    if _is_connection_closed(conn):
        return get_db_connection()
    return conn

def setup_tables(conn):
    """Tạo các bảng cần thiết nếu chúng chưa tồn tại."""
    if not conn:
        return
    
    # Đảm bảo kết nối còn hoạt động
    conn = ensure_connection(conn)
    if not conn:
        return
    
    try:
        cur = conn.cursor()
        
        # 1. Bảng lưu trữ khuyến nghị từ TradingView
        # 'symbol' là khóa chính
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                symbol VARCHAR(10) PRIMARY KEY,
                recommend VARCHAR(50) NOT NULL,
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
    
    # Đảm bảo kết nối còn hoạt động
    conn = ensure_connection(conn)
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
                INSERT INTO stock_info (symbol, recommend, last_updated)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET recommend = EXCLUDED.recommend,
                    last_updated = EXCLUDED.last_updated;
                """,
                (symbol, data['recommend'], current_time)
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
    
    # Đảm bảo kết nối còn hoạt động
    conn = ensure_connection(conn)
    if not conn:
        return pd.DataFrame()
    
    try:
        query = "SELECT symbol AS \"Mã CK\", RECOMMEND AS \"Trạng thái (Recommend)\", last_updated AS \"Cập nhật cuối\" FROM stock_info ORDER BY symbol ASC;"
        df = pd.read_sql(query, conn)
        # Chuyển đổi timestamp sang string cho hiển thị
        df['Cập nhật cuối'] = df['Cập nhật cuối'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu khuyến nghị DB: {e}")
        return pd.DataFrame()

def fetch_symbols_by_margin(conn, margin_value: int = 40) -> list:
    """Lấy danh sách symbol có margin bằng margin_value."""
    if not conn:
        return []
    
    # Đảm bảo kết nối còn hoạt động
    conn = ensure_connection(conn)
    if not conn:
        return []

def fetch_symbols_by_filters(
    conn,
    use_margin: bool = False,
    margin_value: int = 1,
    use_recommend: bool = False,
    recommend_value: str = "1",
) -> list:
    """Lấy danh sách symbol theo các bộ lọc margin và recommend."""
    if not conn:
        return []

    conn = ensure_connection(conn)
    if not conn:
        return []

    try:
        query = "SELECT symbol FROM stock_info"
        conditions = []
        params = []

        if use_margin:
            conditions.append("margin >= %s")
            params.append(margin_value)

        if use_recommend:
            conditions.append("recommend = %s")
            params.append(recommend_value)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY symbol ASC;"

        if params:
            df = pd.read_sql(query, conn, params=tuple(params))
        else:
            df = pd.read_sql(query, conn)

        if df.empty:
            return []
        return df["symbol"].tolist()
    except Exception as e:
        st.error(f"Lỗi khi lọc danh sách symbol: {e}")
        return []
    
    try:
        query = """
            SELECT symbol
            FROM stock_info
            WHERE margin >= %s
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

    # Đảm bảo kết nối còn hoạt động
    conn = ensure_connection(conn)
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
            SELECT symbol, date, close
            FROM stock_prices
            WHERE symbol IN %s AND date BETWEEN %s AND %s
            ORDER BY date, symbol;
        """
        
        df = pd.read_sql(query, conn, params=(symbol_list, start_date, end_date))
        
        if df.empty:
            return pd.DataFrame()

        # Chuyển đổi từ định dạng dài (long) sang định dạng rộng (wide)
        # price_df = df.pivot(index='date', columns='symbol', values='close')
        # price_df.index = pd.to_datetime(price_df.index)
        df["date"] = pd.to_datetime(df["date"])
        return df
        
    except Exception as e:
        st.error(f"Lỗi khi lấy dữ liệu giá DB: {e}")
        return pd.DataFrame()
    
# --- END OF db_connector.py ---