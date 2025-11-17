import streamlit as st
import rrg_gemini
# import rrg_deep
import db_connector

# =====================
# SETUP VÀ KHỞI TẠO
# =====================

def initialize_app_services():
    """Khởi tạo kết nối DB và thiết lập bảng."""
    # 1. Kết nối DB
    conn = db_connector.get_db_connection()
    
    # 2. Thiết lập bảng (Chỉ chạy một lần sau khi kết nối)
    if conn:
        db_connector.setup_tables(conn)
        
    return conn

# =====================
# MAIN APPLICATION ROUTER
# =====================

def main():
    """Hàm main điều hướng ứng dụng Streamlit."""
    
    st.set_page_config(layout="wide", page_title="RRG & Stock Screener App")
    
    # Khởi tạo kết nối DB (đã được cache resource)
    conn = initialize_app_services()
    
    # 1. Định nghĩa các trang
    PAGES = {
        "Phân Tích RRG": lambda: rrg_gemini.rrg_analyzer_page(conn),
        "Cập Nhật Khuyến Nghị": lambda: recommendation_page.recommendation_tracker_page(conn)
    }

    # 2. Tạo Sidebar Navigation
    st.sidebar.title("🛠️ Menu Ứng Dụng")
    
    selection = st.sidebar.radio("Chọn Chức Năng", list(PAGES.keys()))
    
    # 3. Hiển thị trang được chọn
    page_function = PAGES[selection]
    page_function()

if __name__ == '__main__':
    # Thiết lập biến môi trường giả lập (CHỈ CHO MỤC ĐÍCH DEMO TRÊN CANVAS)
    # Vui lòng thay thế bằng chuỗi kết nối Neon.tech thực tế của bạn
    # Ví dụ: postgresql://user:password@host.neon.tech/database_name
    # os.environ["DB_URL"] = "postgresql://user:password@host:port/database" 
    
    main()