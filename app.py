import streamlit as st
import db_connector
from page.generate_charts import render as render_generate_charts
from page.margin_manager import render as render_margin_manager


def initialize_app_services():
    """Khởi tạo kết nối DB và thiết lập bảng."""
    conn = db_connector.get_db_connection()
    if conn:
        db_connector.setup_tables(conn)
    return conn


def render_header_navigation(pages):
    """Hiển thị navigation ở khu vực header."""
    if "active_page" not in st.session_state:
        st.session_state.active_page = next(iter(pages.keys()))

    st.title("RRG & Stock Screener")
    st.caption("Điều hướng tính năng trực tiếp từ Header.")

    selected = st.radio(
        "Điều hướng",
        list(pages.keys()),
        index=list(pages.keys()).index(st.session_state.active_page),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.session_state.active_page = selected
    st.markdown("---")


def main():
    st.set_page_config(layout="wide", page_title="RRG & Stock Screener App")
    conn = initialize_app_services()

    pages = {
        "Generate charts": lambda: render_generate_charts(conn),
        "Margin manager": lambda: render_margin_manager(conn),
    }

    render_header_navigation(pages)

    page_renderer = pages.get(st.session_state.active_page)
    if page_renderer:
        page_renderer()
    else:
        st.error("Không tìm thấy trang được chọn.")


if __name__ == "__main__":
    main()