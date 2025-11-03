import os
from contextlib import contextmanager

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus


def get_db_url() -> str:
    # Prefer Streamlit Cloud secrets; fall back to env vars for local/devcontainer.
    if "postgres" in st.secrets:
        s = st.secrets["postgres"]
        user = s.get("user")
        password = s.get("password")
        host = s.get("host", "localhost")
        port = s.get("port", 5432)
        dbname = s.get("dbname")
        sslmode = s.get("sslmode")
        # Auto-require SSL for Neon if not explicitly set
        if not sslmode and isinstance(host, str) and host.endswith("neon.tech"):
            sslmode = "require"
        user_enc = quote_plus(str(user)) if user is not None else ""
        pwd_enc = quote_plus(str(password)) if password is not None else ""
        base = f"postgresql+psycopg2://{user_enc}:{pwd_enc}@{host}:{port}/{dbname}"
        return f"{base}?sslmode={sslmode}" if sslmode else base

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "appdb")
    sslmode = os.getenv("POSTGRES_SSLMODE")
    if not sslmode and host.endswith("neon.tech"):
        sslmode = "require"
    user_enc = quote_plus(str(user)) if user is not None else ""
    pwd_enc = quote_plus(str(password)) if password is not None else ""
    base = f"postgresql+psycopg2://{user_enc}:{pwd_enc}@{host}:{port}/{dbname}"
    return f"{base}?sslmode={sslmode}" if sslmode else base


@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    return create_engine(get_db_url(), pool_pre_ping=True)


@contextmanager
def get_conn():
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


def ensure_schema() -> None:
    with get_conn() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
        )


def insert_note(content: str) -> None:
    with get_conn() as conn:
        conn.execute(text("INSERT INTO notes (content) VALUES (:c)"), {"c": content})


def fetch_notes() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql(text("SELECT id, content, created_at FROM notes ORDER BY created_at DESC"), conn)
    return df


def main() -> None:
    st.set_page_config(page_title="Streamlit + PostgreSQL Starter", page_icon="🧰", layout="centered")
    st.title("🧰 Streamlit + PostgreSQL Starter")

    # DB health
    try:
        ensure_schema()
        with get_conn() as conn:
            version = conn.execute(text("SELECT version()"))
            st.success("Connected to PostgreSQL")
            st.caption(next(version)[0])
    except Exception as exc:  # noqa: BLE001 - surfaced to UI
        st.error(f"Database connection failed: {exc}")
        st.stop()

    with st.form("add_note", clear_on_submit=True):
        content = st.text_input("Add a note", placeholder="Hello, database!")
        submitted = st.form_submit_button("Save")
        if submitted and content.strip():
            insert_note(content.strip())
            st.toast("Note saved", icon="✅")

    st.subheader("Notes")
    notes = fetch_notes()
    if notes.empty:
        st.info("No notes yet. Add your first one above.")
    else:
        st.dataframe(notes, use_container_width=True)


if __name__ == "__main__":
    main()


