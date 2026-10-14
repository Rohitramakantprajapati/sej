import json
import sqlite3
from pathlib import Path

DB_PATH = Path("sessions.db")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                payload    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


def save_session(session_id: str, payload: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, payload) VALUES (?, ?)",
            (session_id, json.dumps(payload)),
        )


def load_session(session_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def list_sessions() -> list[str]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT session_id FROM sessions ORDER BY created_at DESC").fetchall()
    return [r[0] for r in rows]
