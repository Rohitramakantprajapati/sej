"""sej FastAPI backend routes and runtime services."""

import json
import math
import logging
import os
import sqlite3
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from eda_engine import run_eda
from ingestion import load_bytes, summarize_result
from modeling import run_modeling

# ── App setup ──────────────────────────────────────────────────────────────────

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
MAX_IN_MEMORY_SESSIONS = int(os.getenv("MAX_IN_MEMORY_SESSIONS", "64"))
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "").strip()
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]
ALLOWED_UPLOAD_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

app = FastAPI(title="sej API", version="1.1.0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("sej")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = Path("autoanalyst.db")

# In-memory session store  {session_id: {"df": ..., "eda": ..., "model": ...}}
SESSIONS: dict = {}
REQUEST_WINDOW: dict = {}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                eda_json TEXT,
                model_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )


def save_session_record(
    session_id: str,
    file_name: str,
    file_path: str,
    eda: Optional[dict],
    model: Optional[dict],
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, file_name, file_path, eda_json, model_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                file_name=excluded.file_name,
                file_path=excluded.file_path,
                eda_json=excluded.eda_json,
                model_json=excluded.model_json
            """,
            (
                session_id,
                file_name,
                file_path,
                json.dumps(eda) if eda is not None else None,
                json.dumps(model) if model is not None else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def update_session_model(session_id: str, model: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE sessions SET model_json = ? WHERE session_id = ?",
            (json.dumps(model), session_id),
        )


def load_session_record(session_id: str) -> Optional[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT session_id, file_name, file_path, eda_json, model_json FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return None

    return {
        "session_id": row["session_id"],
        "file_name": row["file_name"],
        "file_path": row["file_path"],
        "eda": json.loads(row["eda_json"]) if row["eda_json"] else None,
        "model": json.loads(row["model_json"]) if row["model_json"] else None,
    }


def validate_upload(filename: str, content: bytes) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_MB}MB.",
        )


def persist_uploaded_file(session_id: str, filename: str, content: bytes) -> Path:
    ext = Path(filename).suffix.lower()
    safe_name = f"{session_id}{ext}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)
    return file_path


def hydrate_session(session_id: str) -> Optional[dict]:
    session = SESSIONS.get(session_id)
    if session:
        return session

    record = load_session_record(session_id)
    if not record:
        return None

    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Session file no longer exists on disk.")

    file_content = file_path.read_bytes()
    result = load_bytes(file_content, record["file_name"])
    eda = record["eda"] or run_eda(result.df)

    session = {
        "df": result.df,
        "eda": eda,
        "model": record["model"],
        "file_name": record["file_name"],
        "file_path": str(file_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    SESSIONS[session_id] = session

    if record["eda"] is None:
        save_session_record(
            session_id=session_id,
            file_name=record["file_name"],
            file_path=str(file_path),
            eda=eda,
            model=record["model"],
        )
    return session


def _parse_created_at(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def cleanup_expired_sessions() -> None:
    now = datetime.now(timezone.utc)
    ttl_seconds = max(1, SESSION_TTL_HOURS) * 3600
    expired = []
    for session_id, payload in list(SESSIONS.items()):
        created_at = _parse_created_at(payload.get("created_at"))
        age = (now - created_at).total_seconds()
        if age > ttl_seconds:
            expired.append(session_id)

    for session_id in expired:
        SESSIONS.pop(session_id, None)

    if len(SESSIONS) <= MAX_IN_MEMORY_SESSIONS:
        return

    ordered = sorted(
        SESSIONS.items(),
        key=lambda item: _parse_created_at(item[1].get("created_at")),
    )
    overflow = len(SESSIONS) - MAX_IN_MEMORY_SESSIONS
    for session_id, _ in ordered[:overflow]:
        SESSIONS.pop(session_id, None)


def _check_rate_limit(client_key: str) -> tuple[bool, int]:
    now = datetime.now(timezone.utc).timestamp()
    window_start = now - 60
    hits = [ts for ts in REQUEST_WINDOW.get(client_key, []) if ts >= window_start]
    allowed = len(hits) < RATE_LIMIT_PER_MINUTE
    if allowed:
        hits.append(now)
    REQUEST_WINDOW[client_key] = hits
    remaining = max(0, RATE_LIMIT_PER_MINUTE - len(hits))
    return allowed, remaining


def _top_entries(data: dict, limit: int = 3) -> list:
    items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    return [{"label": key, "value": value} for key, value in items[:limit]]


def build_dashboard_spec(df: pd.DataFrame, file_name: str, eda: dict, model: Optional[dict] = None) -> dict:
    numeric_stats = eda.get("descriptive_stats", {}).get("numeric", {}) or {}
    categorical_stats = eda.get("descriptive_stats", {}).get("categorical", {}) or {}
    strong_pairs = eda.get("correlation", {}).get("strong_pairs", []) or []
    anomalies = eda.get("anomalies", {}) or {}
    model = model or {}

    total_rows = int(len(df))
    total_cols = int(len(df.columns))
    missing_total = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols = list(numeric_stats.keys())
    categorical_cols = list(categorical_stats.keys())
    anomaly_pct = float(anomalies.get("total_outlier_pct", 0) or 0)
    top_insight = model.get("insights", [{}])[0] if model.get("insights") else None

    visible_sections = ["auto-dashboard", "overview", "model", "raw-data"]
    if numeric_cols:
        visible_sections.extend(["distributions", "correlations", "anomalies", "scatter"])
    visible_sections = list(dict.fromkeys(visible_sections))

    priority_views = [
        {
            "id": "overview",
            "label": "Overview",
            "enabled": True,
            "reason": "Always show the dataset summary first.",
        },
        {
            "id": "distributions",
            "label": "Distributions",
            "enabled": bool(numeric_cols),
            "reason": f"{len(numeric_cols)} numeric column(s) available." if numeric_cols else "No numeric columns detected.",
        },
        {
            "id": "correlations",
            "label": "Correlations",
            "enabled": len(numeric_cols) >= 2,
            "reason": "Correlation view needs at least two numeric columns." if len(numeric_cols) < 2 else f"{len(strong_pairs)} strong pair(s) detected.",
        },
        {
            "id": "anomalies",
            "label": "Anomalies",
            "enabled": anomaly_pct > 0 or bool(anomalies.get("per_column")),
            "reason": "Outlier patterns detected." if anomaly_pct > 0 else "No anomaly signal above threshold.",
        },
        {
            "id": "model",
            "label": "Model",
            "enabled": True,
            "reason": "Model panel is available for supervised or unsupervised runs.",
        },
        {
            "id": "scatter",
            "label": "Explorer",
            "enabled": len(numeric_cols) >= 2,
            "reason": "Scatter explorer needs at least two numeric columns." if len(numeric_cols) < 2 else "Useful for drill-down.",
        },
        {
            "id": "raw-data",
            "label": "Raw Data",
            "enabled": True,
            "reason": "Always keep a data preview available.",
        },
    ]

    summary_cards = [
        {"type": "metric", "label": "Rows", "value": f"{total_rows:,}", "sub": file_name},
        {"type": "metric", "label": "Columns", "value": str(total_cols), "sub": f"{len(numeric_cols)} numeric / {len(categorical_cols)} categorical"},
        {"type": "metric", "label": "Missing", "value": f"{missing_total:,}", "sub": "Total null values"},
        {"type": "metric", "label": "Duplicates", "value": f"{duplicate_rows:,}", "sub": "Exact duplicate rows"},
        {"type": "metric", "label": "Strong Pairs", "value": str(len(strong_pairs)), "sub": "High-correlation relationships"},
        {"type": "metric", "label": "Outlier %", "value": f"{anomaly_pct:.2f}%", "sub": "Combined anomaly signal"},
    ]

    quality_cards = []
    if missing_total > 0:
        quality_cards.append({"type": "warning", "title": "Missingness", "detail": f"{missing_total:,} missing values across the dataset."})
    if duplicate_rows > 0:
        quality_cards.append({"type": "warning", "title": "Duplicates", "detail": f"{duplicate_rows:,} duplicate row(s) should be reviewed."})
    if anomaly_pct > 5:
        quality_cards.append({"type": "warning", "title": "Outliers", "detail": f"{anomaly_pct:.2f}% of rows are flagged as outliers."})
    if not quality_cards:
        quality_cards.append({"type": "info", "title": "Data health", "detail": "No critical data-quality issue detected by the current rules."})

    insights = model.get("insights", []) or []
    focus_cards = []
    if strong_pairs:
        first_pair = strong_pairs[0]
        focus_cards.append({
            "type": "correlation",
            "title": "Top relationship",
            "detail": f"{first_pair['col_a']} and {first_pair['col_b']} with r={first_pair['r']}.",
        })
    if top_insight:
        focus_cards.append({
            "type": top_insight.get("type", "info"),
            "title": top_insight.get("title", "Model insight"),
            "detail": top_insight.get("detail", ""),
        })
    if not focus_cards:
        focus_cards.append({"type": "info", "title": "Auto layout", "detail": "The dashboard will prioritize summary cards and the sections that matter most for this dataset."})

    recommended_views = []
    for view in priority_views:
        if view["enabled"]:
            recommended_views.append(view)

    chart_priority = []
    if numeric_cols:
        chart_priority.extend([
            {"label": "Distributions", "detail": f"{len(numeric_cols)} numeric field(s) ready for histograms and box plots."},
            {"label": "Explorer", "detail": "Scatter explorer available for drill-down across numeric pairs."},
        ])
    if strong_pairs:
        chart_priority.append({"label": "Correlations", "detail": f"{len(strong_pairs)} strong pair(s) are worth reviewing."})
    if anomaly_pct > 0:
        chart_priority.append({"label": "Anomalies", "detail": f"{anomaly_pct:.2f}% outlier signal should be reviewed."})
    if insights:
        chart_priority.append({"label": "Insights", "detail": f"{len(insights)} generated insight(s) surfaced by the model and EDA layers."})

    if not chart_priority:
        chart_priority.append({"label": "Overview", "detail": "Only summary cards are relevant for this dataset shape."})

    return {
        "title": f"Auto Dashboard · {file_name}",
        "subtitle": "Auto-composed from uploaded data, EDA signals, and model output.",
        "visible_sections": visible_sections,
        "summary_cards": summary_cards,
        "quality_cards": quality_cards,
        "focus_cards": focus_cards,
        "recommended_views": recommended_views,
        "chart_priority": chart_priority,
        "model_state": {
            "task": (model.get("task_info") or {}).get("task", "pending"),
            "reason": (model.get("task_info") or {}).get("reason", "Upload a file and train a model to populate this panel."),
        },
    }


init_db()


@app.middleware("http")
async def guardrails_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex
    cleanup_expired_sessions()

    # Trim old rate-limit entries opportunistically.
    if len(REQUEST_WINDOW) > 2000:
        now = datetime.now(timezone.utc).timestamp()
        cutoff = now - 60
        REQUEST_WINDOW.update(
            {key: [ts for ts in values if ts >= cutoff] for key, values in REQUEST_WINDOW.items()}
        )
        stale_keys = [key for key, values in REQUEST_WINDOW.items() if not values]
        for key in stale_keys:
            REQUEST_WINDOW.pop(key, None)

    client_ip = request.client.host if request.client else "unknown"

    if APP_API_TOKEN and request.url.path != "/health" and request.method != "OPTIONS":
        auth_header = request.headers.get("Authorization", "")
        expected = f"Bearer {APP_API_TOKEN}"
        if auth_header != expected:
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"X-Request-ID": request_id},
            )

    allowed, remaining = _check_rate_limit(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry shortly."},
            headers={
                "X-RateLimit-Limit": str(RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
                "X-Request-ID": request_id,
            },
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-Request-ID"] = request_id
    return response


# ── JSON encoder for numpy types ───────────────────────────────────────────────

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric_value = float(value)
        return None if not np.isfinite(numeric_value) else numeric_value
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def jsonify(data) -> dict:
    return json.loads(json.dumps(make_json_safe(data), cls=NpEncoder, default=str, allow_nan=False))


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": APP_ENV,
        "sessions_in_memory": len(SESSIONS),
        "session_ttl_hours": SESSION_TTL_HOURS,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "auth_enabled": bool(APP_API_TOKEN),
        "db": str(DB_PATH),
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV/Excel/JSON file, run EDA, return session_id + EDA results."""
    try:
        content = await file.read()
        validate_upload(file.filename or "", content)
        result = load_bytes(content, file.filename or "uploaded_file")

        if result.df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file produced an empty DataFrame.")

        session_id = str(uuid.uuid4())
        file_path = persist_uploaded_file(session_id, file.filename or "uploaded_file", content)
        eda = run_eda(result.df)

        SESSIONS[session_id] = {
            "df": result.df,
            "eda": eda,
            "model": None,
            "file_name": file.filename,
            "file_path": str(file_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        save_session_record(
            session_id=session_id,
            file_name=file.filename or "uploaded_file",
            file_path=str(file_path),
            eda=eda,
            model=None,
        )

        logger.info("Created session %s for file %s", session_id, file.filename)

        return JSONResponse(jsonify({
            "session_id": session_id,
            "summary": summarize_result(result),
            "eda": eda,
            "dashboard_spec": build_dashboard_spec(result.df, file.filename or "uploaded_file", eda, None),
        }))

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ModelRequest(BaseModel):
    session_id: str
    target_col: Optional[str] = None


@app.post("/model")
async def train_model(req: ModelRequest):
    """Train a model on the uploaded session data."""
    session = None
    try:
        session = hydrate_session(req.session_id)
    except Exception:
        # Force one retry from persisted record if in-memory state became inconsistent.
        SESSIONS.pop(req.session_id, None)
        session = hydrate_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

    try:
        df = session["df"]
        eda = session["eda"]
        model_result = run_modeling(df, req.target_col, eda)
        session["model"] = model_result
        update_session_model(req.session_id, model_result)
        return JSONResponse(jsonify({
            **model_result,
            "dashboard_spec": build_dashboard_spec(df, session["file_name"], eda, model_result),
        }))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/column-data/{session_id}/{column}")
async def column_data(session_id: str, column: str):
    """Return raw column values for drill-down / tooltip context."""
    session = hydrate_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    df = session["df"]
    if column not in df.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found.")

    s = df[column]
    payload = {
        "column": column,
        "dtype": str(s.dtype),
        "values": s.dropna().tolist()[:500],
        "missing": int(s.isnull().sum()),
    }
    if pd.api.types.is_numeric_dtype(s):
        payload["stats"] = {
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
        }
    return JSONResponse(jsonify(payload))


@app.get("/scatter/{session_id}")
async def scatter_data(session_id: str, x: str, y: str, color: Optional[str] = None):
    """Return x,y (,color) data for a scatter plot with full row context for tooltips."""
    session = hydrate_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    df = session["df"]
    for col in [x, y]:
        if col not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found.")

    subset = df[[x, y] + ([color] if color and color in df.columns else [])].dropna()
    records = subset.rename(columns={x: "x", y: "y"}).to_dict(orient="records")
    return JSONResponse(jsonify({"x_col": x, "y_col": y, "color_col": color, "points": records[:2000]}))


@app.get("/rows/{session_id}")
async def row_data(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
):
    """Return paginated rows for table rendering and frontend drill-down."""
    session = hydrate_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    df = session["df"]
    rows = df.iloc[offset : offset + limit].fillna("").to_dict(orient="records")
    return JSONResponse(
        jsonify(
            {
                "rows": rows,
                "offset": offset,
                "limit": limit,
                "total_rows": int(len(df)),
            }
        )
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Return full cached EDA + model results for a session."""
    session = hydrate_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(jsonify({
        "file_name": session["file_name"],
        "eda": session["eda"],
        "model": session["model"],
        "columns": session["df"].columns.tolist(),
        "dashboard_spec": build_dashboard_spec(session["df"], session["file_name"], session["eda"], session["model"]),
    }))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=APP_ENV != "production",
    )
