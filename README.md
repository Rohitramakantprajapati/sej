# sej — Intelligent Data Explorer

sej is a full-stack prototype that converts uploaded tabular data into automated EDA, optional ML outputs, and a professional analytics dashboard.

## Architecture Overview

```
auto-analyst/
├── main.py                        # FastAPI app + API routes
├── ingestion.py                   # Multi-format parsing + type normalization
├── eda_engine.py                  # Automated EDA payload generation
├── modeling.py                    # Auto model workflows + insights
├── dashboard.html                 # Single-file frontend (UI + logic + charts)
└── README.md
```

## Data Flow

```
Upload file
  -> ingestion.py
  -> eda_engine.py
  -> modeling.py (optional or retrain)
  -> dashboard.html rendering from API payload + row fetch
```

## Quick Start

```bash
# 1) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 4) Serve frontend
python -m http.server 3000
```

Open `http://localhost:3000/dashboard.html`.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/upload` | Upload file and return `session_id`, EDA summary, and dashboard metadata |
| POST | `/model` | Retrain model with optional target column |
| GET | `/column-data/{session_id}/{column}` | Raw column values |
| GET | `/scatter/{session_id}?x=col&y=col` | Scatter payload with row context |
| GET | `/rows/{session_id}?limit=2000&offset=0` | Row-level data for frontend dashboards |
| GET | `/sessions/{session_id}` | Full stored session payload |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Vanilla HTML/CSS/JS, Plotly (professional auto dashboard), Chart.js (legacy section charts), Leaflet, jsPDF |
| Backend | FastAPI + Uvicorn |
| Data/EDA | pandas, numpy, scipy |
| Modeling | scikit-learn |

## Dashboard Capabilities

- Professional analytics layout in `Auto Dashboard`:
  - KPI row with delta indicators and Plotly sparklines.
  - Responsive 3x2 analytics grid.
  - Donut, trend, horizontal ranking, grouped bars, retention curve + table, channel ranking.
  - Date range presets (7/30/90/custom), metric switch, line/bar toggle.
  - Cross-chart click filtering and per-panel export (PNG/CSV).
- EDA sections:
  - Overview KPIs + stats + generated insights.
  - Distributions, correlations, anomalies.
  - Model metrics + feature importance.
  - Explorer and raw data table.

## Supported Formats

- `.csv`
- `.tsv`
- `.xlsx` / `.xls`
- `.json`
- `.parquet`

## Known Gaps

- Workspace still uses a flat file layout, but the code now imports from the flat modules directly.
- Session state is partially persistent via sqlite metadata, but the in-memory data cache still resets on restart.
- No formal unit/integration CI suite yet, though a local smoke harness is available.
- Optional bearer auth and rate limiting are available via environment variables.

## Validation Checklist

After edits, validate both flows:

1. Backend-connected flow:
   - Upload file.
   - Confirm auto dashboard panels all render from real data.
   - Confirm chart click filters and exports work.
2. Retrain flow:
   - Select target and retrain.
   - Confirm model section refreshes and dashboard still renders.

## Smoke Test

Run the built-in end-to-end check against the live backend:

```bash
python smoke_test.py
```

This verifies `/health`, `/upload`, `/model`, `/rows`, and `/sessions` with a small sample dataset.
