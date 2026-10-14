# AI Context: AutoAnalyst (Universal)

## Executive Summary
- AutoAnalyst is a robust prototype for rapid EDA and ML model comparison from uploaded tabular data, with a professional dashboard UI.
- The app is functionally complete for local analysis, but is not yet production-hardened for multi-user or cloud deployment.

## What the App Is Now
- App name: AutoAnalyst — Intelligent Data Explorer
- Core behavior: upload tabular data, compute EDA, train multiple models, and render a dashboard with KPIs, trends, correlations, anomalies, model comparison, and raw data.
- Primary users: analysts, learners, developers needing fast insight generation without notebooks.
- Delivery status: advanced prototype, not production-ready.

## Architecture Snapshot

### Backend
- Python + FastAPI + Uvicorn
- pandas, numpy, scipy for data/EDA
- scikit-learn for modeling (multi-model, feature importance, metrics)
- sqlite for session metadata, file cache for uploaded data
- In-memory session cache for speed; session TTL and cleanup present
- Optional bearer auth, rate limiting, request IDs

### Frontend
- Single-file HTML/CSS/JS (dashboard.html)
- Plotly for professional dashboard, Chart.js for legacy sections, Leaflet, jsPDF
- Global STATE object, render-by-section updates

### Repository Shape
- Flat workspace, all logic in top-level modules

## Current Implementation Strengths
- Multi-format ingestion: csv, tsv, xlsx, xls, json, parquet
- Robust EDA: KPIs, descriptive stats, histograms, boxplots, correlations, anomaly detection
- Modeling: task detection, feature prep, regression/classification (multi-model), clustering, feature importance, metrics, insights
- Dashboard: professional Plotly layout, KPI row, analytics grid, click filtering, date presets, metric switching, export
- Model comparison: regression (LinearRegression, Ridge, RandomForest), classification (LogisticRegression, RandomForest, GradientBoosting)

## Fixed Misalignments
- Backend uses Uvicorn, API wiring is configurable, model training is robust for Windows, runtime guardrails for session cleanup and throttling

## Remaining Gaps
- Some legacy Chart.js code remains in dashboard.html
- No automated test suite or CI; manual smoke testing only
- Session data is partially persistent (sqlite + file cache), resets on restart
- Not safe for multi-user production use without stronger auth and isolation

## Missing Tools / Pieces
- No test runner coverage (unit/integration/frontend)
- No deployment manifest (container, service config, CI)
- No observability (metrics, tracing, audit store)
- No automated retention policy beyond in-memory cleanup

## Runtime Contract
- Backend URL: http://localhost:8000 (default)
- Frontend API base can be overridden via query string, localStorage, or runtime global
- Optional auth token via APP_API_TOKEN
- Environment variables used for auth, rate limiting, etc.

## Codebase Rules To Preserve
- Python: snake_case; JS: camelCase
- Explicit, serializable JSON payloads
- Stable backend/frontend schema contract unless both are updated
- Avoid broad refactors without smoke testing (frontend is a large single file)

## Priority Upgrade Path
1. Remove remaining legacy frontend code or isolate it
2. Add automated tests for upload, model, rows, scatter, dashboard
3. Add deployment setup: container, service, environment profiles
4. Add dataset-size guardrails, background processing, better session lifecycle
5. Add stronger auth and multi-user separation if moving beyond demo use

## Important Notes
- The app works, but is not architecturally finished
- Always verify both upload and retrain flows after frontend/backend changes
- Keep dashboard.html and backend JSON in sync
- Use the current flat layout as the source of truth unless doing a full structure refactor
