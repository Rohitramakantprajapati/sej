# GitHub Copilot Instructions: AutoAnalyst

## Project Overview
AutoAnalyst is a prototype data exploration application. Users upload a tabular dataset, the backend computes automated EDA and optional ML outputs, and the frontend renders a multi-section interactive dashboard.

- Purpose: fast exploratory insight generation without notebooks.
- Users: analysts, learners, and developers.
- Maturity: development/prototype, not production-hardened.

## Technical Stack
- Backend: Python + FastAPI + Uvicorn.
- Data/EDA: pandas, numpy, scipy.
- ML: scikit-learn (RandomForest, KMeans, preprocessing/imputation/metrics/model_selection).
- Frontend: single-file HTML/CSS/JavaScript.
- Frontend libs: Plotly, Chart.js, Leaflet, jsPDF, Google Fonts (CDN).
- Services not present: database, auth provider, cloud-managed services.

## Repository Structure
- README.md: high-level docs and intended architecture.
- main.py: API app setup, CORS, session cache, route handlers.
- ingestion.py: input loaders and type casting utilities.
- eda_engine.py: EDA feature computations and payload builder.
- modeling.py: model selection/training + insight generation.
- dashboard.html: UI, local state, event handling, chart/table rendering.

## Implemented Features
- Supported ingestion formats: csv, tsv, xlsx, xls, json, parquet.
- Automatic EDA pipeline:
  - KPIs.
  - Numeric/categorical stats.
  - Distribution data.
  - Correlation matrix + strong pair extraction.
  - IQR/Z-score anomaly detection.
- Automated modeling:
  - Task detection from target/no-target.
  - Regression, classification, clustering workflows.
  - Feature importance and model-level metrics where applicable.
- Interactive dashboard with multiple analysis sections.
- Professional Plotly auto dashboard output:
  - KPI cards with trend sparkline and period delta.
  - Multi-panel analytics layout with auto data mapping.
  - Date presets/custom range, metric switch, chart-type toggle.
  - Cross-chart click filtering and per-panel export.

## Pending / Not Yet Built
- Add persistent storage and stable session lifecycle.
- Add stronger authentication, authorization, and security hardening.
- Add automated tests and CI.
- Add dependency pinning/lock files in this workspace.
- Improve consolidation of legacy Chart.js sections with the newer Plotly dashboard rendering path.
- Add deployment and observability strategy.

## Coding Rules for Copilot Contributions
- Follow existing naming:
  - Python: snake_case functions/variables.
  - JavaScript: camelCase.
  - Constants: uppercase where already used.
- Preserve API contracts unless coordinated frontend updates are included.
- Keep UI style consistent with current token-based CSS system in dashboard.html.
- Maintain current separation of compute modules:
  - ingestion responsibility in ingestion.py.
  - EDA responsibility in eda_engine.py.
  - modeling responsibility in modeling.py.
- Favor explicit, JSON-serializable payloads for responses.

## API and State Patterns
- Backend state: in-memory SESSIONS dict keyed by UUID.
- Frontend state: global STATE object with render-by-section updates.
- API calls use fetch and async/await on frontend.
- Backend uses JSONResponse and custom numpy/pandas serialization handling.

## Error Handling Expectations
- Backend: raise HTTPException for expected conditions (missing session/column, empty dataframe), catch unexpected exceptions and return controlled 500.
- Frontend: show upload errors in banner; retrain errors currently surfaced via alert.

## Important Notes
- API base URL is hardcoded as http://localhost:8000.
- No environment variables currently referenced.
- Session data is volatile (memory only) and disappears on restart.
- README folder map should be treated as intended architecture, not exact current layout.
- Validate both execution modes after edits:
  - Backend connected mode.
  - Upload + retrain flow.
