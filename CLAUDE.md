# CLAUDE.md

## Project Briefing: AutoAnalyst

### Project Overview
- AutoAnalyst is a full-stack analytics prototype that turns uploaded tabular data into immediate EDA, multi-model ML insights, and an interactive dashboard.
- Intended audience: analysts, learners, and developers needing rapid exploratory analysis.
- Delivery status: advanced prototype; not production-hardened.

### Tech Stack
- Backend: Python + FastAPI + Uvicorn
- Data/EDA: pandas, numpy, scipy
- ML: scikit-learn (multi-model: RandomForest, LinearRegression, Ridge, LogisticRegression, GradientBoosting, KMeans, etc.)
- Frontend: single-file HTML/CSS/JS (dashboard.html)
- Frontend libraries: Plotly, Chart.js, Leaflet, jsPDF, Google Fonts (CDN)
- Infra: sqlite metadata persistence, in-memory session cache, optional bearer auth, rate limiting

### Workspace Map
- README.md: up-to-date architecture and product description
- main.py: API routes, session lifecycle, upload/model/clean/schema endpoints, EDA/model retrain logic
- ingestion.py: input loaders, type casting, error handling
- eda_engine.py: EDA feature computation, profiling
- modeling.py: task detection, multi-model training, insight generation
- dashboard.html: full client app UI, state, rendering, model comparison

### Implemented Features
- File ingestion: csv, tsv, xlsx, xls, json, parquet
- Type inference, missingness profiling, descriptive stats, correlations, anomaly detection
- Model workflows: regression/classification (multi-model), clustering
- Model comparison: regression (LinearRegression, Ridge, RandomForest), classification (LogisticRegression, RandomForest, GradientBoosting)
- Feature importance, model metrics, sample predictions
- Insight synthesis layer (EDA + model)
- Interactive dashboard: auto dashboard, overview, distributions, correlations, anomalies, model comparison, explorer, raw data
- Professional Plotly-based dashboard: KPI cards, analytics grid, date presets, metric switch, chart type toggle, cross-chart filtering, export
- Backend endpoints: upload, model, session, scatter, column-data, cleaning, schema override

### Pending / Gaps
- No automated test suite or CI
- Session data is partially persistent (sqlite + file cache), resets on restart
- Optional bearer auth, but multi-user safety is limited
- Some legacy Chart.js code remains in dashboard.html
- No deployment automation or observability instrumentation

### Coding Rules and Patterns
- Python: snake_case, explicit dict payloads, utility-first modules
- JavaScript: camelCase, centralized STATE object, imperative rendering
- UI: tokenized CSS variables, component-like class sections
- API: JSON responses, explicit route handlers, UUID session keys
- Error handling: HTTPException for expected, controlled 500 for unexpected

### Important Notes
- API base URL: http://localhost:8000 (default)
- Environment variables: used for auth, rate limiting, etc.
- Response schema compatibility between main.py and dashboard.html is critical
- Validate both upload and retrain flows after changes
