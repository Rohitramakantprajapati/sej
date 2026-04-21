"""
Automated EDA Engine
Descriptive stats, distributions, correlations, anomaly detection.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()

def _cat_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()

def _date_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

def _safe_mode(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if not m.empty else None


# ── descriptive stats ─────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    num = _numeric_cols(df)
    cat = _cat_cols(df)

    numeric_stats = {}
    for col in num:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numeric_stats[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "mode": round(float(_safe_mode(s)), 4) if _safe_mode(s) is not None else None,
            "std": round(float(s.std()), 4),
            "variance": round(float(s.var()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "q1": round(float(s.quantile(0.25)), 4),
            "q3": round(float(s.quantile(0.75)), 4),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurt()), 4),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique": int(s.nunique()),
        }

    categorical_stats = {}
    for col in cat:
        s = df[col].dropna()
        vc = s.value_counts()
        categorical_stats[col] = {
            "count": int(s.count()),
            "unique": int(s.nunique()),
            "top_values": vc.head(10).to_dict(),
            "missing": int(df[col].isnull().sum()),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
        }

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "date_columns": _date_cols(df),
        "total_rows": len(df),
        "total_cols": len(df.columns),
    }


# ── distribution data ─────────────────────────────────────────────────────────

def distribution_data(df: pd.DataFrame, bins: int = 30) -> Dict[str, Any]:
    """Return histogram + box plot data for each numeric column."""
    result = {}
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 2:
            continue
        counts, edges = np.histogram(s, bins=min(bins, len(s.unique())))
        result[col] = {
            "histogram": {
                "counts": counts.tolist(),
                "bin_edges": [round(x, 4) for x in edges.tolist()],
                "bin_centers": [round((edges[i] + edges[i+1]) / 2, 4) for i in range(len(edges)-1)],
            },
            "boxplot": {
                "min": round(float(s.min()), 4),
                "q1": round(float(s.quantile(0.25)), 4),
                "median": round(float(s.median()), 4),
                "q3": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "outliers": _iqr_outliers(s).tolist(),
            },
        }
    return result


# ── correlation ───────────────────────────────────────────────────────────────

def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    num = _numeric_cols(df)
    if len(num) < 2:
        return {"matrix": {}, "strong_pairs": []}

    corr = df[num].corr(method="pearson").round(4)
    # Find strongly correlated pairs (|r| > 0.6, excluding diagonal)
    strong_pairs = []
    for i, c1 in enumerate(num):
        for j, c2 in enumerate(num):
            if j <= i:
                continue
            r = corr.loc[c1, c2]
            if abs(r) >= 0.6:
                strong_pairs.append({
                    "col_a": c1, "col_b": c2,
                    "r": round(float(r), 4),
                    "strength": "strong" if abs(r) >= 0.8 else "moderate",
                    "direction": "positive" if r > 0 else "negative",
                })
    strong_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    return {
        "columns": num,
        "matrix": corr.to_dict(),
        "strong_pairs": strong_pairs,
    }


# ── anomaly detection ─────────────────────────────────────────────────────────

def _iqr_outliers(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]


def anomaly_detection(df: pd.DataFrame) -> Dict[str, Any]:
    """IQR + Z-score outlier detection per numeric column."""
    result = {}
    outlier_mask = pd.Series(False, index=df.index)

    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 4:
            continue

        # IQR method
        iqr_out = _iqr_outliers(s)
        # Z-score method
        z = np.abs(stats.zscore(s))
        zscore_idx = s.index[z > 3]

        # Union of both
        combined_idx = iqr_out.index.union(zscore_idx)
        outlier_mask.loc[combined_idx] = True

        result[col] = {
            "iqr_outliers_count": int(len(iqr_out)),
            "zscore_outliers_count": int(len(zscore_idx)),
            "combined_outlier_count": int(len(combined_idx)),
            "outlier_pct": round(len(combined_idx) / len(s) * 100, 2),
            "iqr_bounds": {
                "lower": round(float(s.quantile(0.25) - 1.5 * (s.quantile(0.75) - s.quantile(0.25))), 4),
                "upper": round(float(s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25))), 4),
            },
            "sample_outlier_values": iqr_out.head(10).round(4).tolist(),
        }

    return {
        "per_column": result,
        "total_outlier_rows": int(outlier_mask.sum()),
        "total_outlier_pct": round(outlier_mask.mean() * 100, 2),
        "outlier_row_indices": outlier_mask[outlier_mask].index.tolist()[:100],
    }


# ── KPI cards ─────────────────────────────────────────────────────────────────

def generate_kpis(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Top-level KPI cards for the dashboard."""
    num = _numeric_cols(df)
    kpis = [
        {"label": "Total Rows", "value": f"{len(df):,}", "icon": "rows"},
        {"label": "Total Columns", "value": str(len(df.columns)), "icon": "columns"},
        {"label": "Missing Values", "value": f"{df.isnull().sum().sum():,}", "icon": "missing"},
        {"label": "Numeric Features", "value": str(len(num)), "icon": "numeric"},
        {"label": "Categorical Features", "value": str(len(_cat_cols(df))), "icon": "category"},
        {"label": "Duplicate Rows", "value": f"{df.duplicated().sum():,}", "icon": "duplicate"},
    ]
    # Add top numeric KPIs
    for col in num[:3]:
        s = df[col].dropna()
        kpis.append({
            "label": f"Avg {col}",
            "value": f"{s.mean():.2f}",
            "icon": "avg",
            "sub": f"±{s.std():.2f}",
        })
    return kpis


# ── master EDA runner ─────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "kpis": generate_kpis(df),
        "descriptive_stats": descriptive_stats(df),
        "distributions": distribution_data(df),
        "correlation": correlation_matrix(df),
        "anomalies": anomaly_detection(df),
    }
