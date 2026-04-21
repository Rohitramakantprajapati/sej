"""
Insight & Model Training Module
Auto-selects model type, trains, returns results + feature importances.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, silhouette_score
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# Avoid loky physical-core subprocess probing issues on constrained Windows envs.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))


# ── Task detection ─────────────────────────────────────────────────────────────

def detect_task(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Heuristic task detection:
      - If target provided: regression vs classification
      - If no target: clustering
    """
    if target_col is None:
        return {"task": "clustering", "target": None, "reason": "No target column specified"}

    if target_col not in df.columns:
        return {"task": "unknown", "target": None, "reason": f"Column '{target_col}' not found"}

    target = df[target_col].dropna()
    n_unique = target.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target)

    # Non-numeric targets should always be classified, regardless of class count.
    if not is_numeric:
        return {"task": "classification", "target": target_col, "reason": f"{n_unique} unique categorical classes"}

    if n_unique > 20:
        return {"task": "regression", "target": target_col, "reason": f"{n_unique} unique numeric values"}
    return {"task": "classification", "target": target_col, "reason": f"{n_unique} unique classes"}


# ── Feature prep ──────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Select and encode features for ML."""
    work = df.copy()
    feature_cols = []

    drop_cols = [target_col] if target_col else []
    # Drop high-cardinality string cols & date cols
    for col in work.columns:
        if col in drop_cols:
            continue
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            drop_cols.append(col)
        elif pd.api.types.is_object_dtype(work[col]) or pd.api.types.is_string_dtype(work[col]):
            if work[col].nunique(dropna=True) > 50:
                drop_cols.append(col)
                continue
            le = LabelEncoder()
            work[col] = le.fit_transform(work[col].astype(str))
            feature_cols.append(col)
        elif work[col].dtype == object and work[col].nunique() > 50:
            drop_cols.append(col)
        else:
            feature_cols.append(col)

    feature_cols = [c for c in feature_cols if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No usable feature columns remain after preprocessing.")

    X = work[feature_cols]

    # Impute missing
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=feature_cols)

    y = None
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series):
            fill_value = target_series.median()
            if pd.isna(fill_value):
                raise ValueError(f"Target column '{target_col}' is entirely null.")
            y = target_series.fillna(fill_value)
        else:
            target_mode = target_series.mode(dropna=True)
            if target_mode.empty:
                raise ValueError(f"Target column '{target_col}' is entirely null.")
            y = target_series.fillna(target_mode.iloc[0])
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)

    return X_imp, y, feature_cols


# ── Model trainers ────────────────────────────────────────────────────────────

def train_regression(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
    if len(X) < 10:
        return {"error": "Not enough rows for regression"}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    r2 = r2_score(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    cv_folds = max(2, min(5, len(X) // 5))
    try:
        cv = cross_val_score(model, X, y, cv=cv_folds, scoring="r2", n_jobs=1)
        cv_mean = round(float(np.nanmean(cv)), 4)
        cv_std = round(float(np.nanstd(cv)), 4)
    except Exception:
        cv_mean = None
        cv_std = None

    importances = dict(zip(feature_cols, model.feature_importances_.round(4)))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "model": "RandomForestRegressor",
        "r2_score": round(r2, 4),
        "rmse": round(rmse, 4),
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "feature_importances": dict(top_features),
        "top_features": [f[0] for f in top_features],
        "sample_predictions": {
            "actual": y_te.tolist()[:20],
            "predicted": [round(p, 4) for p in preds[:20]],
        },
    }


def train_classification(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Dict[str, Any]:
    if len(X) < 10:
        return {"error": "Not enough rows for classification"}

    value_counts = pd.Series(y).value_counts()
    stratify = y if len(value_counts) > 1 and (value_counts.min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    acc = accuracy_score(y_te, preds)
    importances = dict(zip(feature_cols, model.feature_importances_.round(4)))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "model": "RandomForestClassifier",
        "accuracy": round(acc, 4),
        "feature_importances": dict(top_features),
        "top_features": [f[0] for f in top_features],
        "classes": [str(c) for c in model.classes_],
        "n_classes": len(model.classes_),
    }


def train_clustering(X: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    if len(X) < 6:
        return {"error": "Not enough rows for clustering"}

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Find optimal k via silhouette (k=2..min(8,n-1))
    best_k, best_score = 2, -1
    scores = {}
    for k in range(2, min(9, len(X))):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_sc)
            s = silhouette_score(X_sc, labels)
            scores[k] = round(s, 4)
            if s > best_score:
                best_score, best_k = s, k
        except Exception:
            pass

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_sc)

    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    return {
        "model": "KMeans",
        "optimal_k": best_k,
        "silhouette_score": round(best_score, 4),
        "silhouette_per_k": scores,
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "cluster_labels": labels.tolist()[:200],
        "features_used": feature_cols,
    }


# ── Insights generator ────────────────────────────────────────────────────────

def generate_insights(
    eda_result: Dict[str, Any], model_result: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    insights = []

    # Missing data
    stats = eda_result.get("descriptive_stats", {})
    for col, info in stats.get("numeric", {}).items():
        if info.get("missing_pct", 0) > 20:
            insights.append({
                "type": "warning",
                "title": f"High Missing Rate: {col}",
                "detail": f"{info['missing_pct']}% of values are missing in '{col}'. Consider imputation or dropping.",
            })

    # Strong correlations
    for pair in eda_result.get("correlation", {}).get("strong_pairs", []):
        insights.append({
            "type": "correlation",
            "title": f"Strong {pair['direction'].capitalize()} Correlation",
            "detail": f"'{pair['col_a']}' and '{pair['col_b']}' have r={pair['r']} ({pair['strength']}).",
        })

    # Anomalies
    anom = eda_result.get("anomalies", {})
    if anom.get("total_outlier_pct", 0) > 5:
        insights.append({
            "type": "anomaly",
            "title": "Significant Outliers Detected",
            "detail": f"{anom['total_outlier_pct']}% of rows contain outliers across numeric columns.",
        })

    for col, info in anom.get("per_column", {}).items():
        if info.get("outlier_pct", 0) > 10:
            insights.append({
                "type": "anomaly",
                "title": f"Outliers in {col}",
                "detail": f"{info['outlier_pct']}% of values in '{col}' are flagged as outliers.",
            })

    # Model insights
    if model_result:
        if "r2_score" in model_result:
            quality = "excellent" if model_result["r2_score"] > 0.85 else "moderate" if model_result["r2_score"] > 0.6 else "poor"
            insights.append({
                "type": "model",
                "title": f"Regression Model: {quality.capitalize()} Fit",
                "detail": f"R² = {model_result['r2_score']}, RMSE = {model_result['rmse']}. Top predictor: {model_result.get('top_features', ['N/A'])[0]}.",
            })
        if "accuracy" in model_result:
            insights.append({
                "type": "model",
                "title": "Classification Model Trained",
                "detail": f"Accuracy = {model_result['accuracy']*100:.1f}%. Top predictor: {model_result.get('top_features', ['N/A'])[0]}.",
            })
        if "optimal_k" in model_result:
            insights.append({
                "type": "model",
                "title": f"Clustering: {model_result['optimal_k']} Natural Segments",
                "detail": f"Silhouette score = {model_result['silhouette_score']} (higher is better, max 1.0).",
            })

    if not insights:
        insights.append({"type": "info", "title": "Clean Dataset", "detail": "No critical issues detected. Data looks ready for analysis!"})

    return insights


# ── Master runner ─────────────────────────────────────────────────────────────

def run_modeling(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    eda_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_info = detect_task(df, target_col)
    task = task_info["task"]
    if task == "unknown":
        raise ValueError(task_info.get("reason", "Unknown modeling task."))

    X, y, feature_cols = prepare_features(df, target_col)

    model_result = {}
    if task == "regression" and y is not None:
        model_result = train_regression(X, y, feature_cols)
    elif task == "classification" and y is not None:
        model_result = train_classification(X, y, feature_cols)
    else:
        model_result = train_clustering(X, feature_cols)

    insights = generate_insights(eda_result or {}, model_result)

    return {
        "task_info": task_info,
        "feature_cols": feature_cols,
        "model_result": model_result,
        "insights": insights,
    }
