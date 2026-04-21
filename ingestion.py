"""
Data Ingestion Module
Handles CSV, Excel, JSON files and entire folder structures.
Auto-detects type, loads into standardized DataFrame.
"""

import os
import io
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class IngestionResult:
    df: pd.DataFrame
    file_name: str
    file_type: str
    shape: tuple
    columns: List[str]
    dtypes: Dict[str, str]
    warnings: List[str] = field(default_factory=list)
    sample_rows: int = 5


LOADERS = {
    ".csv": lambda b, **kw: pd.read_csv(io.BytesIO(b), **kw),
    ".tsv": lambda b, **kw: pd.read_csv(io.BytesIO(b), sep="\t", **kw),
    ".xlsx": lambda b, **kw: pd.read_excel(io.BytesIO(b), **kw),
    ".xls": lambda b, **kw: pd.read_excel(io.BytesIO(b), engine="xlrd", **kw),
    ".json": lambda b, **kw: pd.read_json(io.BytesIO(b), **kw),
    ".parquet": lambda b, **kw: pd.read_parquet(io.BytesIO(b), **kw),
}


def _infer_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt intelligent dtype casting: parse dates, coerce numerics."""
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() / max(len(df), 1) > 0.8:
                df[col] = coerced
                continue
            # Try datetime
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / max(len(df), 1) > 0.6:
                    df[col] = parsed
            except Exception:
                pass
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and deduplicate column names to keep JSON payloads stable."""
    cleaned = [str(col).replace("\ufeff", "").strip() or "unnamed" for col in df.columns]
    seen = {}
    normalized = []
    for name in cleaned:
        count = seen.get(name, 0)
        normalized_name = name if count == 0 else f"{name}_{count}"
        normalized.append(normalized_name)
        seen[name] = count + 1
    df.columns = normalized
    return df


def load_bytes(content: bytes, filename: str) -> IngestionResult:
    """Load raw bytes from an uploaded file into a DataFrame."""
    suffix = Path(filename).suffix.lower()
    warnings = []

    if suffix not in LOADERS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(LOADERS.keys())}")

    try:
        df = LOADERS[suffix](content)
    except Exception as e:
        # CSV fallback: try different encodings/separators
        if suffix == ".csv":
            for enc in ("latin-1", "utf-16"):
                try:
                    df = pd.read_csv(io.BytesIO(content), encoding=enc)
                    warnings.append(f"Used encoding fallback: {enc}")
                    break
                except Exception:
                    pass
            else:
                raise ValueError(f"Could not parse CSV: {e}")
        else:
            raise ValueError(f"Could not parse file: {e}")

    df = _normalize_columns(df)

    df = _infer_and_cast(df)

    # Strip whitespace from string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda value: value.strip() if isinstance(value, str) else value)

    if df.empty:
        warnings.append("DataFrame is empty after loading.")

    return IngestionResult(
        df=df,
        file_name=filename,
        file_type=suffix.lstrip("."),
        shape=df.shape,
        columns=df.columns.tolist(),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        warnings=warnings,
        sample_rows=min(5, len(df)),
    )


def load_folder(folder_path: str) -> List[IngestionResult]:
    """Load all supported files from a directory, concatenating same-schema files."""
    results = []
    folder = Path(folder_path)
    for fp in sorted(folder.rglob("*")):
        if fp.suffix.lower() in LOADERS and fp.is_file():
            try:
                content = fp.read_bytes()
                results.append(load_bytes(content, fp.name))
            except Exception as e:
                results.append(
                    IngestionResult(
                        df=pd.DataFrame(),
                        file_name=fp.name,
                        file_type=fp.suffix.lstrip("."),
                        shape=(0, 0),
                        columns=[],
                        dtypes={},
                        warnings=[f"Failed to load: {e}"],
                    )
                )
    return results


def summarize_result(result: IngestionResult) -> Dict[str, Any]:
    """Serializable summary of an IngestionResult."""
    df = result.df
    return {
        "file_name": result.file_name,
        "file_type": result.file_type,
        "rows": result.shape[0],
        "columns": result.shape[1],
        "column_names": result.columns,
        "dtypes": result.dtypes,
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "sample": df.head(result.sample_rows).fillna("").to_dict(orient="records"),
        "warnings": result.warnings,
    }
