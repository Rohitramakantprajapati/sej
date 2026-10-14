# AutoAnalyst Streamlit App
# This is the entry point for the fully Python-based dashboard UI.
# Run with: streamlit run streamlit_app.py


import streamlit as st
import pandas as pd
import numpy as np
import os
from eda_engine import run_eda
from modeling import run_modeling
from ingestion import load_bytes

st.set_page_config(page_title="AutoAnalyst", layout="wide")
st.title("AutoAnalyst: Python EDA & ML Dashboard")

uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["csv", "tsv", "xlsx", "xls", "json", "parquet"]
)

@st.cache_data(show_spinner=False)
def cached_load_bytes(content, filename):
    return load_bytes(content, filename)

@st.cache_data(show_spinner=False)
def cached_run_eda(df):
    return run_eda(df)

@st.cache_data(show_spinner=False)
def cached_run_modeling(df, eda):
    return run_modeling(df, eda)

if uploaded_file:
    content = uploaded_file.read()
    try:
        with st.spinner("Loading and parsing file..."):
            df, meta = cached_load_bytes(content, uploaded_file.name)
        st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        st.dataframe(df.head(20), use_container_width=True)

        with st.spinner("Running EDA..."):
            eda = cached_run_eda(df)
        st.subheader("Exploratory Data Analysis (summary)")
        # Show only summary stats, not full JSON
        st.write(f"**Rows:** {eda.get('total_rows', '-')}, **Columns:** {eda.get('total_cols', '-')} ")
        if 'numeric' in eda and eda['numeric']:
            st.write("**Numeric columns:**", list(eda['numeric'].keys()))
        if 'categorical' in eda and eda['categorical']:
            st.write("**Categorical columns:**", list(eda['categorical'].keys()))
        if st.checkbox("Show full EDA JSON", value=False):
            st.json(eda)

        if st.button("Run AutoML (may take time)"):
            with st.spinner("Training model..."):
                model = cached_run_modeling(df, eda)
            st.subheader("Model Results (summary)")
            if isinstance(model, dict):
                st.write({k: v for k, v in model.items() if k != 'full_model_object'})
                if st.checkbox("Show full model JSON", value=False):
                    st.json(model)
            else:
                st.write(model)
    except Exception as e:
        st.error(f"Failed to load/process file: {e}")
else:
    st.info("Upload a file to begin.")
