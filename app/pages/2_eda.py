"""
2_eda.py
--------
Page 2 — Exploratory Data Analysis (EDA)

This page provides rich interactive visualizations to understand:
- Patient demographics
- Drug efficacy comparisons
- Feature correlations
- Side effect patterns

All plots use Plotly for interactivity (zoom, hover, export).

Author: Banoth Rajesham
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.db             import load_patients
from app.utils.visualizations import (
    plot_success_by_drug, plot_age_distribution, plot_correlation_heatmap,
    plot_trial_phase_success, plot_bmi_vs_success, plot_dosage_side_effects,
)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EDA | PharmaML", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis")
st.markdown("Interactive visualizations to understand clinical trial data patterns.")

# Load data
df = load_patients()

if df.empty:
    st.error("No data found. Run `python data/generate_data.py` first.")
    st.stop()

# ── Section 1: Summary Stats ──────────────────────────────────────────────────
st.subheader("📊 Dataset Statistics")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Numeric Feature Summary:**")
    numeric_summary = df.describe().round(2)
    st.dataframe(numeric_summary, use_container_width=True)

with col2:
    st.markdown("**Categorical Value Counts:**")
    cat_counts = {
        "Drug Assigned"  : df["drug_assigned"].value_counts().to_dict(),
        "Trial Phase"    : df["trial_phase"].value_counts().to_dict(),
        "Disease Area"   : df["disease_area"].value_counts().to_dict(),
        "Side Effects"   : df["side_effects"].value_counts().to_dict(),
        "Gender"         : df["gender"].value_counts().to_dict(),
    }
    selected_cat = st.selectbox("Select Category:", list(cat_counts.keys()))
    import plotly.express as px
    cat_data = list(cat_counts[selected_cat].items())
    cat_df   = __import__("pandas").DataFrame(cat_data, columns=["Value", "Count"])
    fig_cat  = px.pie(cat_df, names="Value", values="Count",
                      title=f"Distribution of {selected_cat}",
                      color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_cat, use_container_width=True)


st.markdown("---")

# ── Section 2: Drug Efficacy ──────────────────────────────────────────────────
st.subheader("💊 Drug Efficacy Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = plot_success_by_drug(df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = plot_trial_phase_success(df)
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# ── Section 3: Patient Demographics ──────────────────────────────────────────
st.subheader("👥 Patient Demographics")
col1, col2 = st.columns(2)

with col1:
    fig = plot_age_distribution(df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = plot_bmi_vs_success(df)
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")

# ── Section 4: Clinical Lab Values ───────────────────────────────────────────
st.subheader("🧪 Lab Values Distribution")

lab_cols = ["cholesterol", "blood_pressure_mmhg", "hemoglobin_gdl", "wbc_count_k_ul"]
selected_lab = st.selectbox("Select Lab Metric:", lab_cols)

import plotly.express as px
df_copy = df.copy()
df_copy["Outcome"] = df_copy["treatment_success"].map({1: "Success", 0: "Failure"})

fig_violin = px.violin(
    df_copy,
    x="Outcome",
    y=selected_lab,
    color="Outcome",
    box=True,
    points="outliers",
    title=f"Distribution of {selected_lab} by Outcome",
    color_discrete_map={"Success": "#0f9d58", "Failure": "#d93025"},
)
st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# ── Section 5: Side Effects ───────────────────────────────────────────────────
st.subheader("⚠️ Side Effects Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = plot_dosage_side_effects(df)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Side effects heatmap by drug × severity
    import pandas as pd
    side_heat = (
        df.groupby(["drug_assigned", "side_effects"])
        .size()
        .unstack(fill_value=0)
    )
    fig_heat = px.imshow(
        side_heat,
        title="Side Effect Severity by Drug (Heatmap)",
        color_continuous_scale="Reds",
        text_auto=True,
    )
    st.plotly_chart(fig_heat, use_container_width=True)


st.markdown("---")

# ── Section 6: Correlation Heatmap ───────────────────────────────────────────
st.subheader("🔗 Feature Correlation Heatmap")
fig = plot_correlation_heatmap(df)
st.plotly_chart(fig, use_container_width=True)

st.info("💡 Positive correlation = features tend to increase together. Negative = inverse relationship.")

st.markdown("---")
st.caption("All charts are interactive — hover to see values, zoom in/out, and download as PNG.")
