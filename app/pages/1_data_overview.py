"""
1_data_overview.py
------------------
Page 1 — Data Overview & SQL Explorer

This page lets users:
1. Browse all patient records from the SQLite database
2. Filter data by drug, trial phase, and disease area
3. Run custom SQL queries interactively
4. View drug metadata table
5. Download filtered data as CSV

Think of this as the "data layer" visibility page — 
important for any data science stakeholder demo.

Author: Banoth Rajesham
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.db import load_patients, load_drug_info, run_query

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Overview | PharmaML", page_icon="📂", layout="wide")

st.title("📂 Data Overview & SQL Explorer")
st.markdown("Browse and filter clinical trial records stored in the SQLite database.")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔽 Filter Records")

# Load unfiltered first to get unique values for dropdowns
df_all = load_patients()

drug_options   = ["All"] + sorted(df_all["drug_assigned"].unique().tolist())
phase_options  = ["All"] + sorted(df_all["trial_phase"].unique().tolist())
disease_options= ["All"] + sorted(df_all["disease_area"].unique().tolist())

selected_drug    = st.sidebar.selectbox("💊 Drug",         drug_options)
selected_phase   = st.sidebar.selectbox("🔬 Trial Phase",  phase_options)
selected_disease = st.sidebar.selectbox("🏥 Disease Area", disease_options)

# Build filters dict
filters = {}
if selected_drug    != "All": filters["drug_assigned"] = selected_drug
if selected_phase   != "All": filters["trial_phase"]   = selected_phase
if selected_disease != "All": filters["disease_area"]  = selected_disease

# Load filtered data
df = load_patients(filters) if filters else df_all.copy()

# ── KPI strip ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("📋 Records",       f"{len(df):,}")
k2.metric("✅ Success Rate",   f"{df['treatment_success'].mean()*100:.1f}%")
k3.metric("⚠️ Dropout Rate",  f"{df['dropout'].mean()*100:.1f}%")
k4.metric("📅 Avg Trial Days", f"{df['trial_duration_days'].mean():.0f} days")

st.markdown("---")

# ── Data table ────────────────────────────────────────────────────────────────
st.subheader("📋 Patient Records")

# Color-code treatment_success column
def highlight_success(val):
    if val == 1:
        return "background-color: #1b5e20; color: white"
    elif val == 0:
        return "background-color: #b71c1c; color: white"
    return ""

st.dataframe(
    df.style.applymap(highlight_success, subset=["treatment_success"]),
    use_container_width=True,
    height=400,
)

# Download button
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label     = "⬇️ Download Filtered Data as CSV",
    data      = csv_data,
    file_name = "filtered_clinical_trials.csv",
    mime      = "text/csv",
)

st.markdown("---")

# ── Drug Info Table ───────────────────────────────────────────────────────────
st.subheader("💊 Drug Information Table")
drug_df = load_drug_info()
st.dataframe(drug_df, use_container_width=True)

st.markdown("---")

# ── Interactive SQL Explorer ─────────────────────────────────────────────────
st.subheader("🔍 Interactive SQL Query Console")
st.markdown("Write any SQL `SELECT` query against the `patients` and `drug_info` tables.")

# Example queries for beginners
example_queries = {
    "-- Select an example --": "",
    "All patients in Phase III": "SELECT * FROM patients WHERE trial_phase = 'Phase III' LIMIT 20;",
    "Success rate per drug": "SELECT drug_assigned, ROUND(AVG(treatment_success)*100,1) as success_pct FROM patients GROUP BY drug_assigned ORDER BY success_pct DESC;",
    "Average BMI by disease area": "SELECT disease_area, ROUND(AVG(bmi),2) as avg_bmi FROM patients GROUP BY disease_area;",
    "Patients with severe side effects": "SELECT * FROM patients WHERE side_effects = 'Severe' LIMIT 15;",
    "Drug info join patients count": "SELECT d.drug_name, d.category, COUNT(p.patient_id) as enrolled FROM drug_info d LEFT JOIN patients p ON d.drug_name = p.drug_assigned GROUP BY d.drug_name;",
}

chosen = st.selectbox("📌 Load an example query:", list(example_queries.keys()))

default_sql = example_queries[chosen] if chosen != "-- Select an example --" else "SELECT * FROM patients LIMIT 10;"
user_sql    = st.text_area("✏️ SQL Query:", value=default_sql, height=120)

if st.button("▶️ Run Query", type="primary"):
    if user_sql.strip():
        result = run_query(user_sql)
        if result.empty:
            st.warning("Query returned no results or an error occurred.")
        else:
            st.success(f"✅ {len(result)} rows returned")
            st.dataframe(result, use_container_width=True)
            # Allow downloading query result
            st.download_button(
                "⬇️ Download Query Result",
                data      = result.to_csv(index=False).encode("utf-8"),
                file_name = "query_result.csv",
                mime      = "text/csv",
            )
    else:
        st.info("Please enter a SQL query first.")

st.markdown("---")
st.caption("💡 Tables available: `patients`, `drug_info`")
