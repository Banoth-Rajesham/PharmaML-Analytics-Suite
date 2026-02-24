"""
main.py
-------
PharmaML Analytics Suite — Main Entry Point

Run with:  streamlit run app/main.py

This is the landing page / home dashboard. It shows:
- Project description
- Key KPI metric cards
- Navigation instructions

Author: Banoth Rajesham
"""

import streamlit as st
import sys
import time
from pathlib import Path

# Add project root to Python path so utils work from any page
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.db import get_summary_stats, DB_PATH, check_and_init_db

# ── Page config (must be FIRST streamlit call) ────────────────────────────────
st.set_page_config(
    page_title  = "PharmaML Analytics Suite",
    page_icon   = "💊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS for premium look ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1a3a5c 100%);
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e8f4fd !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a3a5c, #0d253f);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #1a73e8;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #4fc3f7 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #90caf9 !important;
        font-size: 0.85rem !important;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0f4c75 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #1a73e8;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #1a73e8;
        transition: transform 0.2s;
        height: 100%;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        border-color: #4fc3f7;
    }

    /* General page background */
    .stApp {
        background: #0a1628;
        color: #e8f4fd;
    }

    h1, h2, h3 { color: #4fc3f7; }
    p, li       { color: #cfd8dc; }
</style>
""", unsafe_allow_html=True)


# ── Check database exists ─────────────────────────────────────────────────────
db_ready = check_and_init_db()

with st.sidebar:
    st.markdown("## 💊 PharmaML Suite")
    st.markdown("---")
    st.markdown("""
    **Navigation:**
    - 📂 Data Overview
    - 📈 EDA & Charts
    - 🤖 ML Pipeline
    - 🔮 Predictions
    - 💬 AI Assistant
    """)
    st.markdown("---")
    if db_ready:
        st.success("✅ Database connected")
    else:
        st.error("❌ Database initialization failed")
        if st.button("Retry Initialization"):
            st.rerun()
    st.markdown("---")
    st.info("**Author:** Banoth Rajesham  \n**Role:** Data Scientist")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:#4fc3f7; margin:0; font-size:2.5rem;">
        💊 PharmaML Analytics Suite
    </h1>
    <p style="color:#90caf9; margin:.5rem 0 0; font-size:1.1rem;">
        End-to-End Clinical Trial Data Science Platform · Powered by Python & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)


# ── KPI Metric Cards ──────────────────────────────────────────────────────────
if db_ready:
    try:
        stats = get_summary_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("👥 Total Patients",    f"{stats['total_patients']:,}")
        c2.metric("✅ Avg Success Rate",   f"{stats['avg_success_rate']}%")
        c3.metric("💊 Drugs in Trials",   stats["n_drugs"])
        c4.metric("🔬 Trial Phases",       stats["n_phases"])
    except Exception as e:
        st.warning(f"Could not load stats: {e}")
else:
    st.warning("⚠️ Database not found. Please run `python data/generate_data.py` to generate the dataset.")


st.markdown("---")

# ── Feature Overview Cards ────────────────────────────────────────────────────
st.subheader("🗂️ Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📂 Data Overview</h3>
        <p>
        Browse raw clinical trial records stored in SQLite.
        Run live SQL queries, filter by phase/drug/disease,
        and download results as CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>📈 EDA & Visualization</h3>
        <p>
        Explore patient demographics, drug efficacy,
        lab value distributions, and correlation heatmaps
        through interactive Plotly charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 ML Pipeline</h3>
        <p>
        Train Logistic Regression, Random Forest, or XGBoost
        with configurable hyperparameters. View F1, ROC-AUC,
        confusion matrix, and feature importance. MLflow logging.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    <div class="feature-card">
        <h3>🔮 Predictions</h3>
        <p>
        Enter a new patient's details and get an instant
        treatment success prediction with confidence score
        from the trained model.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="feature-card">
        <h3>💬 AI Assistant</h3>
        <p>
        Ask natural language questions about the clinical
        trial data. Powered by OpenAI GPT with data context
        injected into prompts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Experiment Tracking</h3>
        <p>
        All ML runs logged to MLflow automatically.
        Compare model performance across experiments
        at localhost:5000.
        </p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

# ── Tech stack ────────────────────────────────────────────────────────────────
st.subheader("🛠️ Technology Stack")
tech_cols = st.columns(6)
tech = ["Python 3.10", "Streamlit", "Scikit-learn", "XGBoost", "Plotly", "MLflow"]
icons = ["🐍", "⚡", "🧠", "🚀", "📊", "📋"]
for col, name, icon in zip(tech_cols, tech, icons):
    col.markdown(f"""
    <div style="text-align:center; padding:1rem; background:#1a2744;
         border-radius:10px; border:1px solid #1a73e8;">
        <div style="font-size:2rem;">{icon}</div>
        <div style="color:#4fc3f7; font-size:0.85rem; font-weight:600;">{name}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Built by Banoth Rajesham · Data Scientist · PharmaML Analytics Suite v1.0")
