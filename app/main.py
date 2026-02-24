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

/* ============================= */
/* Import Font */
/* ============================= */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* ============================= */
/* Main App Background */
/* ============================= */

.stApp {
    background:
        linear-gradient(rgba(8,15,30,0.92), rgba(8,15,30,0.96)),
        url("https://images.unsplash.com/photo-1581093458791-9d2d3f6a7b3c?q=80&w=2000");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    color: #e6edf3;
}

/* ============================= */
/* Sidebar */
/* ============================= */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0b1220 100%);
    padding: 24px;
    border-right: 1px solid rgba(255,255,255,0.06);
    box-shadow: 6px 0 25px rgba(0,0,0,0.5);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] button:hover {
    background: rgba(59,130,246,0.15);
    border-radius: 10px;
    transition: 0.3s ease;
}

/* ============================= */
/* Header */
/* ============================= */

.main-header {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(16px);
    padding: 2rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    border: 1px solid rgba(59,130,246,0.3);
    box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}

/* ============================= */
/* Metric Cards */
/* ============================= */

[data-testid="stMetric"] {
    background: rgba(17,24,39,0.65);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(59,130,246,0.25);
    box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    transition: all 0.25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-6px);
    border-color: #3b82f6;
}

[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    color: #94a3b8 !important;
}

/* ============================= */
/* Feature Cards */
/* ============================= */

.feature-card {
    background: rgba(17,24,39,0.6);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid rgba(59,130,246,0.25);
    transition: all 0.25s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-6px);
    border-color: #60a5fa;
    box-shadow: 0 10px 35px rgba(59,130,246,0.25);
}

/* ============================= */
/* Buttons */
/* ============================= */

.stButton>button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(37,99,235,0.5);
}

/* ============================= */
/* Inputs */
/* ============================= */

.stTextInput>div>div>input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
}

/* ============================= */
/* Tables */
/* ============================= */

[data-testid="stDataFrame"] {
    background: rgba(17,24,39,0.7);
    border-radius: 12px;
    border: 1px solid rgba(59,130,246,0.2);
}

/* ============================= */
/* Typography */
/* ============================= */

h1 {
    color: #60a5fa;
    font-weight: 700;
}

h2, h3 {
    color: #93c5fd;
}

p, li {
    color: #cbd5e1;
}

hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, #3b82f6, transparent);
}

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
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("👥 Total Patients",    f"{stats['total_patients']:,}")
        c2.metric("✅ Avg Success Rate",   f"{stats['avg_success_rate']}%")
        c3.metric("💊 Drugs in Trials",   stats["n_drugs"])
        c4.metric("🔬 Trial Phases",       stats["n_phases"])
        c5.metric("🔬 work Phases",       stats["n_phase"])
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
