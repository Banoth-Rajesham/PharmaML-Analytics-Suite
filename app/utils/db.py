"""
db.py
-----
Database helper utilities for the PharmaML Analytics Suite.

This module handles all SQLite connections and provides reusable
functions to query patient and drug data. Using SQLAlchemy makes
it easy to swap SQLite for a cloud database (e.g. BigQuery on GCP)
without changing app code — a real-world best practice.

Author: Banoth Rajesham
"""

import sqlite3
import pandas as pd
import subprocess
import sys
from pathlib import Path

# Path to the SQLite database (relative to this file)
DB_PATH = Path(__file__).parent.parent.parent / "data" / "clinical_trials.db"


def get_connection() -> sqlite3.Connection:
    """Return a live SQLite connection to the clinical trials database."""
    return sqlite3.connect(DB_PATH)


def check_and_init_db():
    """Check if database exists and has tables. If not, run generation script."""
    import streamlit as st
    
    db_needs_init = False
    if not DB_PATH.exists():
        db_needs_init = True
    else:
        # Check if tables exist
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
            if not cursor.fetchone():
                db_needs_init = True
            conn.close()
        except:
            db_needs_init = True

    if db_needs_init:
        try:
            gen_path = Path(__file__).parent.parent.parent / "data" / "generate_data.py"
            # Use sys.executable to ensure we use the same python environment
            subprocess.run([sys.executable, str(gen_path)], check=True)
            return True
        except Exception as e:
            st.error(f"Failed to initialize database: {e}")
            return False
    return True


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute any SQL SELECT query and return results as a DataFrame.
    
    This is great for the interactive SQL console in the Data Overview page.
    """
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        # We don't print to console here to avoid cluttering streamlit logs
        return pd.DataFrame()


def load_patients(filters: dict = None) -> pd.DataFrame:
    """
    Load all patient records, with optional column-value filters.
    """
    sql = "SELECT * FROM patients"
    conditions = []

    if filters:
        for col, val in filters.items():
            if val and val != "All":
                conditions.append(f"{col} = '{val}'")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

    df = run_query(sql)
    return df


def load_drug_info() -> pd.DataFrame:
    """Load the drug metadata table."""
    return run_query("SELECT * FROM drug_info")


def get_summary_stats() -> dict:
    """
    Return high-level summary statistics for the dashboard header cards.
    """
    try:
        conn = get_connection()
        total_patients  = pd.read_sql_query("SELECT COUNT(*) as n FROM patients", conn).iloc[0, 0]
        avg_success     = pd.read_sql_query("SELECT AVG(treatment_success) as s FROM patients", conn).iloc[0, 0]
        n_drugs         = pd.read_sql_query("SELECT COUNT(DISTINCT drug_assigned) as n FROM patients", conn).iloc[0, 0]
        n_phases        = pd.read_sql_query("SELECT COUNT(DISTINCT trial_phase) as n FROM patients", conn).iloc[0, 0]
        conn.close()

        return {
            "total_patients"  : int(total_patients) if total_patients else 0,
            "avg_success_rate": round(float(avg_success) * 100, 1) if avg_success else 0.0,
            "n_drugs"         : int(n_drugs) if n_drugs else 0,
            "n_phases"        : int(n_phases) if n_phases else 0,
        }
    except:
        return {
            "total_patients"  : 0,
            "avg_success_rate": 0.0,
            "n_drugs"         : 0,
            "n_phases"        : 0,
        }
