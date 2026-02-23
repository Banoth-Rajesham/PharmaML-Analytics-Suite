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
from pathlib import Path

# Path to the SQLite database (relative to this file)
DB_PATH = Path(__file__).parent.parent.parent / "data" / "clinical_trials.db"


def get_connection() -> sqlite3.Connection:
    """Return a live SQLite connection to the clinical trials database."""
    return sqlite3.connect(DB_PATH)


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute any SQL SELECT query and return results as a DataFrame.
    
    This is great for the interactive SQL console in the Data Overview page.
    
    Parameters
    ----------
    sql : str
        A valid SQL SELECT statement.
    
    Returns
    -------
    pd.DataFrame
        Query results. Empty DataFrame if query fails.
    """
    try:
        conn = get_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[DB ERROR] Query failed: {e}")
        return pd.DataFrame()


def load_patients(filters: dict = None) -> pd.DataFrame:
    """
    Load all patient records, with optional column-value filters.
    
    Parameters
    ----------
    filters : dict, optional
        E.g. {"trial_phase": "Phase III", "disease_area": "Oncology"}
    
    Returns
    -------
    pd.DataFrame
        Filtered patient records.
    
    Example
    -------
    >>> df = load_patients({"trial_phase": "Phase III"})
    """
    sql = "SELECT * FROM patients"
    conditions = []

    if filters:
        for col, val in filters.items():
            if val and val != "All":
                conditions.append(f"{col} = '{val}'")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

    return run_query(sql)


def load_drug_info() -> pd.DataFrame:
    """Load the drug metadata table."""
    return run_query("SELECT * FROM drug_info")


def get_summary_stats() -> dict:
    """
    Return high-level summary statistics for the dashboard header cards.
    
    Returns
    -------
    dict
        Keys: total_patients, avg_success_rate, n_drugs, n_phases
    """
    conn = get_connection()

    total_patients  = pd.read_sql_query("SELECT COUNT(*) as n FROM patients", conn).iloc[0, 0]
    avg_success     = pd.read_sql_query("SELECT AVG(treatment_success) as s FROM patients", conn).iloc[0, 0]
    n_drugs         = pd.read_sql_query("SELECT COUNT(DISTINCT drug_assigned) as n FROM patients", conn).iloc[0, 0]
    n_phases        = pd.read_sql_query("SELECT COUNT(DISTINCT trial_phase) as n FROM patients", conn).iloc[0, 0]

    conn.close()

    return {
        "total_patients"  : int(total_patients),
        "avg_success_rate": round(float(avg_success) * 100, 1),
        "n_drugs"         : int(n_drugs),
        "n_phases"        : int(n_phases),
    }
