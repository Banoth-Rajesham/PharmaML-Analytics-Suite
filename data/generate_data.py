"""
generate_data.py
----------------
Synthetic Clinical Trial Data Generator.

This script creates a realistic synthetic dataset simulating pharmaceutical
clinical trial records. The data is saved to a SQLite database so the
Streamlit app can query it using SQL — just like a real data pipeline.

Author: Banoth Rajesham
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── constants ─────────────────────────────────────────────────────────────────
N_PATIENTS = 2000          # total synthetic patient records
DB_PATH    = Path(__file__).parent / "clinical_trials.db"

# Drug names we are testing (made-up pharma names)
DRUGS = ["Remivaxin", "Celoxaline", "Provental", "Therizone", "Novalix"]

# Clinical trial phases
PHASES = ["Phase I", "Phase II", "Phase III"]

# Disease areas (relevant to BMS — oncology, immunology, etc.)
DISEASE_AREAS = ["Oncology", "Cardiology", "Immunology", "Neurology", "Rare Disease"]

# Side effect severity levels
SIDE_EFFECT_LEVELS = ["None", "Mild", "Moderate", "Severe"]

def generate_patient_records(n: int) -> pd.DataFrame:
    """
    Generate n synthetic patient records.
    
    Each row represents one patient enrolled in a clinical trial.
    We use realistic distributions for age, vitals, and lab values.
    """
    # Patient demographics
    age = np.random.normal(loc=55, scale=15, size=n).clip(18, 90).astype(int)
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.52, 0.48])
    bmi = np.random.normal(loc=27, scale=5, size=n).clip(16, 45).round(1)

    # Trial assignment
    drug_assigned   = np.random.choice(DRUGS, size=n)
    trial_phase     = np.random.choice(PHASES, size=n, p=[0.2, 0.35, 0.45])
    disease_area    = np.random.choice(DISEASE_AREAS, size=n)
    dosage_mg       = np.random.choice([25, 50, 100, 200], size=n)

    # Lab values (simulate realistic clinical values)
    cholesterol     = np.random.normal(loc=200, scale=40, size=n).clip(100, 350).round(1)
    blood_pressure  = np.random.normal(loc=125, scale=20, size=n).clip(80, 200).round(1)
    hemoglobin      = np.random.normal(loc=13.5, scale=2, size=n).clip(8, 18).round(1)
    wbc_count       = np.random.normal(loc=7.0, scale=2, size=n).clip(2, 20).round(2)

    # Treatment outcomes ── target variable
    # Higher dose, younger patients, and Phase III → higher success probability
    success_prob = (
        0.3
        + 0.002 * (80 - age)               # younger = slight advantage
        + 0.0015 * dosage_mg               # higher dose = better coverage
        + 0.1 * (trial_phase == "Phase III")
        + np.random.normal(0, 0.1, size=n) # random noise
    ).clip(0.05, 0.95)

    treatment_success = (np.random.rand(n) < success_prob).astype(int)

    # Side effects — correlated with dosage slightly
    side_effect_prob = 0.15 + 0.001 * dosage_mg + np.random.normal(0, 0.05, n)
    side_effect_prob = side_effect_prob.clip(0, 1)
    side_effects = np.where(
        side_effect_prob < 0.5, "None",
        np.where(side_effect_prob < 0.70, "Mild",
        np.where(side_effect_prob < 0.85, "Moderate", "Severe"))
    )

    # Trial duration in days
    trial_duration_days = np.random.randint(30, 365, size=n)

    # Dropout flag (patients who left before trial end)
    dropout = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

    # Assemble into DataFrame
    df = pd.DataFrame({
        "patient_id"          : range(1, n + 1),
        "age"                 : age,
        "gender"              : gender,
        "bmi"                 : bmi,
        "disease_area"        : disease_area,
        "drug_assigned"       : drug_assigned,
        "trial_phase"         : trial_phase,
        "dosage_mg"           : dosage_mg,
        "cholesterol"         : cholesterol,
        "blood_pressure_mmhg" : blood_pressure,
        "hemoglobin_gdl"      : hemoglobin,
        "wbc_count_k_ul"      : wbc_count,
        "trial_duration_days" : trial_duration_days,
        "side_effects"        : side_effects,
        "dropout"             : dropout,
        "treatment_success"   : treatment_success,  # ← ML target variable
    })

    return df


def save_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """
    Save the DataFrame to a SQLite database.
    
    We create two tables:
    1. patients  — the main clinical records
    2. drug_info — metadata about each drug
    """
    conn = sqlite3.connect(db_path)

    # ── Table 1: patients ─────────────────────────────────────────────────────
    df.to_sql("patients", conn, if_exists="replace", index=False)
    print(f"✅ Saved {len(df)} patient records to table 'patients'")

    # ── Table 2: drug_info ────────────────────────────────────────────────────
    drug_info = pd.DataFrame({
        "drug_name"    : DRUGS,
        "category"     : ["Chemotherapy", "Immunotherapy", "Targeted Therapy",
                          "Gene Therapy", "Small Molecule"],
        "approval_year": [2018, 2020, 2019, 2022, 2021],
        "manufacturer" : ["BMS-Dev", "BMS-Dev", "BMS-Dev", "BMS-Dev", "BMS-Dev"],
        "mechanism"    : [
            "Inhibits tumor growth by blocking DNA replication",
            "Boosts immune system to attack cancer cells",
            "Blocks specific proteins that promote tumor growth",
            "Modifies defective genes responsible for disease",
            "Small molecule that inhibits a specific enzyme pathway",
        ],
    })
    drug_info.to_sql("drug_info", conn, if_exists="replace", index=False)
    print(f"✅ Saved {len(drug_info)} drug records to table 'drug_info'")

    conn.close()
    print(f"\n📁 Database saved at: {db_path}")


if __name__ == "__main__":
    print("🔬 Generating synthetic clinical trial data...")
    df = generate_patient_records(N_PATIENTS)

    print(f"\n📊 Dataset preview:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Treatment Success Rate: {df['treatment_success'].mean():.2%}")

    save_to_sqlite(df, DB_PATH)
    print("\n✅ Data generation complete!")
