"""
preprocessing.py
----------------
Data Cleaning and Feature Engineering Module.

This module handles all the data preparation steps needed before
machine learning. Clean, reproducible preprocessing is a key
skill for any Data Scientist — especially in pharma where data
quality directly impacts patient outcomes.

Steps covered:
    1. Drop rows with missing values (or impute)
    2. Encode categorical columns
    3. Scale numeric features
    4. Split into train / test sets

Author: Banoth Rajesham
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns not used as model features (identifiers, target, leakage)
DROP_COLS = ["patient_id", "treatment_success", "dropout"]

# Target column
TARGET = "treatment_success"

# Categorical columns that need encoding
CATEGORICAL_COLS = [
    "gender", "disease_area", "drug_assigned",
    "trial_phase", "side_effects"
]

# Numeric columns to scale
NUMERIC_COLS = [
    "age", "bmi", "cholesterol", "blood_pressure_mmhg",
    "hemoglobin_gdl", "wbc_count_k_ul", "trial_duration_days", "dosage_mg"
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning:
    - Drop duplicate rows
    - Fill missing numeric values with column median
    - Fill missing categorical values with 'Unknown'
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw patient records.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df = df.drop_duplicates()

    # Impute numeric columns with median (robust to outliers)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Impute categorical columns with 'Unknown'
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode all categorical columns.
    
    We use LabelEncoder for simplicity. In production you'd explore
    OneHotEncoding or TargetEncoding depending on model type.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical columns.
    dict
        Mapping of {column: LabelEncoder} for inverse transform later.
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le  # save encoder for later use (predictions)

    return df, encoders


def get_features_and_target(df: pd.DataFrame):
    """
    Split DataFrame into feature matrix X and target vector y.
    
    Parameters
    ----------
    df : pd.DataFrame
        Encoded DataFrame.
    
    Returns
    -------
    X : pd.DataFrame
        Feature columns.
    y : pd.Series
        Target column (treatment_success).
    feature_names : list
        List of feature column names.
    """
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, feature_cols


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize numeric features (mean=0, std=1).
    
    Important: We fit the scaler ONLY on training data to prevent
    data leakage — a common mistake beginners make.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test feature sets.
    
    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler : StandardScaler (fitted on train)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)  # ← use fit from train only
    return X_train_scaled, X_test_scaled, scaler


def run_full_pipeline(df: pd.DataFrame, test_size: float = 0.2):
    """
    Full end-to-end preprocessing pipeline.
    
    Combines all steps: clean → encode → split → scale.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw patient records.
    test_size : float
        Fraction of data reserved for testing (default 20%).
    
    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, scaler, encoders, feature_names
    """
    # Step 1 — Clean
    df_clean = clean_data(df)

    # Step 2 — Encode categories
    df_enc, encoders = encode_features(df_clean)

    # Step 3 — Split features / target
    X, y, feature_names = get_features_and_target(df_enc)

    # Step 4 — Train / test split (stratify keeps class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Step 5 — Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    return {
        "X_train"      : X_train_sc,
        "X_test"       : X_test_sc,
        "y_train"      : y_train,
        "y_test"       : y_test,
        "scaler"       : scaler,
        "encoders"     : encoders,
        "feature_names": feature_names,
        "X_train_df"   : X_train,   # unscaled, for tree models
        "X_test_df"    : X_test,
    }
