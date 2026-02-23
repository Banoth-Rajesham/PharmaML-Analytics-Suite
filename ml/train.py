"""
train.py
--------
Standalone Model Training Script

This script can be run directly from the command line to train and
evaluate all three models in one go. Useful for:
- CI/CD pipelines
- Scheduled retraining jobs
- Comparing all models at once

Usage:
    python ml/train.py

Author: Banoth Rajesham
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.db            import load_patients
from app.utils.preprocessing import run_full_pipeline
from app.utils.models        import (
    get_logistic_regression, get_random_forest, get_xgboost,
    train_model, evaluate_model, get_feature_importance,
)

import mlflow
import pandas as pd


def train_all_models():
    """
    Train all three models and log results to MLflow.
    Print a comparison table at the end.
    """
    print("=" * 60)
    print("  PharmaML — Model Training Script")
    print("=" * 60)

    # Load and preprocess data
    print("\n📂 Loading data...")
    df   = load_patients()
    data = run_full_pipeline(df, test_size=0.2)
    print(f"   Train: {len(data['y_train'])} | Test: {len(data['y_test'])}")

    # Define models to train
    models_to_train = {
        "Logistic Regression": (
            get_logistic_regression(C=1.0),
            data["X_train"],      # scaled
            data["X_test"],
        ),
        "Random Forest": (
            get_random_forest(n_estimators=100),
            data["X_train_df"],   # unscaled tree model
            data["X_test_df"],
        ),
        "XGBoost": (
            get_xgboost(learning_rate=0.1, n_estimators=100, max_depth=4),
            data["X_train_df"],
            data["X_test_df"],
        ),
    }

    # Set MLflow experiment
    mlflow.set_experiment("PharmaML_Standalone_Training")

    results_summary = []

    for name, (model, X_tr, X_te) in models_to_train.items():
        print(f"\n🤖 Training: {name} ...")

        with mlflow.start_run(run_name=name):
            # Train
            model = train_model(model, X_tr, data["y_train"])

            # Evaluate
            metrics = evaluate_model(model, X_te, data["y_test"])

            # Log to MLflow
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy",  metrics["accuracy"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall",    metrics["recall"])
            mlflow.log_metric("f1",        metrics["f1"])
            if metrics["roc_auc"] != "N/A":
                mlflow.log_metric("roc_auc", metrics["roc_auc"])

            results_summary.append({
                "Model"     : name,
                "Accuracy"  : metrics["accuracy"],
                "Precision" : metrics["precision"],
                "Recall"    : metrics["recall"],
                "F1"        : metrics["f1"],
                "ROC-AUC"   : metrics["roc_auc"],
            })

            print(f"   Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 60)
    results_df = pd.DataFrame(results_summary)
    print(results_df.to_string(index=False))
    print("\n✅ All models trained and logged to MLflow.")
    print("   Run `mlflow ui` and open http://localhost:5000 to view.")


if __name__ == "__main__":
    train_all_models()
