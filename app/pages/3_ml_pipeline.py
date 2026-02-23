"""
3_ml_pipeline.py
----------------
Page 3 — Machine Learning Pipeline

This page lets users:
1. Select a classification model
2. Configure hyperparameters via UI sliders
3. Train the model and see evaluation metrics
4. View confusion matrix, ROC curve, feature importance
5. All runs are automatically logged to MLflow

This demonstrates: ML lifecycle, model comparison, MLOps basics.

Author: Banoth Rajesham
"""

import streamlit as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.db             import load_patients
from app.utils.preprocessing  import run_full_pipeline
from app.utils.models         import (
    get_logistic_regression, get_random_forest, get_xgboost,
    train_model, evaluate_model, get_feature_importance,
)
from app.utils.visualizations import (
    plot_confusion_matrix, plot_feature_importance, plot_roc_curve,
)

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Pipeline | PharmaML", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Pipeline")
st.markdown("Train and evaluate ML models to predict clinical trial treatment success.")

# ── Sidebar: model config ─────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.radio(
    "Select Model:",
    ["Logistic Regression", "Random Forest", "XGBoost"],
)

test_size = st.sidebar.slider("🧪 Test Split (%)", min_value=10, max_value=40, value=20, step=5) / 100

st.sidebar.markdown("---")

# Model-specific hyperparameters
if model_choice == "Logistic Regression":
    st.sidebar.subheader("Logistic Regression Params")
    lr_C       = st.sidebar.select_slider("Regularization (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
    lr_iter    = st.sidebar.slider("Max Iterations", 100, 1000, 500, 100)

elif model_choice == "Random Forest":
    st.sidebar.subheader("Random Forest Params")
    rf_trees   = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
    rf_depth   = st.sidebar.select_slider("Max Depth", options=[3, 5, 10, 15, None], value=None)

else:  # XGBoost
    st.sidebar.subheader("XGBoost Params")
    xgb_lr     = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    xgb_trees  = st.sidebar.slider("Num Estimators", 50, 500, 100, 50)
    xgb_depth  = st.sidebar.slider("Max Depth", 2, 8, 4, 1)


# ── Train button ──────────────────────────────────────────────────────────────
st.markdown(f"### 🚀 Training: **{model_choice}**")

if st.button("▶️ Train Model", type="primary", use_container_width=True):

    # Step 1 — Load data
    with st.spinner("📂 Loading data from database..."):
        df = load_patients()
        st.success(f"✅ Loaded {len(df):,} patient records")

    # Step 2 — Preprocess
    with st.spinner("🔧 Preprocessing data (clean → encode → scale)..."):
        data = run_full_pipeline(df, test_size=test_size)
        st.success(f"✅ Train: {len(data['y_train'])} samples | Test: {len(data['y_test'])} samples")

    # Step 3 — Build model
    if model_choice == "Logistic Regression":
        model = get_logistic_regression(C=lr_C, max_iter=lr_iter)
        # Logistic regression uses scaled features
        X_train = data["X_train"]
        X_test  = data["X_test"]
    elif model_choice == "Random Forest":
        model = get_random_forest(n_estimators=rf_trees, max_depth=rf_depth)
        # Tree models use unscaled features
        X_train = data["X_train_df"]
        X_test  = data["X_test_df"]
    else:
        model = get_xgboost(learning_rate=xgb_lr, n_estimators=xgb_trees, max_depth=xgb_depth)
        X_train = data["X_train_df"]
        X_test  = data["X_test_df"]

    # Step 4 — Train
    with st.spinner(f"🤖 Training {model_choice}..."):
        t_start = time.time()
        model   = train_model(model, X_train, data["y_train"])
        t_end   = time.time()
        train_time = round(t_end - t_start, 2)

    # Step 5 — Evaluate
    results = evaluate_model(model, X_test, data["y_test"])

    # Step 6 — MLflow Logging
    try:
        import mlflow
        mlflow.set_experiment("PharmaML_Clinical_Trials")
        with mlflow.start_run(run_name=model_choice):
            mlflow.log_param("model",      model_choice)
            mlflow.log_param("test_size",  test_size)
            mlflow.log_metric("accuracy",  results["accuracy"])
            mlflow.log_metric("precision", results["precision"])
            mlflow.log_metric("recall",    results["recall"])
            mlflow.log_metric("f1",        results["f1"])
            if results["roc_auc"] != "N/A":
                mlflow.log_metric("roc_auc", results["roc_auc"])
        mlflow_logged = True
    except Exception:
        mlflow_logged = False

    st.success(f"✅ Model trained in {train_time}s {'| Logged to MLflow ✅' if mlflow_logged else ''}")

    # ── Results section ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Model Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🎯 Accuracy",  f"{results['accuracy']*100:.1f}%")
    m2.metric("📐 Precision", f"{results['precision']*100:.1f}%")
    m3.metric("🔁 Recall",    f"{results['recall']*100:.1f}%")
    m4.metric("⚖️ F1 Score", f"{results['f1']*100:.1f}%")
    m5.metric("📈 ROC-AUC",   str(results["roc_auc"]))

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔲 Confusion Matrix")
        fig_cm = plot_confusion_matrix(results["confusion_matrix"])
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("""
        **Rows** = Actual labels · **Cols** = Predicted labels  
        Top-left = True Negatives · Bottom-right = True Positives
        """)

    with col2:
        if results["roc_auc"] != "N/A":
            st.subheader("📈 ROC Curve")
            fig_roc = plot_roc_curve(data["y_test"], results["y_prob"])
            st.plotly_chart(fig_roc, use_container_width=True)
            st.caption("AUC = 0.5 (random) → 1.0 (perfect). Ours should be > 0.7 to be useful.")
        else:
            st.info("ROC curve not available for this model.")

    st.markdown("---")

    # Feature importance
    st.subheader("⭐ Feature Importance")
    importance = get_feature_importance(model, data["feature_names"])
    if importance:
        fig_imp = plot_feature_importance(importance, top_n=10)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown("**Top features affecting treatment success prediction →**")
        top_3 = list(importance.keys())[:3]
        for i, feat in enumerate(top_3, 1):
            st.markdown(f"{i}. **{feat}** — importance: `{importance[feat]:.4f}`")
    else:
        st.info("Feature importance not available for this model.")

    # Save model to session state for prediction page
    st.session_state["trained_model"]   = model
    st.session_state["model_data"]      = data
    st.session_state["model_name"]      = model_choice
    st.session_state["model_scaled"]    = (model_choice == "Logistic Regression")
    st.success("✅ Model saved! Go to 🔮 Predictions page to test it on new patients.")

else:
    st.info("👆 Configure the model in the sidebar, then click **Train Model** to start.")

    # Show pipeline diagram explanation
    st.markdown("### 🔄 Pipeline Steps")
    steps = [
        ("1️⃣ Data Loading",      "Load patient records from SQLite database"),
        ("2️⃣ Data Cleaning",      "Remove duplicates, handle missing values"),
        ("3️⃣ Feature Encoding",   "Convert categorical cols to numbers (LabelEncoder)"),
        ("4️⃣ Train/Test Split",   "80/20 split with stratification"),
        ("5️⃣ Feature Scaling",    "StandardScaler — mean=0, std=1"),
        ("6️⃣ Model Training",     "Fit selected model on training data"),
        ("7️⃣ Evaluation",         "Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("8️⃣ MLflow Logging",     "Automatic experiment tracking"),
    ]
    for step, desc in steps:
        st.markdown(f"- **{step}**: {desc}")

st.markdown("---")
st.caption("💡 MLflow UI: run `mlflow ui` in terminal and visit http://localhost:5000")
