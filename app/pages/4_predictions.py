"""
4_predictions.py
----------------
Page 4 — Patient Outcome Prediction

Enter a new patient's clinical data and get an instant
treatment success prediction from the trained model.

This page uses the model saved in session state from
the ML Pipeline page (Page 3).

Author: Banoth Rajesham
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictions | PharmaML", page_icon="🔮", layout="wide")

st.title("🔮 Patient Outcome Prediction")
st.markdown("Enter a patient's clinical data to predict treatment success probability.")

# ── Check for trained model ────────────────────────────────────────────────────
if "trained_model" not in st.session_state:
    st.warning("""
    ⚠️ **No trained model found!**  
    Please go to the **🤖 ML Pipeline** page, configure a model, and click **Train Model** first.
    Then come back here to make predictions.
    """)
    st.stop()

model      = st.session_state["trained_model"]
data       = st.session_state["model_data"]
model_name = st.session_state["model_name"]
use_scaler = st.session_state["model_scaled"]

st.success(f"✅ Using trained **{model_name}** model")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("🧑‍⚕️ Enter Patient Clinical Data")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age       = st.slider("Age",        18, 90,  55)
        gender    = st.selectbox("Gender",  ["Male", "Female"])
        bmi       = st.slider("BMI",        16.0, 45.0, 27.0, 0.5)

    with col2:
        st.markdown("**Trial Information**")
        drug_assigned = st.selectbox("Drug Assigned",  ["Remivaxin", "Celoxaline", "Provental", "Therizone", "Novalix"])
        trial_phase   = st.selectbox("Trial Phase",    ["Phase I", "Phase II", "Phase III"])
        disease_area  = st.selectbox("Disease Area",   ["Oncology", "Cardiology", "Immunology", "Neurology", "Rare Disease"])
        dosage_mg     = st.selectbox("Dosage (mg)",    [25, 50, 100, 200])

    with col3:
        st.markdown("**Lab Values**")
        cholesterol          = st.slider("Cholesterol",          100.0, 350.0, 200.0, 1.0)
        blood_pressure_mmhg  = st.slider("Blood Pressure (mmHg)", 80.0, 200.0, 125.0, 1.0)
        hemoglobin_gdl       = st.slider("Hemoglobin (g/dL)",     8.0,  18.0,  13.5,  0.1)
        wbc_count_k_ul       = st.slider("WBC Count (K/μL)",      2.0,  20.0,   7.0,  0.1)
        trial_duration_days  = st.slider("Trial Duration (days)", 30,   365,   180,   10)
        side_effects         = st.selectbox("Side Effects",  ["None", "Mild", "Moderate", "Severe"])

    submitted = st.form_submit_button("🔮 Predict Outcome", type="primary", use_container_width=True)


# ── Prediction Logic ──────────────────────────────────────────────────────────
if submitted:
    # Build the input row as a DataFrame matching training columns
    # We need to encode categoricals the same way as training

    encoders = data["encoders"]
    feature_names = data["feature_names"]

    # Create raw input dict
    raw_input = {
        "age"                 : age,
        "gender"              : gender,
        "bmi"                 : bmi,
        "disease_area"        : disease_area,
        "drug_assigned"       : drug_assigned,
        "trial_phase"         : trial_phase,
        "dosage_mg"           : dosage_mg,
        "cholesterol"         : cholesterol,
        "blood_pressure_mmhg" : blood_pressure_mmhg,
        "hemoglobin_gdl"      : hemoglobin_gdl,
        "wbc_count_k_ul"      : wbc_count_k_ul,
        "trial_duration_days" : trial_duration_days,
        "side_effects"        : side_effects,
    }

    input_df = pd.DataFrame([raw_input])

    # Encode categoricals using saved encoders
    CATEGORICAL_COLS = ["gender", "disease_area", "drug_assigned", "trial_phase", "side_effects"]
    for col in CATEGORICAL_COLS:
        if col in encoders:
            le = encoders[col]
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except ValueError:
                # If unseen label, use 0
                input_df[col] = 0

    # Select only the feature columns used in training
    input_features = input_df[feature_names]

    # Scale if model requires it (Logistic Regression)
    if use_scaler:
        scaler         = data["scaler"]
        input_features = scaler.transform(input_features)

    # Predict
    prediction = model.predict(input_features)[0]
    try:
        probability = model.predict_proba(input_features)[0]
        prob_success = probability[1]
        prob_fail    = probability[0]
    except AttributeError:
        prob_success = float(prediction)
        prob_fail    = 1.0 - prob_success

    # ── Display Prediction ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    result_col, gauge_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.success(f"""
            ## ✅ Treatment Likely Successful
            **Confidence:** {prob_success*100:.1f}%
            
            The model predicts this patient has a **{prob_success*100:.1f}%** probability 
            of responding positively to **{drug_assigned}** ({trial_phase}).
            """)
        else:
            st.error(f"""
            ## ❌ Treatment May Not Succeed
            **Confidence:** {prob_fail*100:.1f}%
            
            The model predicts this patient has a **{prob_fail*100:.1f}%** probability 
            that **{drug_assigned}** ({trial_phase}) will NOT be effective.
            """)

    with gauge_col:
        import plotly.graph_objects as go
        fig_gauge = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob_success * 100,
            title = {"text": "Success Probability (%)"},
            gauge = {
                "axis"  : {"range": [0, 100]},
                "bar"   : {"color": "#1a73e8"},
                "steps" : [
                    {"range": [0,  40], "color": "#d93025"},
                    {"range": [40, 70], "color": "#f4b400"},
                    {"range": [70, 100],"color": "#0f9d58"},
                ],
                "threshold": {
                    "line" : {"color": "white", "width": 4},
                    "value": 50,
                },
            },
            delta = {"reference": 50, "increasing": {"color": "#0f9d58"}},
            number = {"suffix": "%"},
        ))
        fig_gauge.update_layout(height=300, paper_bgcolor="#0a1628", font_color="#e8f4fd")
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Input Summary ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Patient Data Summary")
    summary_df = pd.DataFrame([raw_input]).T.reset_index()
    summary_df.columns = ["Feature", "Value"]
    st.dataframe(summary_df, use_container_width=True, height=250)

    st.caption(f"Model used: {model_name} | Prediction: {'Success' if prediction==1 else 'Failure'} | Confidence: {max(prob_success, prob_fail)*100:.1f}%")
