"""
visualizations.py
-----------------
Reusable Chart / Plot Helper Functions.

All charts use Plotly for interactive web-ready plots.
Each function takes a DataFrame and returns a Plotly figure
that Streamlit can render with `st.plotly_chart()`.

Author: Banoth Rajesham
"""

import pandas as pd
import numpy as np
import plotly.express       as px
import plotly.graph_objects as go
from plotly.subplots        import make_subplots


# ── Color palette (pharma-inspired blues/greens) ──────────────────────────────
COLORS = {
    "primary"  : "#1a73e8",
    "success"  : "#0f9d58",
    "warning"  : "#f4b400",
    "danger"   : "#d93025",
    "palette"  : px.colors.qualitative.Vivid,
}


def plot_success_by_drug(df: pd.DataFrame) -> go.Figure:
    """Bar chart: Average treatment success rate per drug."""
    summary = (
        df.groupby("drug_assigned")["treatment_success"]
        .mean()
        .reset_index()
        .rename(columns={"treatment_success": "success_rate"})
        .sort_values("success_rate", ascending=False)
    )
    summary["success_rate_pct"] = (summary["success_rate"] * 100).round(1)

    fig = px.bar(
        summary,
        x="drug_assigned",
        y="success_rate_pct",
        color="success_rate_pct",
        color_continuous_scale="Blues",
        title="💊 Treatment Success Rate by Drug",
        labels={"drug_assigned": "Drug Name", "success_rate_pct": "Success Rate (%)"},
        text="success_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(coloraxis_showscale=False, showlegend=False)
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram: Patient age distribution, split by treatment outcome."""
    fig = px.histogram(
        df,
        x="age",
        color="treatment_success",
        barmode="overlay",
        title="👥 Patient Age Distribution by Outcome",
        labels={"age": "Patient Age", "treatment_success": "Treatment Success"},
        color_discrete_map={0: COLORS["danger"], 1: COLORS["success"]},
        opacity=0.75,
        nbins=30,
    )
    fig.update_layout(legend_title_text="Success (1=Yes, 0=No)")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: Pearson correlation between numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix  = df[numeric_cols].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        title="🔗 Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True,
        aspect="auto",
    )
    fig.update_traces(textfont_size=10)
    return fig


def plot_trial_phase_success(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: Success rate by trial phase and drug."""
    summary = (
        df.groupby(["trial_phase", "drug_assigned"])["treatment_success"]
        .mean()
        .mul(100).round(1)
        .reset_index()
        .rename(columns={"treatment_success": "success_rate"})
    )
    fig = px.bar(
        summary,
        x="trial_phase",
        y="success_rate",
        color="drug_assigned",
        barmode="group",
        title="📊 Success Rate by Trial Phase & Drug",
        labels={"trial_phase": "Trial Phase", "success_rate": "Success Rate (%)",
                "drug_assigned": "Drug"},
        color_discrete_sequence=COLORS["palette"],
    )
    return fig


def plot_bmi_vs_success(df: pd.DataFrame) -> go.Figure:
    """Box plot: BMI distribution by treatment success."""
    df_copy = df.copy()
    df_copy["Outcome"] = df_copy["treatment_success"].map({1: "Success ✅", 0: "Failure ❌"})

    fig = px.box(
        df_copy,
        x="Outcome",
        y="bmi",
        color="Outcome",
        title="⚖️ BMI Distribution by Treatment Outcome",
        labels={"bmi": "BMI"},
        color_discrete_map={"Success ✅": COLORS["success"], "Failure ❌": COLORS["danger"]},
        points="outliers",
    )
    return fig


def plot_confusion_matrix(cm: list, labels=["No Success", "Success"]) -> go.Figure:
    """Heatmap: Confusion matrix from model evaluation."""
    cm_array = np.array(cm)
    fig = px.imshow(
        cm_array,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title="🔲 Confusion Matrix",
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_traces(textfont_size=18)
    return fig


def plot_feature_importance(importance_dict: dict, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart: Top N most important features."""
    items  = list(importance_dict.items())[:top_n]
    names  = [i[0] for i in items][::-1]
    scores = [i[1] for i in items][::-1]

    fig = go.Figure(go.Bar(
        x=scores, y=names,
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"⭐ Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
    )
    return fig


def plot_roc_curve(y_test, y_prob) -> go.Figure:
    """
    ROC Curve plot showing model discrimination ability.
    AUC closer to 1.0 = better model.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {roc_auc:.3f})",
        line=dict(color=COLORS["primary"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="gray", dash="dash"),
    ))
    fig.update_layout(
        title="📈 ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        width=600, height=400,
    )
    return fig


def plot_dosage_side_effects(df: pd.DataFrame) -> go.Figure:
    """Side-by-side: Side effect severity distribution per dosage."""
    counts = (
        df.groupby(["dosage_mg", "side_effects"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        counts,
        x="dosage_mg",
        y="count",
        color="side_effects",
        barmode="stack",
        title="💊 Side Effect Severity by Dosage (mg)",
        labels={"dosage_mg": "Dosage (mg)", "count": "Patient Count", "side_effects": "Side Effects"},
        color_discrete_map={
            "None": "#0f9d58", "Mild": "#f4b400",
            "Moderate": "#ff6d00", "Severe": "#d93025"
        },
    )
    return fig
