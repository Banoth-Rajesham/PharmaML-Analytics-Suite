"""
5_ai_assistant.py
-----------------
Page 5 — AI Data Assistant (GenAI Integration)

This page implements a chatbot that answers natural language questions
about the clinical trial dataset. It:
1. Loads a snapshot of the dataset into context
2. Uses OpenAI GPT-4 (or GPT-3.5) to answer questions
3. Falls back to a local rule-based system if no API key is set

This demonstrates GenAI / LLM integration — a key "Good to Have" skill
from the BMS Data Scientist JD.

Author: Banoth Rajesham
"""

import streamlit as st
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.db import load_patients

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Assistant | PharmaML", page_icon="💬", layout="wide")

st.title("💬 AI Data Assistant")
st.markdown("""
Ask natural language questions about the clinical trial dataset.  
The AI will analyze the data and give you data-driven answers.
""")

# ── API Key (optional) ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your OpenAI API key (optional):",
        type="password",
        help="Leave blank to use the built-in rule-based fallback assistant.",
        placeholder="sk-..."
    )
    if api_key:
        st.success("✅ OpenAI key set")
    else:
        st.info("ℹ️ Using local fallback assistant (no API needed)")

    st.markdown("---")
    st.markdown("**Sample Questions:**")
    examples = [
        "Which drug has the highest success rate?",
        "What is the average age of patients?",
        "How many patients are in Phase III?",
        "Which disease area has the most patients?",
        "What percentage of patients had severe side effects?",
        "What is the success rate for Remivaxin?",
        "How many patients dropped out?",
    ]
    for eq in examples:
        st.markdown(f"- *{eq}*")


# ── Load Data Context ─────────────────────────────────────────────────────────
@st.cache_data
def load_data_context():
    """
    Build a concise data summary to inject into the LLM prompt.
    We don't send ALL 2000 rows — just statistical summaries (token efficient).
    """
    df = load_patients()
    if df.empty:
        return "No data available.", df

    summary_lines = []
    summary_lines.append(f"Dataset: Clinical Trial Records")
    summary_lines.append(f"Total patients: {len(df)}")
    summary_lines.append(f"Overall success rate: {df['treatment_success'].mean()*100:.1f}%")
    summary_lines.append(f"Dropout rate: {df['dropout'].mean()*100:.1f}%")
    summary_lines.append(f"Average age: {df['age'].mean():.1f} years")
    summary_lines.append(f"Average BMI: {df['bmi'].mean():.1f}")
    summary_lines.append("")

    # Per drug stats
    drug_stats = (
        df.groupby("drug_assigned")
        .agg(count=("patient_id", "count"), success_rate=("treatment_success", "mean"))
        .round(3)
    )
    summary_lines.append("Drug success rates:")
    for drug, row in drug_stats.iterrows():
        summary_lines.append(f"  {drug}: {row['success_rate']*100:.1f}% ({int(row['count'])} patients)")

    summary_lines.append("")

    # Phase stats
    phase_stats = (
        df.groupby("trial_phase")
        .agg(count=("patient_id", "count"), success_rate=("treatment_success", "mean"))
        .round(3)
    )
    summary_lines.append("Trial phase breakdown:")
    for phase, row in phase_stats.iterrows():
        summary_lines.append(f"  {phase}: {row['success_rate']*100:.1f}% success ({int(row['count'])} patients)")

    summary_lines.append("")

    # Disease area stats
    disease_stats = (
        df.groupby("disease_area")["patient_id"].count()
    )
    summary_lines.append("Patients by disease area:")
    for disease, count in disease_stats.items():
        summary_lines.append(f"  {disease}: {count} patients")

    summary_lines.append("")

    # Side effects
    side_counts = df["side_effects"].value_counts()
    summary_lines.append("Side effect distribution:")
    for level, count in side_counts.items():
        summary_lines.append(f"  {level}: {count} patients ({count/len(df)*100:.1f}%)")

    return "\n".join(summary_lines), df


data_context, df_raw = load_data_context()


# ── Rule-based fallback system ────────────────────────────────────────────────
def rule_based_answer(question: str, df) -> str:
    """
    Simple keyword-based Q&A system as fallback when no OpenAI key is set.
    Covers the most common questions a stakeholder might ask.
    """
    q = question.lower()

    if "highest success" in q or "best drug" in q or "most effective" in q:
        drug_success = df.groupby("drug_assigned")["treatment_success"].mean().sort_values(ascending=False)
        best = drug_success.index[0]
        rate = drug_success.iloc[0] * 100
        return f"📊 **{best}** has the highest success rate at **{rate:.1f}%** across all trial phases."

    elif "average age" in q or "mean age" in q:
        avg = df["age"].mean()
        return f"👥 The average patient age in the dataset is **{avg:.1f} years** (range: {df['age'].min()}–{df['age'].max()})."

    elif "phase iii" in q or "phase 3" in q:
        phase_df = df[df["trial_phase"] == "Phase III"]
        return f"🔬 There are **{len(phase_df):,}** patients enrolled in **Phase III** trials, with a success rate of **{phase_df['treatment_success'].mean()*100:.1f}%**."

    elif "disease area" in q and ("most" in q or "largest" in q or "highest" in q):
        top_disease = df["disease_area"].value_counts().idxmax()
        count       = df["disease_area"].value_counts().max()
        return f"🏥 **{top_disease}** has the most patients with **{count:,}** enrolled."

    elif "severe side effect" in q or "severe" in q:
        severe_count = (df["side_effects"] == "Severe").sum()
        pct = severe_count / len(df) * 100
        return f"⚠️ **{severe_count}** patients ({pct:.1f}%) experienced **severe side effects**."

    elif "dropout" in q or "drop out" in q or "dropped" in q:
        dropouts = df["dropout"].sum()
        pct = dropouts / len(df) * 100
        return f"📉 **{dropouts}** patients ({pct:.1f}%) dropped out before completing their trial."

    elif "total patient" in q or "how many patient" in q:
        return f"👥 There are **{len(df):,}** total patient records in the clinical trial database."

    elif "success rate" in q and "overall" in q:
        return f"✅ The overall treatment success rate across all drugs and phases is **{df['treatment_success'].mean()*100:.1f}%**."

    elif "remivaxin" in q:
        sub = df[df["drug_assigned"] == "Remivaxin"]
        return f"💊 **Remivaxin**: {len(sub)} patients enrolled, success rate = **{sub['treatment_success'].mean()*100:.1f}%**."

    elif "celoxaline" in q:
        sub = df[df["drug_assigned"] == "Celoxaline"]
        return f"💊 **Celoxaline**: {len(sub)} patients enrolled, success rate = **{sub['treatment_success'].mean()*100:.1f}%**."

    elif "average bmi" in q or "mean bmi" in q:
        return f"⚖️ Average BMI of patients is **{df['bmi'].mean():.1f}** (range: {df['bmi'].min():.1f}–{df['bmi'].max():.1f})."

    elif "cholesterol" in q:
        return f"🩺 Average cholesterol: **{df['cholesterol'].mean():.1f} mg/dL** (range: {df['cholesterol'].min():.0f}–{df['cholesterol'].max():.0f})."

    else:
        return (
            f"🤔 I couldn't find a specific answer to that. Here's a quick summary:\n\n"
            f"- **{len(df):,}** patients across {df['drug_assigned'].nunique()} drugs\n"
            f"- Overall success rate: **{df['treatment_success'].mean()*100:.1f}%**\n"
            f"- Average age: **{df['age'].mean():.1f}** years\n\n"
            f"Try asking about: *drug success rates, patient counts, trial phases, side effects, dropouts, or specific drugs.*"
        )


# ── OpenAI-powered answer ─────────────────────────────────────────────────────
def openai_answer(question: str, context: str, key: str) -> str:
    """
    Use OpenAI GPT to answer questions using dataset context.
    We inject a statistical summary into the system prompt.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        system_prompt = f"""
You are a data analyst assistant for a pharmaceutical clinical trial analytics platform.
You have access to the following dataset summary:

{context}

Answer the user's question based ONLY on this data. Be concise, specific, and give numbers.
If you cannot answer from the data, say so clearly.
Format your responses with markdown (bold numbers, bullet points).
        """

        response = client.chat.completions.create(
            model    = "gpt-3.5-turbo",
            messages = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": question},
            ],
            temperature = 0.3,
            max_tokens  = 500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ OpenAI API error: {e}\n\nFalling back to local assistant:\n\n" + rule_based_answer(question, df_raw)


# ── Chat Interface ─────────────────────────────────────────────────────────────
# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Show chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about the clinical trial data...")

if user_input:
    # Add user message to history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing data..."):
            if api_key:
                answer = openai_answer(user_input, data_context, api_key)
            else:
                answer = rule_based_answer(user_input, df_raw)
        st.markdown(answer)

    # Save assistant response
    st.session_state["chat_history"].append({"role": "assistant", "content": answer})


# ── Clear chat ─────────────────────────────────────────────────────────────────
if st.session_state["chat_history"]:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

st.markdown("---")
st.markdown("**Data Context Summary (injected into AI prompt):**")
with st.expander("📋 View dataset summary (what the AI sees)"):
    st.text(data_context)
