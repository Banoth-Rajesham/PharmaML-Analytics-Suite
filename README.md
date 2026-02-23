# 💊 PharmaML Analytics Suite

> An end-to-end Data Science platform for Clinical Trial Data Analysis, Machine Learning, and AI-Powered Insights — built with Python, Streamlit, Scikit-learn, and SQLite.

---

## 🎯 Project Overview

PharmaML Analytics Suite is a **production-ready, interactive data science application** that simulates real-world pharmaceutical analytics workflows. It covers the full data science lifecycle:

1. **Data Ingestion & Storage** — SQLite database backend simulating clinical trial records
2. **Exploratory Data Analysis (EDA)** — Rich interactive charts and statistical summaries
3. **Machine Learning Pipeline** — Train, compare, and evaluate ML models (Logistic Regression, Random Forest, XGBoost)
4. **AI Chatbot** — Natural language Q&A on the dataset using OpenAI / local LLM
5. **MLflow Experiment Tracking** — Log and compare model runs
6. **Reporting Dashboard** — Export results as CSV / PDF

---

## 🏗️ Project Structure

```
DS_PROJECT/
│
├── app/                        # Main Streamlit application
│   ├── main.py                 # App entry point
│   ├── pages/
│   │   ├── 1_data_overview.py  # Data exploration page
│   │   ├── 2_eda.py            # EDA charts & statistics
│   │   ├── 3_ml_pipeline.py    # ML model training page
│   │   ├── 4_predictions.py    # Inference / prediction page
│   │   └── 5_ai_assistant.py   # AI chatbot page
│   └── utils/
│       ├── db.py               # SQLite database helpers
│       ├── preprocessing.py    # Data cleaning & feature engineering
│       ├── models.py           # ML model definitions
│       └── visualizations.py  # Chart helpers
│
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   └── clinical_trials.db     # SQLite database (auto-generated)
│
├── ml/
│   ├── train.py                # Standalone model training script
│   └── evaluate.py             # Model evaluation metrics
│
├── mlruns/                     # MLflow experiment logs (auto-generated)
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools Used |
|---|---|
| **Language** | Python 3.10+ |
| **UI Framework** | Streamlit |
| **Data Processing** | pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Database** | SQLite (via sqlite3 + SQLAlchemy) |
| **Experiment Tracking** | MLflow |
| **AI / GenAI** | OpenAI API (optional) |
| **Version Control** | Git |
| **Containerization** | Docker |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/pharmaml-analytics.git
cd pharmaml-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Synthetic Data
```bash
python data/generate_data.py
```

### 4. Run the App
```bash
streamlit run app/main.py
```

### 5. (Optional) Track ML Experiments with MLflow
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 🐳 Docker Support
```bash
docker build -t pharmaml-suite .
docker run -p 8501:8501 pharmaml-suite
```

---

## 📊 Features

### 📂 Page 1: Data Overview
- View raw clinical trial records from SQLite
- Filter by trial phase, drug category, patient age group
- Run custom SQL queries interactively
- Download filtered data as CSV

### 📈 Page 2: Exploratory Data Analysis
- Distribution plots, correlation heatmaps
- Patient demographics analysis
- Drug efficacy comparisons across trial phases
- Interactive Plotly charts

### 🤖 Page 3: ML Pipeline
- Choose from 3 ML models: Logistic Regression, Random Forest, XGBoost
- Configure hyperparameters via sliders
- Train model and view: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix and feature importance plots
- MLflow run logging (automatic)

### 🔮 Page 4: Predictions
- Input patient data and get drug trial outcome predictions
- Confidence scores with probability distributions

### 💬 Page 5: AI Assistant
- Natural language Q&A about the dataset
- Powered by OpenAI GPT / local LLM fallback
- Context-aware answers grounded in actual data

---

## 🧠 Interview Talking Points

- **"I built a full ML pipeline from data generation to deployment"**
- **"Used MLflow for experiment tracking — industry standard MLOps practice"**
- **"Streamlit app simulates a real analyst dashboard used in pharma companies"**
- **"SQL backend with SQLAlchemy for scalable data access patterns"**
- **"Added a GenAI assistant to demonstrate LLM integration skills"**
- **"Dockerized for easy deployment on GCP Cloud Run"**

---

## 📝 Author
**Banoth Rajesham** | Data Scientist | Hyderabad  
Built as part of interview preparation for Data Scientist I role.
