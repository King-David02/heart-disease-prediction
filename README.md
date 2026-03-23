<div align="center">

# Heart Disease Prediction
### End-to-End Production ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![ZenML](https://img.shields.io/badge/ZenML-0.92-431D93?style=flat-square)](https://zenml.io)
[![MLflow](https://img.shields.io/badge/MLflow-3.10-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Predicting 10-year risk of coronary heart disease using the Framingham Heart Study dataset*

---

[Overview](#overview) · [Architecture](#architecture) · [Quick Start](#quick-start) · [Pipeline](#pipeline) · [Results](#results) · [Stack](#tech-stack)

</div>

---

## Overview

World Health Organization estimates 12 million deaths occur worldwide every year due to heart disease. Early prognosis of cardiovascular diseases can aid in making decisions on lifestyle changes in high-risk patients and reduce complications.

This project builds a **complete MLOps pipeline** — from raw data ingestion to a deployed REST API with drift monitoring — predicting whether a patient has a 10-year risk of coronary heart disease (CHD).
```
Raw Data → Validation → Preprocessing → Feature Engineering
    → Training → Evaluation → Registry → API → Monitoring
```

---

## Architecture
```
heart-disease-prediction/
│
├── src/
│   ├── config/          ← Settings, logging
│   ├── data/            ← Ingestion, validation, preprocessing, features
│   ├── models/          ← Training, evaluation, inference, registry
│   ├── steps/           ← ZenML @steps
│   ├── pipelines/       ← ZenML @pipelines
│   ├── monitoring/      ← Drift detection, metrics, alerts
│   └── api/             ← FastAPI serving layer
│
├── configs/             ← YAML configuration files
├── data/                ← raw / processed / features
├── models/              ← experiments / staging / production
├── scripts/             ← Setup and run scripts
├── tests/               ← Unit, integration, load tests
└── infrastructure/      ← Terraform, Kubernetes, Helm
```

---

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone & Install
```bash
git clone https://github.com/King-David02/heart-disease-prediction.git
cd heart-disease-prediction
uv pip install -e .
```

### 2. Setup
```bash
# Setup environment and directories
bash scripts/setup_environment.sh

# Register ZenML stack
uv run scripts/setup_zenml_stack.py
```

### 3. Add Dataset

Download `framingham.csv` from [Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression) and place it in:
```bash
data/raw/framingham.csv
```

### 4. Run Pipeline
```bash
uv run scripts/run_pipeline.py
```

### 5. View Dashboards
```bash
# ZenML — pipeline runs, artifacts, lineage
uv run zenml login --local

# MLflow — experiments, metrics, models
uv run mlflow ui
```

---

## Dataset

| Property | Value |
|---|---|
| Source | Framingham Heart Study |
| Rows | 4,238 |
| Features | 15 |
| Target | `TenYearCHD` (binary) |
| Task | Binary Classification |

### Features

| Category | Features |
|---|---|
| Demographic | `age`, `male` |
| Behavioral | `currentSmoker`, `cigsPerDay` |
| Medical History | `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes` |
| Medical Current | `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose` |
| Engineered | `pulse_pressure` (sysBP - diaBP) |

---

## Pipeline
```
load_data_step
      ↓
validate_data_step      ← schema check, null detection, duplicate check
      ↓
preprocess_data_step    ← median/mode imputation, drop duplicates
      ↓
engineer_features_step  ← StandardScaler, pulse_pressure feature
      ↓
train_model_step        ← Logistic Regression + mlflow.autolog()
      ↓
evaluate_step           ← accuracy, ROC-AUC, F1, confusion matrix
      ↓
register_model_step     ← ZenML model registry → staging → production
```

Each step is:
- **Tracked** — logged in ZenML dashboard
- **Cached** — skipped if inputs haven't changed
- **Versioned** — artifacts stored with full lineage

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~0.85 |
| ROC-AUC | ~0.70 |
| Precision | ~0.60 |
| Recall | ~0.30 |

> Model is more specific than sensitive — optimized to minimize false positives. For clinical use, threshold should be lowered to reduce false negatives.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Package Manager | `uv` | Fast dependency management |
| Orchestration | `ZenML` | Pipeline tracking, caching, registry |
| Experiment Tracking | `MLflow` | Metrics, params, artifacts |
| ML | `scikit-learn` | Logistic Regression |
| API | `FastAPI` | REST endpoint |
| Monitoring | `Evidently AI` | Data & model drift |
| Logging | `Loguru` | Structured logging |
| CI/CD | `GitHub Actions` | Automated testing & deployment |
| Containers | `Docker` | Reproducible environments |
| Cloud | `AWS` | Production deployment |

---

## Author

**King-David Ajana**

[![GitHub](https://img.shields.io/badge/GitHub-King--David02-181717?style=flat-square&logo=github)](https://github.com/King-David02)

---

<div align="center">
<sub>Built with precision. Deployed with confidence.</sub>
</div>
