# 🏥 FedTriage — Federated Learning Medical Triage System
### 📦 Recommended Repository Name: `fedtriage-fl`

> A privacy-preserving AI triage system that trains across 3 clinics using Federated Learning, Reinforcement Learning, and Ensemble Meta-Learning — without sharing raw patient data.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [AI Models Used](#ai-models-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [GitHub Deployment](#github-deployment)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## 🔍 Overview

**FedTriage** is a Federated Learning-based medical triage tool that simulates 3 hospitals (clinics) collaboratively training an AI model — **without any clinic sharing its raw patient data**.

Each clinic trains a local Gradient Boosting model on its own patient records. Only model weights are shared with a central aggregator (FedProx), which combines them and redistributes a global model. A stacked meta-learner then consolidates outputs from all 3 clinics into a final triage decision.

### 🎯 Triage Categories
| Label | Urgency | Action |
|-------|---------|--------|
| 🟢 Green | Non-urgent | Monitor at home, follow up in 48–72 hrs |
| 🟡 Yellow | Semi-urgent | Appointment within 24 hours |
| 🔴 Red | Immediate | Emergency care required — do not delay |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FEDERATED LEARNING SYSTEM                   │
├──────────────────┬──────────────────┬──────────────────┐        │
│    CLINIC 1      │    CLINIC 2      │    CLINIC 3      │        │
│  (Age 50+)       │  (Age 18–49)     │  (Age < 18)      │        │
│  Urban/Elderly   │  Adult/General   │  Pediatric       │        │
│                  │                  │                  │        │
│  Gradient        │  Gradient        │  Gradient        │        │
│  Boosting        │  Boosting        │  Boosting        │        │
│  + RL Agent      │  + RL Agent      │  + RL Agent      │        │
│                  │                  │                  │        │
│  [Patient data stays local — never shared]               │        │
└────────┬─────────┴────────┬─────────┴────────┬─────────┘        │
         │  weights only    │  weights only    │  weights only    │
         ▼                  ▼                  ▼                  │
┌─────────────────────────────────────────────────────────────────┤
│               FedProx AGGREGATOR (Central Server)               │
│          Weighted FedAvg + Proximal Term Correction             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ global weights redistributed
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             STACKED ENSEMBLE META-LEARNER                       │
│   Input: 9 probabilities (3 clinics × 3 classes)                │
│   Model: Logistic Regression                                    │
│   Output: Final Triage Decision  🟢 Green │ 🟡 Yellow │ 🔴 Red  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 AI Models Used

### 1. Local Clinic Models — Gradient Boosting Classifier
- **Library**: `scikit-learn`
- One independent model per clinic (3 total)
- 100 estimators, max depth 4, learning rate 0.1
- Trained with **cost-sensitive sample weights** (RL-inspired):
  - 🔴 Red patients → weight `3.0` (critical — must not be missed)
  - 🟡 Yellow patients → weight `1.5`
  - 🟢 Green patients → weight `1.0`

### 2. Reinforcement Learning — Policy Gradient Agent
- **Library**: Custom NumPy implementation
- Learns optimal **per-class escalation thresholds**
- Reward function = negative triage cost matrix:

```
Cost Matrix (true → predicted):
          Green  Yellow   Red
Green  [  0.0,   0.5,   2.0 ]
Yellow [  1.5,   0.0,   0.5 ]
Red    [  5.0,   2.0,   0.0 ]  ← missing a Red patient = highest penalty
```

- Uses REINFORCE-style gradient to update thresholds over 10 iterations

### 3. Federated Aggregation — FedProx
- **Library**: Custom NumPy implementation
- Weighted FedAvg: averages clinic weights proportional to sample count
- Proximal correction term (μ=0.01) handles non-IID data across clinics
- 3 federated rounds

### 4. Meta-Learner — Stacked Ensemble
- **Library**: `scikit-learn` LogisticRegression
- Input: concatenated softmax outputs from all 3 clinics (9 features)
- Learns to optimally weight each clinic's opinion
- Produces the **final triage prediction**

---

## 📊 Dataset

**Source**: [KTAS Emergency Triage Dataset — Kaggle](https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application)

| Feature | Description |
|---------|-------------|
| Age | Patient age |
| Sex | Gender (M/F) |
| Chief Complaint | Presenting symptom (text → binary flags) |
| SBP / DBP | Blood pressure |
| HR | Heart rate |
| BT | Body temperature |
| Saturation | SpO₂ oxygen saturation |
| KTAS_expert | Triage label (1–5, remapped to Red/Yellow/Green) |

**Preprocessing**:
- KTAS 1–2 → 🔴 Red | KTAS 3 → 🟡 Yellow | KTAS 4–5 → 🟢 Green
- Travel history added synthetically (not in original dataset)
- Split into 3 clinics by age group to simulate non-IID federated data:

| Clinic | Age Group | Patients |
|--------|-----------|----------|
| Clinic 1 | 50+ (Elderly) | 422 |
| Clinic 2 | 18–49 (Adult) | 422 |
| Clinic 3 | Under 18 (Pediatric) | 423 |

---

## 📁 Project Structure

```
medtool/
├── data/
│   └── raw/
│       ├── clinic_1.csv          # Elderly patients (age 50+)
│       ├── clinic_2.csv          # Adult patients (age 18–49)
│       └── clinic_3.csv          # Pediatric patients (age <18)
├── models/
│   ├── __init__.py
│   ├── clinic_model.py           # GBM + RL Agent
│   └── saved/                    # Trained .pkl model files
├── federated/
│   ├── __init__.py
│   ├── aggregator.py             # FedProx + Meta-Learner
│   ├── round_history.json        # Per-round accuracy logs
│   └── training_summary.json     # Final results
├── utils/
│   ├── __init__.py
│   └── preprocessing.py          # Feature encoding & scaling
├── app/
│   ├── __init__.py
│   └── streamlit_app.py          # Dashboard UI
├── datasetslipting.py            # Dataset adapter & clinic splitter
├── train.py                      # Main training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/fedtriage.git
cd fedtriage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare the dataset
#    Download from: https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application
#    Place the CSV in the medtool folder, then run:
python datasetslipting.py

# 4. Train the federated model
python train.py

# 5. Launch the dashboard
streamlit run app/streamlit_app.py
```

---

## 🚀 Usage

### Training Output
```
STEP 1: Loading clinic data...
  clinic_1: 337 train, 85 test
  clinic_2: 337 train, 85 test
  clinic_3: 338 train, 85 test

STEP 2: Baseline training...
  clinic_1 baseline: 0.4118
  clinic_2 baseline: 0.4235
  clinic_3 baseline: 0.5059

STEP 3: Federated Learning (fedprox)...
  Round 1/3 → clinic_1: 0.4118 | clinic_2: 0.4235 | clinic_3: 0.5059
  Round 2/3 → ...
  Round 3/3 → ...

STEP 4: Post-federated evaluation...
  clinic_1 federated: 0.5529  ← +14.1% improvement
  clinic_2 federated: 0.4824  ← +5.9% improvement
  clinic_3 federated: 0.5059

STEP 6: Meta-learner accuracy: 0.5294
```

### Dashboard Pages
| Page | Description |
|------|-------------|
| 🩺 Patient Triage | Enter patient details → get instant triage decision from all 3 clinic models + meta-learner |
| 📊 Clinic Dashboard | Compare baseline vs federated accuracy, view dataset distributions |
| 🔬 Federated Learning | Visualize accuracy improvement over federated rounds |
| ℹ️ About | System architecture and tech stack |

---

## 📈 Results

| Metric | Clinic 1 | Clinic 2 | Clinic 3 | Meta-Learner |
|--------|----------|----------|----------|--------------|
| Baseline Accuracy | 41.2% | 42.4% | 50.6% | — |
| Federated Accuracy | **55.3%** | **48.2%** | **50.6%** | **52.9%** |
| Improvement | **+14.1%** | **+5.9%** | 0.0% | — |

> Note: Dataset size (1,267 patients) limits maximum accuracy. Larger datasets (e.g., MIMIC-IV) would yield significantly higher performance.

---

## 📤 GitHub Deployment

### First Time Setup

```bash
# Navigate to project folder
cd C:\Users\nivet\Downloads\hca\medtool

# Initialize git
git init
git add .
git commit -m "Initial commit: FedTriage Federated Learning Triage System"

# Create repo on GitHub at github.com/new, then:
git remote add origin https://github.com/YOUR_USERNAME/fedtriage.git
git branch -M main
git push -u origin main
```

### .gitignore (already included)
```
models/saved/
federated/round_history.json
federated/training_summary.json
data/raw/
__pycache__/
*.pkl
*.pyc
.env
```

### Subsequent Updates
```bash
git add .
git commit -m "Update: describe your changes here"
git push
```

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12 | Core language |
| scikit-learn | 1.3+ | ML models (GBM, Logistic Regression) |
| NumPy | 1.24+ | RL agent, federated aggregation |
| Pandas | 2.0+ | Data preprocessing |
| Streamlit | 1.28+ | Interactive dashboard |
| Plotly | 5.17+ | Charts and visualizations |

---

## 👩‍💻 Author

**Nivetha Sre**
Undergraduate Project — Federated Learning in Healthcare
Department of Computer Science

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [KTAS Emergency Triage Dataset](https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application) — Kaggle
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) — McMahan et al. (FedAvg)
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) — Li et al. (FedProx)
