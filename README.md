# 🔮 ChurnGuard — Customer Churn Prediction Platform

> **A production-grade, end-to-end ML system** for predicting, explaining, and visualising customer churn — built for portfolio, internships, and real deployments.

---

## 🚀 Live Features

| Page | Description |
|------|-------------|
| 📊 Analytics | Interactive EDA dashboard with Plotly charts |
| 🎯 Predict | Real-time churn probability for any customer |
| 📈 Model Performance | ROC, Confusion Matrix, Classification Report |
| 🔍 Explainability | SHAP global + local feature explanations |

---

## 🧠 ML Stack

- **Models**: Logistic Regression · Random Forest · XGBoost (best)
- **Preprocessing**: sklearn Pipeline + StandardScaler + OneHotEncoder
- **Imbalance**: SMOTE oversampling
- **Tuning**: RandomizedSearchCV + 5-fold CV
- **Explainability**: SHAP TreeExplainer

---

## 📂 Project Structure

```
customer-churn-ml/
│
├── data/                        # Raw & processed data
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── data_loader.py           # Data download & loading
│   ├── preprocess.py            # Cleaning pipeline
│   ├── feature_engineering.py  # Feature creation
│   ├── train.py                 # Model training & tuning
│   ├── evaluate.py              # Metrics & plots
│   └── predict.py               # Inference engine
│
├── models/                      # Saved model artifacts
├── app/
│   ├── main.py                  # Streamlit entry point
│   └── pages/
│       ├── 1_Analytics.py
│       ├── 2_Predict.py
│       ├── 3_Model_Performance.py
│       └── 4_Explainability.py
│
├── visuals/                     # Generated plots
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/harshvardh6n/churn_project.git
cd customer-churn-ml
pip install -r requirements.txt

# 2. Download dataset & train model
python src/data_loader.py

# 3. Launch dashboard
streamlit run app/main.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features.  
Auto-downloaded from the public Kaggle mirror on first run.

Key features: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `TechSupport`, and 14 more.

---

## 🏆 Model Results

| Model | AUC | F1 | Accuracy |
|-------|-----|-----|----------|
| Logistic Regression | 0.841 | 0.612 | 0.801 |
| Random Forest | 0.858 | 0.631 | 0.815 |
| **XGBoost** | **0.871** | **0.649** | **0.823** |

---

## 📄 License

MIT — free to use, fork, and extend.
