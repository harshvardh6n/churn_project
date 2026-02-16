# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview
Customer churn refers to customers leaving or discontinuing a service.  
This project builds an **end-to-end Machine Learning system** that predicts whether a customer is likely to churn based on their demographic details, service usage, contract type, and billing information.

The trained ML model is deployed using a **Streamlit web application** that allows real-time churn prediction through an interactive user interface.

---

## 🎯 Objective
- Predict the probability of customer churn  
- Identify high-risk customers early  
- Help businesses take preventive actions to improve customer retention  

---

## 🧠 Problem Type
- **Machine Learning Task:** Binary Classification  
- **Target Variable:** `Churn`
  - `1` → Customer will churn  
  - `0` → Customer will not churn  

---

## 📂 Dataset
- **Name:** Telco Customer Churn Dataset  
- **Source:** Kaggle  
- **Rows:** ~7,000 customers  
- **Features:** Demographics, services, contracts, payments, billing  
- **Target Column:** `Churn`

> ⚠️ Dataset is not included due to licensing restrictions.  
> Download from: https://www.kaggle.com/blastchar/telco-customer-churn

---

## 🏗️ Project Structure
```
Customer-Churn-Prediction/
│── data/
│ └── Telco-Customer-Churn.csv
│── notebooks/
│ └── churn_prediction.ipynb
│── src/
│ ├── preprocess.py
│ ├── model.py
│ └── utils.py
│── app/
│ └── streamlit_app.py
│── models/
│ ├── churn_model.pkl
│ ├── scaler.pkl
│ └── feature_names.pkl
│── visuals/
│ └── (EDA & visualization outputs)
│── README.md
│── requirements.txt

```

---

## 🔄 Machine Learning Pipeline

### 1️⃣ Data Loading
- Load raw CSV data using Pandas

### 2️⃣ Data Cleaning
- Convert `TotalCharges` to numeric
- Handle missing values
- Remove irrelevant features (`customerID`)

### 3️⃣ Feature Encoding
- Convert binary categories (Yes/No → 1/0)
- Apply One-Hot Encoding to categorical features

### 4️⃣ Feature Scaling
- Standardize numerical features using `StandardScaler`

### 5️⃣ Handling Class Imbalance
- Apply **SMOTE** to balance churn vs non-churn samples

### 6️⃣ Model Training
Multiple models were trained and compared:
- Logistic Regression
- Random Forest
- XGBoost (Best performing)

### 7️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- ROC-AUC Score
- Confusion Matrix

### 8️⃣ Model Persistence
Saved trained artifacts:
- `churn_model.pkl`
- `scaler.pkl`
- `feature_names.pkl`

---

## 🚀 Deployment (Streamlit App)

The Streamlit app:
- Takes customer input via UI
- Reconstructs feature vector exactly as in training
- Scales input data
- Predicts churn probability
- Displays risk level and visual indicators

### Risk Categories:
- **Low Risk:** < 0.4  
- **Medium Risk:** 0.4 – 0.6  
- **High Risk:** > 0.6  

---
this is still in development phase and I am working to improve the accuracy of the models
