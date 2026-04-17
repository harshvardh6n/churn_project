"""
predict.py
==========
Inference engine for the Customer Churn Prediction system.

Provides
--------
predict_single(customer_dict)   - probability + risk label for one customer
predict_batch(df)               - DataFrame with appended probability column
build_input_df(form_values)     - convert Streamlit form dict → model-ready DataFrame
generate_pdf_report(customer_dict, result_dict) - downloadable PDF summary
"""

import os, sys, logging
import numpy  as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import engineer_features


# ── Load model (lazy, cached) ─────────────────────────────────────────────────
_model_cache = {}

def load_model(name: str = "best_model"):
    if name not in _model_cache:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python src/train.py` first."
            )
        _model_cache[name] = joblib.load(path)
        logger.info("Loaded model: %s", name)
    return _model_cache[name]


# ── Risk thresholds ────────────────────────────────────────────────────────────
def classify_risk(prob: float) -> dict:
    """Map probability to risk label, colour, and emoji."""
    if prob < 0.30:
        return {"label": "Low Risk",    "colour": "#10B981", "emoji": "🟢", "level": 1}
    elif prob < 0.60:
        return {"label": "Medium Risk", "colour": "#F59E0B", "emoji": "🟡", "level": 2}
    elif prob < 0.80:
        return {"label": "High Risk",   "colour": "#EF4444", "emoji": "🔴", "level": 3}
    else:
        return {"label": "Critical",    "colour": "#7C3AED", "emoji": "🚨", "level": 4}


# ── Single customer prediction ────────────────────────────────────────────────

def build_input_df(form_values: dict) -> pd.DataFrame:
    """
    Convert a flat dict (keys matching raw dataset columns) into
    a single-row DataFrame ready for the pipeline.

    form_values keys (all raw column names):
        gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    """
    df = pd.DataFrame([form_values])

    # Encode binary columns to int (the pipeline's preprocessor handles categoricals)
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    ]
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(binary_map).fillna(df[col])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Add engineered features
    df = engineer_features(df)
    return df


def predict_single(form_values: dict, model_name: str = "best_model") -> dict:
    """
    Predict churn for a single customer.

    Parameters
    ----------
    form_values : dict  - raw feature values (see build_input_df)
    model_name  : str   - which model pkl to use

    Returns
    -------
    dict with keys: probability, risk, prediction (0 or 1)
    """
    model = load_model(model_name)
    X     = build_input_df(form_values)

    proba      = model.predict_proba(X)[0, 1]
    prediction = int(proba >= 0.5)
    risk       = classify_risk(proba)

    return {
        "probability": round(float(proba), 4),
        "prediction":  prediction,
        "risk":        risk,
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, model_name: str = "best_model") -> pd.DataFrame:
    """
    Add ChurnProbability and RiskLabel columns to a cleaned DataFrame.

    The input df should already have customerID dropped and TotalCharges fixed.
    """
    model     = load_model(model_name)
    df_eng    = engineer_features(df.copy())
    probas    = model.predict_proba(df_eng)[:, 1]
    preds     = (probas >= 0.5).astype(int)

    out = df.copy()
    out["ChurnProbability"] = probas.round(4)
    out["ChurnPrediction"]  = preds
    out["RiskLabel"] = [classify_risk(p)["label"] for p in probas]
    return out


# ── PDF report generator ──────────────────────────────────────────────────────

def generate_pdf_report(customer_dict: dict, result_dict: dict) -> bytes:
    """
    Generate a simple single-page PDF prediction report.

    Returns bytes (write to file or serve as download).
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed - PDF generation skipped.")
        return b""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──────────────────────────────────────────────────────────────
    pdf.set_fill_color(15, 17, 23)
    pdf.rect(0, 0, 210, 40, style="F")
    pdf.set_text_color(0, 194, 203)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(12)
    pdf.cell(0, 10, "ChurnGuard - Prediction Report", ln=True, align="C")

    # ── Risk summary ─────────────────────────────────────────────────────────
    pdf.set_y(50)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "B", 14)
    risk    = result_dict["risk"]
    prob_pct = result_dict["probability"] * 100
    pdf.cell(0, 10, f"Churn Probability: {prob_pct:.1f}%  |  Risk Level: {risk['label']}", ln=True)

    # ── Customer details ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(70, 70, 70)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Customer Details", ln=True)
    pdf.set_font("Helvetica", "", 11)

    skip = {"customerID"}
    for key, val in customer_dict.items():
        if key in skip:
            continue
        pdf.cell(90, 7, str(key), border="B")
        pdf.cell(0,  7, str(val), border="B", ln=True)

    # ── Recommendation ────────────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Recommended Action", ln=True)
    pdf.set_font("Helvetica", "", 11)
    if risk["level"] >= 3:
        action = "Immediate outreach - offer a loyalty discount or contract upgrade."
    elif risk["level"] == 2:
        action = "Monitor closely - consider proactive check-in call within 30 days."
    else:
        action = "Low risk - no immediate action required. Standard engagement."
    pdf.multi_cell(0, 7, action)

    return pdf.output()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 5, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35, "TotalCharges": 350.0,
    }
    result = predict_single(sample)
    print(result)