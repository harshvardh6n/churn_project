"""
evaluate.py
===========
Comprehensive model evaluation module.

Functions
---------
get_all_metrics        - dict of accuracy / precision / recall / F1 / AUC
plot_confusion_matrix  - Plotly heatmap
plot_roc_curve         - Plotly ROC with AUC annotation
plot_model_comparison  - Plotly grouped bar chart (all 3 models)
plot_feature_importance - Plotly horizontal bar (top-N features)
generate_classification_report - formatted string
"""

import os, sys, json, logging
import numpy  as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

logger = logging.getLogger(__name__)

ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# ── Colour palette (matches the Streamlit theme) ──────────────────────────────
TEAL   = "#00C2CB"
CORAL  = "#FF6B6B"
PURPLE = "#8B5CF6"
AMBER  = "#F59E0B"
BG     = "#0F1117"
CARD   = "#1A1D27"

# ── Metric computation ────────────────────────────────────────────────────────

def get_all_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc":       round(roc_auc_score(y_true, y_proba), 4),
    }


def generate_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["Stay", "Churn"])


# ── Plotly charts ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred) -> go.Figure:
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Stay", "Churn"]
    z_text = [[f"{v:,}" for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z          = cm,
        x          = labels,
        y          = labels,
        text       = z_text,
        texttemplate = "<b>%{text}</b>",
        colorscale = [[0, CARD], [0.5, PURPLE], [1, TEAL]],
        showscale  = False,
    ))
    fig.update_layout(
        title      = "Confusion Matrix",
        xaxis_title = "Predicted",
        yaxis_title = "Actual",
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font  = dict(color="white", family="Syne, sans-serif"),
        title_font = dict(size=16),
        margin = dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_roc_curve(y_true, y_proba, model_name="XGBoost") -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score   = roc_auc_score(y_true, y_proba)

    fig = go.Figure()
    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", line=dict(dash="dash", color="#4B5563", width=1.5),
        name="Random (AUC = 0.50)",
    ))
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        line=dict(color=TEAL, width=3),
        name=f"{model_name} (AUC = {auc_score:.4f})",
        fill="tozeroy",
        fillcolor="rgba(0,194,203,0.12)",
    ))
    fig.update_layout(
        title       = "ROC Curve",
        xaxis_title = "False Positive Rate",
        yaxis_title = "True Positive Rate",
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font   = dict(color="white", family="Syne, sans-serif"),
        title_font = dict(size=16),
        legend = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin = dict(l=20, r=20, t=50, b=20),
        xaxis  = dict(gridcolor="#1F2937", range=[0, 1]),
        yaxis  = dict(gridcolor="#1F2937", range=[0, 1]),
    )
    return fig


def plot_model_comparison(results: list) -> go.Figure:
    """Grouped bar chart comparing all models on AUC / F1 / Accuracy."""
    models   = [r["model"] for r in results]
    metrics  = ["auc", "f1", "accuracy"]
    labels   = ["AUC", "F1 Score", "Accuracy"]
    colours  = [TEAL, PURPLE, AMBER]

    fig = go.Figure()
    for metric, label, colour in zip(metrics, labels, colours):
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=[r[metric] for r in results],
            marker_color=colour,
            text=[f"{r[metric]:.4f}" for r in results],
            textposition="outside",
            textfont=dict(color="white"),
        ))

    fig.update_layout(
        barmode     = "group",
        title       = "Model Comparison",
        yaxis       = dict(range=[0, 1.05], gridcolor="#1F2937"),
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font   = dict(color="white", family="Syne, sans-serif"),
        title_font = dict(size=16),
        legend = dict(bgcolor="rgba(0,0,0,0)"),
        margin = dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_feature_importance(model_pipeline, feature_names: list, top_n=20) -> go.Figure:
    """Extract and plot feature importances from XGBoost inside the pipeline."""
    try:
        clf = model_pipeline.named_steps.get("clf")
        if clf is None:                          # ImbPipeline uses different key
            clf = model_pipeline[-1]
        importances = clf.feature_importances_
    except Exception as exc:
        logger.warning("Could not extract feature importances: %s", exc)
        return go.Figure()

    # Align lengths
    n = min(len(importances), len(feature_names))
    importance_df = (
        pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
        .nlargest(top_n, "importance")
        .sort_values("importance")
    )

    fig = go.Figure(go.Bar(
        x            = importance_df["importance"],
        y            = importance_df["feature"],
        orientation  = "h",
        marker_color = TEAL,
        text         = importance_df["importance"].round(4),
        textposition = "outside",
    ))
    fig.update_layout(
        title       = f"Top {top_n} Feature Importances",
        xaxis_title = "Importance",
        paper_bgcolor = BG,
        plot_bgcolor  = BG,
        font   = dict(color="white", family="Syne, sans-serif"),
        title_font = dict(size=16),
        margin = dict(l=20, r=20, t=50, b=20),
        xaxis  = dict(gridcolor="#1F2937"),
        yaxis  = dict(gridcolor="rgba(0,0,0,0)"),
        height = 500,
    )
    return fig


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_best_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    return joblib.load(path)


def load_metadata() -> dict:
    path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(path) as fh:
        return json.load(fh)


def load_test_set():
    path = os.path.join(ROOT_DIR, "data", "test_set.csv")
    df   = pd.read_csv(path)
    y    = df.pop("Churn")
    return df, y


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = load_best_model()
    meta  = load_metadata()
    X_test, y_test = load_test_set()

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = get_all_metrics(y_test, y_pred, y_proba)
    print("Metrics:", metrics)
    print(generate_classification_report(y_test, y_pred))