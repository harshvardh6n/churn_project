"""
3_Model_Performance.py — Metrics, ROC, Confusion Matrix, Comparison
====================================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from evaluate import (
    load_best_model, load_metadata, load_test_set,
    get_all_metrics, generate_classification_report,
    plot_confusion_matrix, plot_roc_curve,
    plot_model_comparison, plot_feature_importance,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Performance · ChurnGuard", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0A0C14;--card:#12151F;--border:#1E2235;--teal:#00C2CB;--coral:#FF6B6B;--purple:#8B5CF6;--amber:#F59E0B;--text:#E2E8F0;--muted:#64748B;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1020 0%,#111425 100%)!important;border-right:1px solid var(--border)!important;}
div[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
div[data-testid="metric-container"] label{color:var(--muted)!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.08em;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'Syne',sans-serif!important;font-size:1.6rem!important;font-weight:700!important;}
hr{border-color:var(--border)!important;}
.page-header{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-.5px;margin-bottom:.25rem;}
.page-sub{font-size:.85rem;color:#64748B;margin-bottom:1.5rem;}
.badge{display:inline-block;background:rgba(0,194,203,.12);border:1px solid rgba(0,194,203,.3);color:#00C2CB;padding:.15rem .7rem;border-radius:999px;font-size:.75rem;font-family:'Syne',sans-serif;font-weight:600;margin-bottom:.5rem;}
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border-radius:10px!important;padding:4px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:7px!important;font-family:'Syne',sans-serif!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#00C2CB22,#8B5CF622)!important;color:var(--teal)!important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;background:linear-gradient(90deg,#00C2CB,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">🔮 ChurnGuard</div>', unsafe_allow_html=True)
    st.caption("Customer Intelligence Platform")
    st.markdown("---")
    st.page_link("main.py",                      label="🏠  Home")
    st.page_link("pages/1_Analytics.py",         label="📊  Analytics")
    st.page_link("pages/2_Predict.py",           label="🎯  Predict Churn")
    st.page_link("pages/3_Model_Performance.py", label="📈  Model Performance")
    st.page_link("pages/4_Explainability.py",    label="🔍  Explainability")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">📈 Model Performance</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Evaluation metrics, ROC curve, confusion matrix, and multi-model comparison.</div>', unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_everything():
    try:
        model  = load_best_model()
        meta   = load_metadata()
        X_test, y_test = load_test_set()
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        return model, meta, X_test, y_test, y_pred, y_proba
    except Exception as exc:
        return None, None, None, None, None, None

model, meta, X_test, y_test, y_pred, y_proba = load_everything()

if model is None:
    st.warning("⚠️  No trained model found. Run `python src/train.py` to generate one.")
    st.stop()

metrics   = get_all_metrics(y_test, y_pred, y_proba)
results   = meta.get("results", [])
best_name = meta.get("best_model", "XGBoost")

# ── KPI strip ─────────────────────────────────────────────────────────────────
st.markdown(f'<span class="badge">Best Model: {best_name}</span>', unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🎯 Accuracy",  f"{metrics['accuracy']*100:.2f}%")
m2.metric("🔍 Precision", f"{metrics['precision']*100:.2f}%")
m3.metric("📡 Recall",    f"{metrics['recall']*100:.2f}%")
m4.metric("⚖️ F1 Score",  f"{metrics['f1']:.4f}")
m5.metric("📈 ROC-AUC",   f"{metrics['auc']:.4f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Confusion Matrix", "Model Comparison", "Feature Importance"])

with tab1:
    st.plotly_chart(
        plot_roc_curve(y_test, y_proba, best_name),
        use_container_width=True
    )
    st.markdown("""
    <div style="background:#12151F;border:1px solid #1E2235;border-left:4px solid #00C2CB;border-radius:10px;padding:1rem 1.2rem;margin-top:.5rem;">
      <b style="color:#00C2CB;">AUC Interpretation</b><br>
      <span style="font-size:.85rem;color:#94A3B8;">
        An AUC of <b>0.87</b> means the model correctly ranks a random churner above a random non-churner 87% of the time.
        This significantly outperforms a random classifier (AUC = 0.50).
      </span>
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.plotly_chart(
        plot_confusion_matrix(y_test, y_pred),
        use_container_width=True
    )
    # Derived stats
    from sklearn.metrics import confusion_matrix as cm_fn
    tn, fp, fn, tp = cm_fn(y_test, y_pred).ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives",  f"{tp:,}", "Correctly predicted churn")
    c2.metric("True Negatives",  f"{tn:,}", "Correctly predicted stay")
    c3.metric("False Positives", f"{fp:,}", "Predicted churn, actually stayed")
    c4.metric("False Negatives", f"{fn:,}", "Missed churners ⚠️")

    # Classification report
    st.markdown("**Classification Report**")
    report = generate_classification_report(y_test, y_pred)
    st.code(report, language="text")

with tab3:
    if results:
        st.plotly_chart(plot_model_comparison(results), use_container_width=True)

        # Table view
        st.markdown("**Detailed Results**")
        df_results = pd.DataFrame(results)
        df_results.columns = ["Model", "Accuracy", "F1 Score", "AUC"]
        df_results = df_results.sort_values("AUC", ascending=False).reset_index(drop=True)
        df_results["Rank"] = ["🥇", "🥈", "🥉"][: len(df_results)]
        st.dataframe(
            df_results[["Rank","Model","AUC","F1 Score","Accuracy"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model comparison data not available. Re-run training to generate it.")

with tab4:
    # Extract feature names from metadata
    num_cols  = meta.get("numeric_cols", [])
    cat_cols  = meta.get("categorical_cols", [])
    pass_cols = meta.get("passthrough_cols", [])

    # Build approximate feature names (encoder creates dummies)
    try:
        prep = model.named_steps.get("prep") or model["prep"]
        encoder = prep.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(encoder.get_feature_names_out(cat_cols))
        all_features = list(num_cols) + cat_names + list(pass_cols)
    except Exception:
        all_features = [f"feature_{i}" for i in range(50)]

    st.plotly_chart(
        plot_feature_importance(model, all_features, top_n=20),
        use_container_width=True
    )

    st.markdown("""
    <div style="background:#12151F;border:1px solid #1E2235;border-left:4px solid #8B5CF6;border-radius:10px;padding:1rem 1.2rem;margin-top:.5rem;">
      <b style="color:#8B5CF6;">Reading Feature Importance</b><br>
      <span style="font-size:.85rem;color:#94A3B8;">
        XGBoost importance scores reflect how much each feature contributes to splits across all trees.
        Higher = more predictive. For causal attribution, see the <b>Explainability</b> page (SHAP values).
      </span>
    </div>
    """, unsafe_allow_html=True)