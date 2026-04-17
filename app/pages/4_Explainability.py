"""
4_Explainability.py — SHAP Global + Local Explanations
=======================================================
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import pandas as pd
import numpy  as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from evaluate import load_best_model, load_metadata, load_test_set

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Explainability · ChurnGuard", page_icon="🔍", layout="wide")

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
    st.markdown("---")
    n_background = st.slider("Background samples (SHAP)", 50, 500, 100, 50)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">🔍 Model Explainability</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">SHAP (SHapley Additive exPlanations) — global feature importance and per-customer prediction explanations.</div>', unsafe_allow_html=True)

# ── Load & compute SHAP ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Computing SHAP values (this may take ~30s)…")
def compute_shap(n_bg: int):
    import joblib, os

    meta = load_metadata()

    # Always prefer XGBoost for SHAP TreeExplainer.
    # Fall back to Random Forest, then best_model — but reject linear models.
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    tree_candidates = ["xgboost", "random_forest", "best_model"]
    model = None
    for name in tree_candidates:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            candidate = joblib.load(path)
            clf_step = candidate.named_steps.get("clf")
            # Skip linear / non-tree models
            if clf_step is not None and hasattr(clf_step, "feature_importances_"):
                model = candidate
                break

    if model is None:
        st.error("No tree-based model found. Re-run `python src/train.py`.")
        return None, None, None, None, None

    X_test, y_test = load_test_set()

    try:
        clf  = model.named_steps["clf"]
        prep = model.named_steps["prep"]
    except KeyError:
        return None, None, None, None, None

    X_transformed = prep.transform(X_test)

    num_cols  = meta.get("numeric_cols", [])
    cat_cols  = meta.get("categorical_cols", [])
    pass_cols = meta.get("passthrough_cols", [])
    try:
        enc       = prep.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(enc.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = [f"cat_{i}" for i in range(
            X_transformed.shape[1] - len(num_cols) - len(pass_cols)
        )]
    feature_names = list(num_cols) + cat_names + list(pass_cols)

    background  = shap.sample(X_transformed, min(n_bg, X_transformed.shape[0]))
    explainer   = shap.TreeExplainer(clf, background)
    shap_values = explainer.shap_values(X_transformed[:300])

    return shap_values, X_transformed[:300], feature_names, X_test.head(300), y_test.values[:300]

    # Extract XGBoost from pipeline
    try:
        clf  = model.named_steps["clf"]
        prep = model.named_steps["prep"]
    except KeyError:
        return None, None, None, None, None
    # Transform test set
    X_transformed = prep.transform(X_test)

    # Feature names
    num_cols  = meta.get("numeric_cols", [])
    cat_cols  = meta.get("categorical_cols", [])
    pass_cols = meta.get("passthrough_cols", [])
    try:
        enc       = prep.named_transformers_["cat"].named_steps["encoder"]
        cat_names = list(enc.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = [f"cat_{i}" for i in range(X_transformed.shape[1] - len(num_cols) - len(pass_cols))]
    feature_names = list(num_cols) + cat_names + list(pass_cols)

    # SHAP TreeExplainer
    background = shap.sample(X_transformed, min(n_bg, X_transformed.shape[0]))
    explainer  = shap.TreeExplainer(clf, background)
    shap_values = explainer.shap_values(X_transformed[:300])   # first 300 for speed

    return shap_values, X_transformed[:300], feature_names, X_test.head(300), y_test.values[:300]

with st.spinner("Computing SHAP values…"):
    try:
        shap_values, X_trans, feature_names, X_raw, y_true = compute_shap(n_background)
    except Exception as exc:
        shap_values = None
        st.error(f"Could not compute SHAP: {exc}")

if shap_values is None:
    st.warning("⚠️  Train the model first (`python src/train.py`), then reload.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Global Importance", "SHAP Summary Plot", "Individual Explanation"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🌍 Global Feature Importance (Mean |SHAP|)")

    mean_shap = np.abs(shap_values).mean(axis=0)
    n = min(len(mean_shap), len(feature_names))
    importance_df = (
        pd.DataFrame({"Feature": feature_names[:n], "Mean |SHAP|": mean_shap[:n]})
        .nlargest(20, "Mean |SHAP|")
        .sort_values("Mean |SHAP|")
    )

    # Colour gradient by magnitude
    max_val = importance_df["Mean |SHAP|"].max()
    colours = [
        f"rgba({int(139 + (0-139)*v/max_val)},{int(92 + (194-92)*v/max_val)},{int(246 + (203-246)*v/max_val)},0.85)"
        for v in importance_df["Mean |SHAP|"]
    ]

    fig_global = go.Figure(go.Bar(
        x=importance_df["Mean |SHAP|"],
        y=importance_df["Feature"],
        orientation="h",
        marker_color=colours,
        text=importance_df["Mean |SHAP|"].round(4),
        textposition="outside",
    ))
    fig_global.update_layout(
        paper_bgcolor="#0A0C14", plot_bgcolor="#12151F",
        font=dict(color="#E2E8F0", family="DM Sans"),
        margin=dict(l=10, r=60, t=20, b=10),
        height=550,
        xaxis=dict(gridcolor="#1E2235"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_global, use_container_width=True)

    st.markdown("""
    <div style="background:#12151F;border:1px solid #1E2235;border-left:4px solid #00C2CB;border-radius:10px;padding:1rem 1.2rem;">
      <b style="color:#00C2CB;">How to read this chart</b><br>
      <span style="font-size:.85rem;color:#94A3B8;">
        Each bar shows the average absolute SHAP value for that feature across all test customers.
        Larger bar = stronger influence on model predictions (both for churn and non-churn).
        Unlike XGBoost's built-in importance, SHAP accounts for feature interactions.
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🌡️ SHAP Beeswarm / Summary Plot")
    st.markdown("Each dot = one customer. Colour = feature value. X-axis = impact on churn prediction.")

    fig_bee, ax = plt.subplots(figsize=(10, 7))
    fig_bee.patch.set_facecolor("#0A0C14")
    ax.set_facecolor("#12151F")

    # Trim to top 15 features
    top_idx = np.argsort(np.abs(shap_values).mean(axis=0))[-15:]
    shap.summary_plot(
        shap_values[:, top_idx],
        X_trans[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False, plot_size=None, color_bar=True,
    )
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1E2235")
    ax.xaxis.label.set_color("#94A3B8")
    plt.tight_layout()
    st.pyplot(fig_bee, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧍 Individual Customer Explanation")

    # Customer selector
    col_sel, col_info = st.columns([1, 2])
    with col_sel:
        max_idx = len(X_raw) - 1
        idx = st.number_input("Customer index (0 to {})".format(max_idx), 0, max_idx, 0)

    shap_row   = shap_values[idx]
    true_label = "Churned 🔴" if y_true[idx] == 1 else "Retained 🟢"
    base_value = shap_values.mean()   # approximate expected value

    with col_info:
        st.metric("True Label", true_label)
        pred_prob = 1 / (1 + np.exp(-(base_value + shap_row.sum())))
        st.metric("Predicted Probability", f"{pred_prob:.1%}")

    # Waterfall chart — top N drivers
    n = min(len(shap_row), len(feature_names))
    df_shap = pd.DataFrame({
        "Feature": feature_names[:n],
        "SHAP":    shap_row[:n],
    }).reindex(pd.Series(np.abs(shap_row[:n])).nlargest(15).index)

    colours = ["#FF6B6B" if v > 0 else "#00C2CB" for v in df_shap["SHAP"]]
    df_sorted = df_shap.sort_values("SHAP")

    fig_wf = go.Figure(go.Bar(
        x=df_sorted["SHAP"],
        y=df_sorted["Feature"],
        orientation="h",
        marker_color=["#FF6B6B" if v > 0 else "#00C2CB" for v in df_sorted["SHAP"]],
        text=[f"{v:+.4f}" for v in df_sorted["SHAP"]],
        textposition="outside",
    ))
    fig_wf.update_layout(
        title=f"SHAP Contribution — Customer #{idx}",
        paper_bgcolor="#0A0C14", plot_bgcolor="#12151F",
        font=dict(color="#E2E8F0", family="DM Sans"),
        margin=dict(l=10, r=60, t=50, b=10),
        height=500,
        xaxis=dict(gridcolor="#1E2235", title="SHAP value (impact on log-odds of churn)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    fig_wf.add_vline(x=0, line_color="#4B5563", line_width=1)
    st.plotly_chart(fig_wf, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:2rem;margin-top:.5rem;font-size:.83rem;">
      <div><span style="color:#FF6B6B;font-weight:700;">■ Red bars</span> — push towards churn (increase probability)</div>
      <div><span style="color:#00C2CB;font-weight:700;">■ Teal bars</span> — push against churn (reduce probability)</div>
    </div>
    """, unsafe_allow_html=True)

    # Raw feature values for selected customer
    with st.expander("📋 Raw feature values for this customer"):
        raw_row = X_raw.iloc[idx]
        disp_df = pd.DataFrame({"Feature": raw_row.index, "Value": raw_row.values})
        st.dataframe(disp_df, use_container_width=True, hide_index=True)