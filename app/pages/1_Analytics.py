"""
1_Analytics.py — Interactive EDA Dashboard
==========================================
Churn distribution · Tenure analysis · Monthly charges ·
Service breakdown · Correlation heatmap
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import pandas as pd
import numpy  as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader         import load_raw_data, download_dataset
from preprocess          import clean
from feature_engineering import engineer_features

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Analytics · ChurnGuard", page_icon="📊", layout="wide")

# ── Shared CSS (re-applied on every page) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0A0C14;--card:#12151F;--border:#1E2235;--teal:#00C2CB;--coral:#FF6B6B;--purple:#8B5CF6;--amber:#F59E0B;--text:#E2E8F0;--muted:#64748B;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1020 0%,#111425 100%)!important;border-right:1px solid var(--border)!important;}
div[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
div[data-testid="metric-container"] label{color:var(--muted)!important;font-size:0.75rem!important;text-transform:uppercase;letter-spacing:.08em;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'Syne',sans-serif!important;font-size:1.6rem!important;font-weight:700!important;}
hr{border-color:var(--border)!important;}
.page-header{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-.5px;margin-bottom:.25rem;}
.page-sub{font-size:.85rem;color:#64748B;margin-bottom:1.5rem;}
.chart-card{background:#12151F;border:1px solid #1E2235;border-radius:14px;padding:1.2rem;}
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

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_data():
    download_dataset()
    raw = load_raw_data()
    df  = engineer_features(clean(raw))
    # Add readable churn label
    df["ChurnLabel"] = df["Churn"].map({1: "Churned", 0: "Retained"})
    return df

df = get_data()

# ── Theme helpers ─────────────────────────────────────────────────────────────
BG   = "#0A0C14"
CARD = "#12151F"
LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(color="#E2E8F0", family="DM Sans, sans-serif"),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#1E2235", zerolinecolor="#1E2235"),
    yaxis=dict(gridcolor="#1E2235", zerolinecolor="#1E2235"),
)
COLORS = {"Churned": "#FF6B6B", "Retained": "#00C2CB"}

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Explore the Telco Churn dataset — distributions, drivers, and correlations.</div>', unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
total     = len(df)
churned   = df["Churn"].sum()
retained  = total - churned
churn_pct = churned / total * 100
avg_tenure      = df["tenure"].mean()
avg_monthly     = df["MonthlyCharges"].mean()
avg_services    = df["ServiceCount"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Customers",   f"{total:,}")
k2.metric("Churned",           f"{churned:,}",  f"{churn_pct:.1f}%")
k3.metric("Retained",          f"{retained:,}", f"{100-churn_pct:.1f}%")
k4.metric("Avg Tenure",        f"{avg_tenure:.1f} mo")
k5.metric("Avg Monthly $",     f"${avg_monthly:.2f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# Row 1 — Churn distribution + Pie
# ══════════════════════════════════════════════════════════════════════════════
col_a, col_b = st.columns([2, 1])

with col_a:
    # Churn by Contract type
    contract_churn = (
        df.groupby(["Contract", "ChurnLabel"])
        .size().reset_index(name="count")
    )
    # Map encoded Contract back to labels if needed
    if df["Contract"].dtype != object:
        contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
        contract_churn["Contract"] = contract_churn["Contract"].map(contract_map)

    fig_contract = px.bar(
        contract_churn, x="Contract", y="count", color="ChurnLabel",
        barmode="group",
        color_discrete_map=COLORS,
        title="Churn by Contract Type",
        labels={"count": "Customers"},
    )
    fig_contract.update_layout(**LAYOUT)
    st.plotly_chart(fig_contract, use_container_width=True)

with col_b:
    fig_pie = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[retained, churned],
        hole=0.62,
        marker=dict(colors=["#00C2CB", "#FF6B6B"],
                    line=dict(color=BG, width=3)),
        textfont=dict(size=12),
    ))
    fig_pie.add_annotation(
        text=f"<b>{churn_pct:.1f}%</b>",
        x=0.5, y=0.5, font_size=22, showarrow=False,
        font=dict(color="#FF6B6B", family="Syne"),
    )
    fig_pie.update_layout(
        title="Churn Rate",
        showlegend=True,
        **{k: v for k, v in LAYOUT.items() if k != "xaxis" and k != "yaxis"},
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 2 — Tenure distribution + Monthly charges
# ══════════════════════════════════════════════════════════════════════════════
col_c, col_d = st.columns(2)

with col_c:
    fig_tenure = px.histogram(
        df, x="tenure", color="ChurnLabel",
        nbins=30, barmode="overlay", opacity=0.75,
        color_discrete_map=COLORS,
        title="Tenure Distribution by Churn",
        labels={"tenure": "Tenure (months)"},
    )
    fig_tenure.update_layout(**LAYOUT)
    st.plotly_chart(fig_tenure, use_container_width=True)

with col_d:
    fig_monthly = px.box(
        df, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel",
        color_discrete_map=COLORS,
        title="Monthly Charges vs Churn",
        points="outliers",
        labels={"MonthlyCharges": "Monthly Charges ($)", "ChurnLabel": ""},
    )
    fig_monthly.update_layout(**LAYOUT)
    st.plotly_chart(fig_monthly, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 3 — Service count + Internet service breakdown
# ══════════════════════════════════════════════════════════════════════════════
col_e, col_f = st.columns(2)

with col_e:
    svc_churn = (
        df.groupby(["ServiceCount", "ChurnLabel"])
        .size().reset_index(name="count")
    )
    fig_svc = px.bar(
        svc_churn, x="ServiceCount", y="count", color="ChurnLabel",
        barmode="stack",
        color_discrete_map=COLORS,
        title="Churn by Number of Services Subscribed",
        labels={"ServiceCount": "Services Subscribed", "count": "Customers"},
    )
    fig_svc.update_layout(**LAYOUT)
    st.plotly_chart(fig_svc, use_container_width=True)

with col_f:
    # Internet service breakdown
    isp_col = "InternetService"
    if df[isp_col].dtype != object:
        isp_map = {0: "No", 1: "DSL", 2: "Fiber optic"}
        df["_ISP"] = df[isp_col].map(isp_map)
    else:
        df["_ISP"] = df[isp_col]

    isp_churn = df.groupby(["_ISP", "ChurnLabel"]).size().reset_index(name="count")
    fig_isp = px.bar(
        isp_churn, x="_ISP", y="count", color="ChurnLabel",
        barmode="group",
        color_discrete_map=COLORS,
        title="Churn by Internet Service Type",
        labels={"_ISP": "Internet Service", "count": "Customers"},
    )
    fig_isp.update_layout(**LAYOUT)
    st.plotly_chart(fig_isp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 4 — Scatter + Tenure group
# ══════════════════════════════════════════════════════════════════════════════
col_g, col_h = st.columns(2)

with col_g:
    sample = df.sample(min(1500, len(df)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="tenure", y="MonthlyCharges",
        color="ChurnLabel",
        color_discrete_map=COLORS,
        opacity=0.55,
        title="Tenure vs Monthly Charges",
        labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly Charges ($)"},
        hover_data=["TotalCharges", "ServiceCount"],
    )
    fig_scatter.update_layout(**LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_h:
    tg_labels = {0: "New (0–12 mo)", 1: "Growing (1–2 yr)", 2: "Loyal (2–4 yr)", 3: "Champion (4+ yr)"}
    df["_TG"] = df["TenureGroup"].map(tg_labels)
    tg_churn  = df.groupby(["_TG", "ChurnLabel"]).size().reset_index(name="count")
    fig_tg = px.bar(
        tg_churn, x="_TG", y="count", color="ChurnLabel",
        barmode="stack",
        color_discrete_map=COLORS,
        title="Churn Rate by Tenure Group",
        labels={"_TG": "Tenure Group", "count": "Customers"},
        category_orders={"_TG": list(tg_labels.values())},
    )
    fig_tg.update_layout(**LAYOUT)
    st.plotly_chart(fig_tg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Row 5 — Correlation heatmap
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔥 Feature Correlation Heatmap")
num_df = df.select_dtypes(include=[np.number]).drop(
    columns=[c for c in ["_TG", "ChurnLabel"] if c in df.columns],
    errors="ignore"
).iloc[:, :18]   # cap at 18 for readability

corr = num_df.corr().round(2)
fig_heat = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.index.tolist(),
    colorscale=[[0,"#FF6B6B"],[0.5,CARD],[1,"#00C2CB"]],
    zmid=0,
    text=corr.values.round(2),
    texttemplate="%{text}",
    textfont=dict(size=9),
))
fig_heat.update_layout(
    paper_bgcolor=BG, plot_bgcolor=CARD,
    font=dict(color="#E2E8F0", family="DM Sans, sans-serif", size=10),
    margin=dict(l=10, r=10, t=40, b=10),
    height=480,
    title="Pearson Correlation Matrix",
)
st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Insights callout
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:#12151F;border:1px solid #1E2235;border-left:4px solid #00C2CB;
border-radius:10px;padding:1.2rem 1.5rem;margin-top:1rem;">
  <div style="font-family:Syne,sans-serif;font-weight:700;margin-bottom:.5rem;color:#00C2CB;">
    💡 Key Insights
  </div>
  <ul style="margin:0;padding-left:1.2rem;line-height:1.9;font-size:.88rem;color:#94A3B8;">
    <li><b>Month-to-month contracts</b> drive the majority of churn — 3× higher rate than two-year contracts.</li>
    <li><b>Fiber optic users</b> churn significantly more despite higher spend — a service quality signal.</li>
    <li><b>New customers (0–12 months)</b> are the highest-risk cohort; early engagement is critical.</li>
    <li><b>MonthlyCharges</b> correlates strongly with churn — customers paying more leave more often.</li>
    <li><b>Service bundles reduce churn</b> — customers with 4+ add-ons show markedly lower attrition.</li>
  </ul>
</div>
""", unsafe_allow_html=True)