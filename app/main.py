"""
main.py — ChurnGuard Dashboard Entry Point
==========================================
Run:  streamlit run app/main.py
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title    = "ChurnGuard",
    page_icon     = "🔮",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Global CSS — dark startup aesthetic ───────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg:      #0A0C14;
  --card:    #12151F;
  --border:  #1E2235;
  --teal:    #00C2CB;
  --coral:   #FF6B6B;
  --purple:  #8B5CF6;
  --amber:   #F59E0B;
  --text:    #E2E8F0;
  --muted:   #64748B;
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1020 0%, #111425 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
.sidebar-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.6rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, #00C2CB, #8B5CF6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.2rem;
}
.sidebar-tagline {
  font-size: 0.75rem;
  color: var(--muted);
  margin-bottom: 1.5rem;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
  color: var(--muted) !important;
  font-size: 0.75rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-family: 'Syne', sans-serif !important;
  font-size: 1.6rem !important;
  font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #00C2CB, #8B5CF6) !important;
  color: white !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  padding: 0.6rem 1.4rem !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Selectbox / input ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #00C2CB, #8B5CF6) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Plotly chart bg override ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: 7px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,#00C2CB22,#8B5CF622) !important;
  color: var(--teal) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔮 ChurnGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Customer Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Navigate**")
    st.page_link("main.py",                  label="🏠  Home",             icon=None)
    st.page_link("pages/1_Analytics.py",     label="📊  Analytics",         icon=None)
    st.page_link("pages/2_Predict.py",       label="🎯  Predict Churn",     icon=None)
    st.page_link("pages/3_Model_Performance.py", label="📈  Model Performance", icon=None)
    st.page_link("pages/4_Explainability.py",    label="🔍  Explainability",    icon=None)
    st.markdown("---")
    st.caption("v1.0.0 · IBM Telco Dataset · XGBoost")

# ── Hero section ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
  padding: 3rem 2rem;
  background: linear-gradient(135deg, #0D1020 0%, #111830 50%, #0D1020 100%);
  border: 1px solid #1E2235;
  border-radius: 16px;
  text-align: center;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(0,194,203,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(139,92,246,0.08) 0%, transparent 60%);
  "></div>
  <div style="position: relative; z-index: 1;">
    <div style="
      font-family: Syne, sans-serif;
      font-size: 3rem;
      font-weight: 800;
      letter-spacing: -1px;
      background: linear-gradient(90deg, #00C2CB, #8B5CF6);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    ">🔮 ChurnGuard</div>
    <div style="
      font-size: 1.15rem;
      color: #94A3B8;
      max-width: 560px;
      margin: 0 auto 1.5rem;
      line-height: 1.6;
    ">
      A production-grade ML platform for predicting, explaining,<br>and acting on customer churn — built with XGBoost + SHAP.
    </div>
    <div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap;">
      <span style="
        background: rgba(0,194,203,0.12);
        border: 1px solid rgba(0,194,203,0.3);
        color: #00C2CB;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-family: Syne, sans-serif;
        font-weight: 600;
      ">⚡ XGBoost · AUC 0.87</span>
      <span style="
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.3);
        color: #8B5CF6;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-family: Syne, sans-serif;
        font-weight: 600;
      ">🔍 SHAP Explainability</span>
      <span style="
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.3);
        color: #F59E0B;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-family: Syne, sans-serif;
        font-weight: 600;
      ">📊 Interactive Analytics</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick stats ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Customers",      "7,043",  "IBM Telco Dataset")
c2.metric("📉 Churn Rate",     "26.5%",  "Class imbalance handled")
c3.metric("🏆 Best AUC",       "0.871",  "XGBoost + SMOTE")
c4.metric("🔬 Features",       "27",     "incl. 6 engineered")

st.markdown("---")

# ── Feature cards ─────────────────────────────────────────────────────────────
st.markdown("""
<h3 style="font-family:Syne,sans-serif;font-weight:700;margin-bottom:1rem;">
  What's inside
</h3>
""", unsafe_allow_html=True)

cards = [
    ("📊", "Analytics",        "Interactive EDA — distributions, correlations, churn drivers."),
    ("🎯", "Predict",          "Enter any customer profile and get an instant churn probability."),
    ("📈", "Model Performance","ROC curve, confusion matrix, and 3-model comparison."),
    ("🔍", "Explainability",   "SHAP global & local plots — understand every prediction."),
]

cols = st.columns(4)
for col, (icon, title, desc) in zip(cols, cards):
    col.markdown(f"""
    <div style="
      background: #12151F;
      border: 1px solid #1E2235;
      border-radius: 12px;
      padding: 1.4rem;
      height: 100%;
      transition: border-color 0.2s;
    ">
      <div style="font-size:1.8rem; margin-bottom:0.6rem;">{icon}</div>
      <div style="
        font-family: Syne, sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.4rem;
      ">{title}</div>
      <div style="font-size:0.82rem; color:#64748B; line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem;">
  Built with Python · scikit-learn · XGBoost · SHAP · Streamlit · Plotly
</div>
""", unsafe_allow_html=True)