"""
2_Predict.py — Real-time Churn Prediction
=========================================
Full customer input form → churn probability → risk indicator → PDF download
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from predict import predict_single

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predict · ChurnGuard", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0A0C14;--card:#12151F;--border:#1E2235;--teal:#00C2CB;--coral:#FF6B6B;--purple:#8B5CF6;--amber:#F59E0B;--text:#E2E8F0;--muted:#64748B;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1020 0%,#111425 100%)!important;border-right:1px solid var(--border)!important;}
div[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
div[data-testid="metric-container"] label{color:var(--muted)!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.08em;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{color:var(--text)!important;font-family:'Syne',sans-serif!important;font-size:1.6rem!important;font-weight:700!important;}
.stButton>button{background:linear-gradient(135deg,#00C2CB,#8B5CF6)!important;color:white!important;border:none!important;border-radius:8px!important;font-family:'Syne',sans-serif!important;font-weight:600!important;letter-spacing:.04em!important;padding:.6rem 1.4rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div>input,.stTextInput>div>div>input{background:var(--card)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stProgress>div>div{background:linear-gradient(90deg,#00C2CB,#8B5CF6)!important;}
hr{border-color:var(--border)!important;}
.page-header{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-.5px;margin-bottom:.25rem;}
.page-sub{font-size:.85rem;color:#64748B;margin-bottom:1.5rem;}
.section-label{font-family:'Syne',sans-serif;font-weight:700;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#64748B;margin:.8rem 0 .4rem;}
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
    st.markdown("**Sample Presets**")
    if st.button("🔴 High Risk Profile"):
        st.session_state["preset"] = "high"
    if st.button("🟢 Low Risk Profile"):
        st.session_state["preset"] = "low"

# ── Preset values ─────────────────────────────────────────────────────────────
PRESETS = {
    "high": dict(
        gender="Female", SeniorCitizen=1, Partner="No", Dependents="No",
        tenure=3, PhoneService="Yes", MultipleLines="No",
        InternetService="Fiber optic", OnlineSecurity="No",
        OnlineBackup="No", DeviceProtection="No", TechSupport="No",
        StreamingTV="Yes", StreamingMovies="Yes",
        Contract="Month-to-month", PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=89.95, TotalCharges=269.85,
    ),
    "low": dict(
        gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
        tenure=54, PhoneService="Yes", MultipleLines="Yes",
        InternetService="DSL", OnlineSecurity="Yes",
        OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
        StreamingTV="No", StreamingMovies="No",
        Contract="Two year", PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)",
        MonthlyCharges=55.20, TotalCharges=2975.00,
    ),
}

preset = PRESETS.get(st.session_state.get("preset"), {})

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-header">🎯 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Fill in the customer profile below — the model predicts churn probability in real time.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Input form — 3-column layout
# ══════════════════════════════════════════════════════════════════════════════
with st.form("churn_form"):
    # ── Demographics ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">👤 Demographics</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    gender       = d1.selectbox("Gender",       ["Female", "Male"],      index=["Female","Male"].index(preset.get("gender","Female")))
    senior       = d2.selectbox("Senior Citizen", [0, 1],               index=preset.get("SeniorCitizen", 0))
    partner      = d3.selectbox("Partner",       ["No", "Yes"],          index=["No","Yes"].index(preset.get("Partner","No")))
    dependents   = d4.selectbox("Dependents",    ["No", "Yes"],          index=["No","Yes"].index(preset.get("Dependents","No")))

    # ── Account info ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🗃️ Account Info</div>', unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    tenure       = a1.number_input("Tenure (months)", 0, 72, int(preset.get("tenure", 12)))
    contract     = a2.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                                index=["Month-to-month","One year","Two year"].index(preset.get("Contract","Month-to-month")))
    paperless    = a3.selectbox("Paperless Billing", ["No", "Yes"],
                                index=["No","Yes"].index(preset.get("PaperlessBilling","No")))
    a4, a5 = st.columns(2)
    payment      = a4.selectbox("Payment Method",
                                ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],
                                index=["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"].index(preset.get("PaymentMethod","Electronic check")))
    monthly_chg  = a5.number_input("Monthly Charges ($)", 18.0, 120.0, float(preset.get("MonthlyCharges", 55.0)), step=0.5)

    total_chg    = st.number_input("Total Charges ($)", 0.0, 10000.0, float(preset.get("TotalCharges", monthly_chg * tenure or 55.0)), step=10.0)

    # ── Phone services ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📞 Phone Services</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    phone        = p1.selectbox("Phone Service", ["Yes", "No"],
                                index=["Yes","No"].index(preset.get("PhoneService","Yes")))
    multi_lines  = p2.selectbox("Multiple Lines", ["No", "Yes", "No phone service"],
                                index=["No","Yes","No phone service"].index(preset.get("MultipleLines","No")))

    # ── Internet services ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🌐 Internet Services</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    internet     = i1.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                                index=["DSL","Fiber optic","No"].index(preset.get("InternetService","DSL")))
    online_sec   = i2.selectbox("Online Security", ["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("OnlineSecurity","No")))
    online_bkp   = i3.selectbox("Online Backup",   ["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("OnlineBackup","No")))
    i4, i5, i6 = st.columns(3)
    dev_prot     = i4.selectbox("Device Protection",["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("DeviceProtection","No")))
    tech_sup     = i5.selectbox("Tech Support",     ["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("TechSupport","No")))
    stream_tv    = i6.selectbox("Streaming TV",     ["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("StreamingTV","No")))
    stream_mov   = st.selectbox("Streaming Movies", ["No","Yes","No internet service"],
                                index=["No","Yes","No internet service"].index(preset.get("StreamingMovies","No")))

    # ── Submit ─────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Churn Probability", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Prediction results
# ══════════════════════════════════════════════════════════════════════════════
if submitted:
    customer = {
        "gender": gender, "SeniorCitizen": senior,
        "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multi_lines, "InternetService": internet,
        "OnlineSecurity": online_sec, "OnlineBackup": online_bkp,
        "DeviceProtection": dev_prot, "TechSupport": tech_sup,
        "StreamingTV": stream_tv, "StreamingMovies": stream_mov,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_chg, "TotalCharges": total_chg,
    }

    with st.spinner("Running prediction…"):
        try:
            result = predict_single(customer)
        except FileNotFoundError:
            st.error("⚠️  Model not found. Run `python src/train.py` first, then reload this page.")
            st.stop()

    prob  = result["probability"]
    risk  = result["risk"]
    pred  = result["prediction"]

    st.markdown("---")
    st.markdown("### 🔮 Prediction Results")

    # ── Result cards ───────────────────────────────────────────────────────────
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn Probability", f"{prob*100:.1f}%")
    r2.metric("Risk Level",        f"{risk['emoji']}  {risk['label']}")
    r3.metric("Prediction",        "🔴 Will Churn" if pred else "🟢 Will Stay")

    # ── Gauge chart ───────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob * 100,
        number = {"suffix": "%", "font": {"family": "Syne", "size": 36, "color": risk["colour"]}},
        delta  = {"reference": 26.5, "valueformat": ".1f",
                  "increasing": {"color": "#FF6B6B"}, "decreasing": {"color": "#10B981"}},
        gauge  = {
            "axis":  {"range": [0, 100], "tickcolor": "#64748B", "dtick": 20},
            "bar":   {"color": risk["colour"], "thickness": 0.25},
            "bgcolor": "#12151F",
            "bordercolor": "#1E2235",
            "steps": [
                {"range": [0,  30], "color": "rgba(16,185,129,0.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 80], "color": "rgba(239,68,68,0.15)"},
                {"range": [80,100], "color": "rgba(124,58,237,0.15)"},
            ],
            "threshold": {"line": {"color": risk["colour"], "width": 3}, "value": prob * 100},
        },
        title  = {"text": "Churn Risk Gauge", "font": {"family": "Syne", "size": 14, "color": "#64748B"}},
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0A0C14",
        font=dict(color="#E2E8F0"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Progress bar ──────────────────────────────────────────────────────────
    st.markdown(f"**Risk Score: {prob*100:.1f}%**")
    st.progress(prob)

    # ── Recommended action ────────────────────────────────────────────────────
    actions = {
        1: ("🟢 **Low Risk** — No immediate action required. Standard engagement plan.", "#10B981"),
        2: ("🟡 **Medium Risk** — Schedule a proactive check-in within 30 days. Offer a loyalty perk.", "#F59E0B"),
        3: ("🔴 **High Risk** — Immediate outreach required. Offer a discount or plan upgrade.", "#EF4444"),
        4: ("🚨 **Critical** — Customer is almost certainly leaving. Escalate to retention team NOW.", "#7C3AED"),
    }
    action_text, action_color = actions[risk["level"]]

    st.markdown(f"""
    <div style="background:#12151F;border:1px solid {action_color}44;
    border-left:4px solid {action_color};border-radius:10px;
    padding:1rem 1.2rem;margin-top:.5rem;">
      <div style="font-family:Syne,sans-serif;font-weight:700;color:{action_color};margin-bottom:.3rem;">
        💼 Recommended Action
      </div>
      <div style="font-size:.9rem;color:#94A3B8;">{action_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Factor summary ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    factors = []
    if contract == "Month-to-month":
        factors.append(("📋 Month-to-month contract", "High churn risk factor", "#FF6B6B"))
    elif contract == "Two year":
        factors.append(("📋 Two-year contract", "Loyalty indicator — reduces churn risk", "#10B981"))
    if internet == "Fiber optic":
        factors.append(("🌐 Fiber optic internet", "Associated with elevated churn rate", "#F59E0B"))
    if tenure < 12:
        factors.append((f"⏱ Short tenure ({tenure} months)", "New customers churn more", "#FF6B6B"))
    elif tenure > 36:
        factors.append((f"⏱ Long tenure ({tenure} months)", "Loyal customer — lower churn risk", "#10B981"))
    if payment == "Electronic check":
        factors.append(("💳 Electronic check payment", "Correlated with higher churn", "#F59E0B"))
    if senior == 1 and partner == "No":
        factors.append(("👴 Senior, no partner", "Higher churn risk group", "#EF4444"))

    if factors:
        st.markdown("**🔑 Key Risk Factors**")
        for icon_label, desc, color in factors:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:.8rem;
            background:#12151F;border:1px solid #1E2235;border-radius:8px;
            padding:.6rem 1rem;margin:.3rem 0;">
              <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div>
              <div>
                <span style="font-weight:600;font-size:.88rem;">{icon_label}</span>
                <span style="color:#64748B;font-size:.83rem;margin-left:.5rem;">{desc}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── PDF download ──────────────────────────────────────────────────────────
    def generate_pdf_report(customer, result):
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()

        # ── Header ─────────────────────────────────────
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "ChurnGuard Report", ln=True, align="C")
        pdf.ln(5)

        # ── Prediction Summary ─────────────────────────
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Prediction Summary", ln=True)

        pdf.set_font("Arial", size=12)

        prediction = "Will Churn" if result["prediction"] else "Will Stay"
        probability = f"{result['probability']*100:.2f}%"
        risk = result["risk"]

        pdf.cell(0, 8, f"Prediction : {prediction}", ln=True)
        pdf.cell(0, 8, f"Probability: {probability}", ln=True)
        pdf.cell(0, 8, f"Risk Level : {risk['label']}", ln=True)

        pdf.ln(5)

        # ── Recommended Action ─────────────────────────
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recommended Action", ln=True)

        pdf.set_font("Arial", size=11)

        actions = {
            1: "Low Risk - No immediate action required.",
            2: "Medium Risk - Schedule a proactive check-in. Offer a loyalty perk.",
            3: "High Risk - Immediate outreach. Offer discount or upgrade.",
            4: "Critical - Customer likely to churn. Escalate immediately.",
        }

        pdf.multi_cell(0, 7, actions[risk["level"]])

        pdf.ln(5)

        # ── Key Risk Factors ───────────────────────────
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Key Risk Factors", ln=True)

        pdf.set_font("Arial", size=11)

        factors = []

        if customer["Contract"] == "Month-to-month":
            factors.append("Month-to-month contract (high churn risk)")

        if customer["InternetService"] == "Fiber optic":
            factors.append("Fiber optic internet (higher churn rate)")

        if customer["tenure"] < 12:
            factors.append(f"Short tenure ({customer['tenure']} months)")

        if customer["PaymentMethod"] == "Electronic check":
            factors.append("Electronic check payment (high churn correlation)")

        if customer["SeniorCitizen"] == 1 and customer["Partner"] == "No":
            factors.append("Senior customer with no partner (higher churn group)")

        if not factors:
            factors.append("No strong risk factors detected")

        for f in factors:
            pdf.cell(0, 7, f"- {f}", ln=True)

        pdf.ln(5)

        # ── Customer Details ──────────────────────────
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Customer Details", ln=True)

        pdf.set_font("Arial", size=10)

        for key, value in customer.items():
            pdf.cell(0, 6, f"{key}: {value}", ln=True)

        # ── Footer ────────────────────────────────────
        pdf.ln(8)
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 10, "Generated by Harsh's ChurnGuard AI", align="C")

        # ── Return bytes ──────────────────────────────
        output = pdf.output(dest='S')

        if isinstance(output, str):
            return output.encode('latin-1')
        else:
            return bytes(output)
        
    pdf_bytes = generate_pdf_report(customer, result)

    if isinstance(pdf_bytes, bytes):
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="churnguard_report.pdf",
            mime="application/pdf",
        )
    else:
        st.error("PDF generation failed")