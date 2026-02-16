import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#-----------------------------------------------------------
#background
import base64
def set_bg(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("assets/bg.jpg")


#-----------------------------------------------------------------------

# Load trained artifacts
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction Dashboard")
st.caption("Predict customer churn probability using Machine Learning")

st.markdown("---")

# USER INPUTS
st.subheader("🧾 Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

st.markdown("---")

# BUILD INPUT VECTOR 
input_df = pd.DataFrame(0, index=[0], columns=feature_names)

# Numerical
if "tenure" in input_df.columns:
    input_df["tenure"] = tenure
if "MonthlyCharges" in input_df.columns:
    input_df["MonthlyCharges"] = monthly

# Binary
if "SeniorCitizen" in input_df.columns:
    input_df["SeniorCitizen"] = 1 if senior == "Yes" else 0
if "TechSupport" in input_df.columns:
    input_df["TechSupport"] = 1 if tech_support == "Yes" else 0
if "OnlineSecurity" in input_df.columns:
    input_df["OnlineSecurity"] = 1 if online_security == "Yes" else 0

# One-hot
for col in [
    f"Contract_{contract}",
    f"InternetService_{internet_service}",
    f"PaymentMethod_{payment}"
]:
    if col in input_df.columns:
        input_df[col] = 1

# PREDICTION
if st.button("🔮 Predict Churn", use_container_width=True):
    input_scaled = scaler.transform(input_df.values)
    churn_prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 📈 Prediction Result")

    # Metric card
    st.metric(
        label="Churn Probability",
        value=f"{churn_prob:.2f}"
    )

    # Progress bar
    st.progress(int(churn_prob * 100))

    # Risk label
    if churn_prob >= 0.6:
        st.error("🚨 High Risk of Churn")
    elif churn_prob >= 0.4:
        st.warning("⚠️ Medium Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

    st.markdown("---")

    # SIMPLE VISUAL EXPLANATION (Heuristic)
    st.subheader("📊 Key Factors Influencing This Prediction")

    factors = {
        "Low Tenure": max(0, (12 - tenure) / 12),
        "High Monthly Charges": monthly / 200,
        "Month-to-Month Contract": 1 if contract == "Month-to-month" else 0,
        "No Tech Support": 1 if tech_support == "No" else 0,
        "No Online Security": 1 if online_security == "No" else 0
    }

    factor_df = pd.DataFrame({
        "Factor": factors.keys(),
        "Impact Level": factors.values()
    })

    fig, ax = plt.subplots()
    ax.barh(factor_df["Factor"], factor_df["Impact Level"])
    ax.set_xlabel("Relative Impact")
    ax.set_xlim(0, 1)

    st.pyplot(fig)

    st.markdown("---")

    # Input Summary
    st.subheader("📝 Input Summary")

    st.dataframe(
        pd.DataFrame({
            "Feature": [
                "Tenure", "Monthly Charges", "Contract",
                "Payment Method", "Tech Support",
                "Online Security", "Internet Service"
            ],
            "Value": [
                tenure, monthly, contract,
                payment, tech_support,
                online_security, internet_service
            ]
        }),
        use_container_width=True
    )
