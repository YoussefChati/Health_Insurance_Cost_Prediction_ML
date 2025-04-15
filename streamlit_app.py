import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/insurance_model.pkl")

# Configure the app
st.set_page_config(page_title="Insurance Estimator", page_icon="🧾", layout="centered")

# --- Header ---
st.title("🧾 Health Insurance Estimator")
st.caption("Estimate your expected health insurance cost in seconds.")

st.divider()

# --- Input Form ---
with st.container():
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("🎂 Age", 18, 100, 30)
        sex = st.selectbox("🧑 Sex", ["Male", "Female"])
    
    with col2:
        children = st.selectbox("👶 Number of Children", list(range(0, 11)))
        smoker = st.radio("🚬 Smoker", ["Yes", "No"], horizontal=True)

st.divider()

with st.container():
    st.subheader("🌍 Lifestyle")
    col3, col4 = st.columns(2)
    
    with col3:
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
    
    with col4:
        region = st.selectbox("🗺️ Region", ["Southwest", "Southeast", "Northwest", "Northeast"])

# --- Prediction Button ---
st.divider()
predict_btn = st.button("🔮 Estimate My Insurance Cost")

if predict_btn:
    # Prepare input
    input_data = {
        "age": age,
        "sex": sex.lower(),
        "bmi": bmi,
        "children": children,
        "smoker": smoker.lower(),
        "region": region.lower()
    }

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    # --- Output Section ---
    st.success("✅ Prediction Complete!")
    st.subheader("💰 Estimated Insurance Cost:")
    st.markdown(f"""
        <h2 style='text-align: center; color: green; font-size: 36px;'>
            ${prediction:,.2f}
        </h2>
    """, unsafe_allow_html=True)

    st.info("This estimate is based on your inputs and our trained ML model.\nResults may vary from actual insurance quotes.")
