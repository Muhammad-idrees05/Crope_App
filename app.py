import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")

model = load_model()

# ----------------------------
# UI Header
# ----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🌾 Crop Yield Prediction System</h1>
    <p style='text-align: center; color: grey;'>
    AI-based crop yield estimation using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🔧 Input Parameters")

Fertilizer = st.sidebar.number_input("Fertilizer ", min_value=0.0, value=100.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, value=25.0)
nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, value=50.0)
phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, value=40.0)
potassium = st.sidebar.number_input("Potassium", min_value=0.0, value=40.0)

# ----------------------------
# Prediction Section
# ----------------------------
st.subheader("📊 Prediction")

input_data = np.array([[  
    Fertilizer,
    temperature,
    nitrogen,
    phosphorus,
    potassium
]])

if st.button("🚀 Predict Crop Yield", use_container_width=True):
    try:
        prediction = model.predict(input_data)[0]

        st.success("✅ Prediction Successful!")
        st.metric(
            label="🌱 Estimated Crop Yield",
            value=f"{prediction:.2f} tons/hectare"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: grey;'>Developed using Streamlit & Scikit-Learn</p>",
    unsafe_allow_html=True
)
