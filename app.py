import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tempfile
from fpdf import FPDF

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Crop Yield System",
    page_icon="🌾",
    layout="wide"
)

# =========================================================
# BACKGROUND + UI STYLE
# =========================================================
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1605000797499-95a51c5269ae");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.block-container {
    padding-top: 1rem;
}

.card {
    background: black;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("crop_model.pkl")

model = load_model()

# =========================================================
# TITLE
# =========================================================
st.markdown("""
<h1 style='text-align:center;color:#1B5E20;'>🌾 AI Crop Yield Prediction System</h1>
<p style='text-align:center;color:#2E7D32;'>Smart Agriculture AI Dashboard</p>
""", unsafe_allow_html=True)

st.divider()

# =========================================================
# INPUT SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:
    fertilizer = st.number_input("Fertilizer", 0.0, 500.0, 100.0)
    nitrogen = st.number_input("Nitrogen", 0.0, 150.0, 50.0)
    phosphorus = st.number_input("Phosphorus", 0.0, 150.0, 40.0)

with col2:
    temperature = st.number_input("Temperature", 0.0, 60.0, 25.0)
    potassium = st.number_input("Potassium", 0.0, 150.0, 40.0)

# =========================================================
# DATAFRAME
# =========================================================
input_df = pd.DataFrame({
    "Fertilizer": [fertilizer],
    "temp": [temperature],
    "N": [nitrogen],
    "P": [phosphorus],
    "K": [potassium]
})

# =========================================================
# EXPLANATION ENGINE
# =========================================================
def explain(pred, n, p, k, f, t):
    msg = []

    if pred < 20:
        msg.append("Low yield detected → Improve soil quality")
    elif pred < 50:
        msg.append("Moderate yield → Optimization needed")
    else:
        msg.append("Good yield conditions")

    if n < 40:
        msg.append("Increase Nitrogen (leaf growth)")
    if p < 40:
        msg.append("Increase Phosphorus (root growth)")
    if k < 40:
        msg.append("Increase Potassium (resistance)")

    if f < 80:
        msg.append("Fertilizer is slightly low")
    if t > 38:
        msg.append("High temperature may reduce yield")

    return msg

# =========================================================
# CLEAN TEXT
# =========================================================
def clean(text):
    return str(text).encode("ascii", "ignore").decode("ascii")

# =========================================================
# PDF CLASS
# =========================================================
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Crop Yield Prediction Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 9)
        self.cell(0, 10, "AI Engineer Muhammad Idrees", align="C")

def create_pdf(data):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for k, v in data.items():
        pdf.cell(0, 10, f"{clean(k)}: {clean(v)}", ln=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🚀 Predict Crop Yield"):

    try:
        pred = model.predict(input_df)[0]

        if pred < 20:
            category = "Low Yield"
            color = "#e74c3c"
        elif pred < 50:
            category = "Moderate Yield"
            color = "#f1c40f"
        else:
            category = "High Yield"
            color = "#2ecc71"

        # =================================================
        # RESULT CARD (FIXED)
        # =================================================
        st.markdown(f"""
        <div class="card" style="border-left:6px solid {color};">
            <h3>Predicted Yield</h3>
            <h1>{pred:.2f} Tons / Hectare</h1>
            <h3 style="color:{color};">{category}</h3>
        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # SMALL CHART (FIXED SIZE)
        # =================================================
        st.subheader("📊 Parameter Visualization")

        fig, ax = plt.subplots(figsize=(4, 2))

        ax.bar(
            ["Fertilizer", "Temp", "N", "P", "K"],
            [fertilizer, temperature, nitrogen, phosphorus, potassium],
            color="green"
        )

        ax.set_title("Crop Inputs", fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        # =================================================
        # AI EXPLANATION
        # =================================================
        st.subheader("🧠 AI Explanation")

        for m in explain(pred, nitrogen, phosphorus, potassium, fertilizer, temperature):
            st.info(m)

        # =================================================
        # PDF REPORT
        # =================================================
        report_data = {
            "Fertilizer": fertilizer,
            "Temperature": temperature,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Yield": round(float(pred), 2),
            "Category": category
        }

        st.subheader("📄 Download Report")

        pdf_path = create_pdf(report_data)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                f,
                file_name="crop_yield_report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div style="text-align:center;color:#1B5E20;">
Built with Streamlit • Machine Learning • AI Agriculture System
</div>
""", unsafe_allow_html=True)