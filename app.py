import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from io import StringIO

# Load the model
try:
    model = joblib.load('models/diabetes_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'diabetes_model.pkl' is in the correct directory.")
    st.stop()

# Page configuration
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Enhanced CSS Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        background-attachment: fixed;
    }
    .glass {
        background: rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
        max-width: 500px;
        margin: 0 auto;
        margin-top: 20px;
    }
    .header {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        animation: glow 2s infinite alternate;
        margin-bottom: 10px;
    }
    @keyframes glow {
        from { text-shadow: 0 0 10px #00bcd4; }
        to { text-shadow: 0 0 25px #00bcd4; }
    }
    .subheader {
        color: #34495e;
        font-size: 20px;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00bcd4, #2196f3);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.4s;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #0288d1, #1976d2);
        transform: scale(1.05);
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
    }
    .high-risk {
        background: rgba(255, 102, 102, 0.2);
        color: #d63031;
        border: 2px solid #d63031;
    }
    .low-risk {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 2px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">🩺 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict diabetes risk with simple health indicators</div>', unsafe_allow_html=True)

# Glass Container outside the form
#st.markdown('<div class="glass">', unsafe_allow_html=True)

# Form
with st.form(key='diabetes_form'):
    patient_name = st.text_input("👤 Patient Name", help="Enter the patient's full name")

    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("🎂 Age", 10, 100, 40)
        bmi = st.slider("⚖️ BMI", 10.0, 40.0, 25.0, step=0.1)
        glucose = st.slider("🩸 Glucose (mg/dL)", 50, 200, 120)
        bp = st.slider("💓 Blood Pressure (mmHg)", 50, 120, 80)
    
    with col2:
        insulin = st.slider("🧬 Insulin (μU/mL)", 10, 300, 100)
        activity = st.slider("🏃 Physical Activity (min/day)", 0, 60, 30)
        family = st.selectbox("👨‍👩‍👧 Family History of Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
        smoking = st.selectbox("🚬 Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")

    submit = st.form_submit_button("🚀 Predict Risk")

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if submit:
    if not patient_name.strip():
        st.error("Please enter the patient's name.")
    else:
        input_data = np.array([[age, bmi, glucose, bp, insulin, activity, family, smoking]])
        
        with st.spinner("Analyzing data..."):
            time.sleep(1)
            prediction = model.predict(input_data)[0]
        
        st.session_state.prediction = prediction
        st.session_state.input_data = {
            'Patient Name': patient_name,
            'Age': f"{age} years",
            'BMI': bmi,
            'Glucose': f"{glucose} mg/dL",
            'Blood Pressure': f"{bp} mmHg",
            'Insulin': f"{insulin} μU/mL",
            'Physical Activity': f"{activity} min/day",
            'Family History': 'Yes' if family else 'No',
            'Smoking': 'Yes' if smoking else 'No'
        }

# Show Results
if st.session_state.get('prediction') is not None:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader(f"📊 Prediction Result for {st.session_state.input_data['Patient Name']}")
    
    risk_box_class = "high-risk" if st.session_state.prediction == 1 else "low-risk"
    risk_text = "⚠️ High Risk of Diabetes" if st.session_state.prediction == 1 else "✅ Low Risk of Diabetes"
    
    st.markdown(f'<div class="result-box {risk_box_class}">{risk_text}</div>', unsafe_allow_html=True)

    if st.session_state.prediction == 1:
        st.warning("It is recommended to consult a healthcare professional.")
    else:
        st.success("Maintain a healthy lifestyle to keep your risk low.")
    
    st.subheader("📄 Input Summary")
    for key, value in st.session_state.input_data.items():
        st.write(f"**{key}**: {value}")

    # Downloadable Report
    report = StringIO()
    report.write("🩺 Diabetes Risk Assessment Report\n")
    report.write("==============================\n\n")
    for key, value in st.session_state.input_data.items():
        report.write(f"{key}: {value}\n")
    report.write("\nPrediction: ")
    report.write("High Risk\n" if st.session_state.prediction == 1 else "Low Risk\n")
    report.write("\nGenerated by Diabetes Risk Predictor\n")

    st.download_button(
        "💾 Download Report",
        report.getvalue(),
        file_name=f"Diabetes_Report_{st.session_state.input_data['Patient Name'].replace(' ', '_')}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #7f8c8d;">© 2025 Diabetes Risk Predictor</div>', unsafe_allow_html=True)
