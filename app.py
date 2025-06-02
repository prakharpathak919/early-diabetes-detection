import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('models/diabetes_model.pkl')

st.title("🩺 Early Diabetes Detection")
st.write("Enter patient details to predict the risk of diabetes.")

# Input fields
age = st.slider("Age", 10, 100, 40)
bmi = st.slider("BMI", 10.0, 40.0, 25.0)
glucose = st.slider("Glucose", 50, 200, 120)
bp = st.slider("Blood Pressure", 50, 120, 80)
insulin = st.slider("Insulin", 10, 300, 100)
activity = st.slider("Physical Activity (min/day)", 0, 60, 30)
family = st.selectbox("Family History of Diabetes", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])

input_data = np.array([[age, bmi, glucose, bp, insulin, activity, family, smoking]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of diabetes.")
    else:
        st.success("✅ Low risk of diabetes.")
