import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from io import StringIO
from streamlit_option_menu import option_menu

# ============ PAGE CONFIG ==========
st.set_page_config(page_title="Early Diabetes Predictor", page_icon="🩺", layout="wide")

# ============ THEME SLIDER ===========
# Theme toggle (add to top of sidebar, above navigation)
selected_theme = option_menu(
    menu_title=None,
    options=["Day", "Night"],
    icons=["sun", "moon"],
    menu_icon="cast",
    default_index=0 if st.session_state.get('theme_mode', 'Day') == 'Day' else 1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "rgba(0,0,0,0)"},
        "icon": {"color": "#fdc43a", "font-size": "24px"},
        "nav-link": {
            "font-size": "18px",
            "margin": "0 8px",
            "color": "#38404a",
            "border-radius": "8px",
            "background-color": "#f0f2f6",
            "padding": "6px 20px"
        },
        "nav-link-selected": {
            "background-color": "#00337C",
            "color": "#fff",
            "font-weight": "bold",
            "box-shadow": "0 2px 12px rgba(0,51,124,0.15)"
        }
    }
)

st.session_state.theme_mode = selected_theme
# ============ THEME CSS ============
day_css = """
<style>
body { background: linear-gradient(110deg, #e8f0fe 0%, #fceabb 100%) !important; }
.main {
    background: linear-gradient(135deg, #e0f7fa 10%, #ffffff 100%) !important;
}
.glass {
    background: rgba(255,255,255,0.38) !important;
    box-shadow: 0 6px 32px rgba(60,60,160,.16) !important;
    border: 1.5px solid #e8eaf6 !important;
    border-radius: 18px !important;
    padding: 32px 38px !important;
    margin: 14px 0 !important;
}
.header { font-size:48px; font-weight:bold; text-align:center; color:#082032; margin-bottom:0; }
.subheader { color:#34495e; font-size:20px; text-align:center; margin-bottom:18px; }
.result-box { padding:20px 10px; border-radius:16px; text-align:center; font-size:25px; font-weight:600; margin-top:20px; box-shadow:0 0 15px rgba(0,0,0,0.07);}
.high-risk { background:rgba(255, 102, 102, 0.17); color:#d63031; border:2.2px solid #d63031;}
.low-risk { background:rgba(46,204,113,0.14); color:#008800; border:2.2px solid #38c172;}
.prob-bar { height:30px; background:#eaf6fb; border-radius:12px; margin:10px 0 4px 0;}
.prob-inner { height:30px; background:linear-gradient(90deg,#00bcd4,#2196f3); border-radius:12px; text-align:right; color:#fff; padding:5px 18px 0 0; font-weight:600; font-size:18px;}
.stButton > button {
    background: linear-gradient(90deg, #10c6ff 0%, #004aad 100%) !important;
    color: white !important; border:none; border-radius:12px; font-size:20px; font-weight:700; padding:16px 38px; margin:8px auto; display:block;
    box-shadow:0 2px 6px rgba(0,0,0,0.08); transition:transform 0.17s, box-shadow 0.17s; outline:none;}
.stButton > button:hover {
    background: linear-gradient(90deg, #2196f3 0%, #006aff 100%) !important;
    transform:scale(1.07); box-shadow:0 6px 18px rgba(50,100,220,0.18);}
</style>
"""

night_css = """
<style>
body { background: linear-gradient(120deg, #22242c 0%, #444 100%) !important; }
.main {
    background: linear-gradient(135deg, #232946 70%, #121212 100%) !important;
}
.glass {
    background: rgba(50,50,70,0.50) !important;
    box-shadow: 0 6px 42px rgba(30,40,80,0.16) !important;
    border: 1.5px solid #313552 !important;
    border-radius: 18px !important;
    padding: 32px 38px !important;
    margin: 14px 0 !important;
}
.header { font-size:48px; font-weight:bold; text-align:center; color:#f3f6fc; margin-bottom:0; }
.subheader { color:#c3cadd; font-size:20px; text-align:center; margin-bottom:18px; }
.result-box { padding:20px 10px; border-radius:16px; text-align:center; font-size:25px; font-weight:600; margin-top:20px; box-shadow:0 0 30px rgba(20,30,80,0.18);}
.high-risk { background:rgba(255,60,60,0.13); color:#ff6868; border:2.2px solid #ff6868;}
.low-risk { background:rgba(46, 204, 113, 0.14); color:#36fd8c; border:2.2px solid #36fd8c;}
.prob-bar { height:30px; background:#31355b; border-radius:12px; margin:10px 0 4px 0;}
.prob-inner { height:30px; background:linear-gradient(90deg,#2193b0,#6dd5ed); border-radius:12px; text-align:right; color:#fff; padding:5px 18px 0 0; font-weight:600; font-size:18px;}
.stButton > button {
    background: linear-gradient(90deg, #355c7d 0%, #6c5b7b 100%) !important;
    color: #fafafa !important; border:none; border-radius:12px; font-size:20px; font-weight:700; padding:16px 38px; margin:8px auto; display:block;
    box-shadow:0 2px 10px rgba(0,0,0,0.21); transition:transform 0.17s, box-shadow 0.17s; outline:none;}
.stButton > button:hover {
    background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%) !important;
    color:#212121 !important; transform:scale(1.07); box-shadow:0 6px 21px rgba(80,170,240,0.14);}
table, th, td {color:#f3f6fc !important;}
</style>
"""

st.markdown(day_css if st.session_state.theme_mode == 'Day' else night_css, unsafe_allow_html=True)

# ============ LOAD MODEL ============
try:
    model = joblib.load('models/diabetes_model.pkl')
except FileNotFoundError:
    st.error("🚫 Model file not found. Please make sure 'models/diabetes_model.pkl' exists.")
    st.stop()

# ============ SIDEBAR NAV =============
#st.sidebar.image("logo.png", width=100, caption="Early Diabetes Predictor", use_container_width=True)
st.sidebar.image("logo.png", width=100, caption="Early Diabetes Prediction")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "💉 Prediction", "💡 About", "🌐 Resources", "🧑‍🤝‍🧑 Contributors"]
)
st.sidebar.write(" ")
st.sidebar.markdown("© 2025 Early Diabetes Predictor")

# ============ SESSION STATE CLEAR ON TAB SWITCH ============
if page != "💉 Prediction":
    for k in ["prediction", "proba", "input_data"]:
        if k in st.session_state:
            del st.session_state[k]

# ============ HOME PAGE ============
if page == "🏠 Home":
    st.markdown('<div class="header">🩺 Early Diabetes Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Predict diabetes risk with state-of-the-art medical ML and personalized visuals.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="glass">
        <h3>👋 Welcome!</h3>
        This smart web app predicts your diabetes risk using key health factors:
        <ul>
        <li>Age, BMI, Blood Glucose, Pressure, Insulin levels...</li>
        <li>Family history, lifestyle and more</li>
        </ul>
        <b>Features:</b>
        <ul>
        <li>Modern, easy interface</li>
        <li>Instant AI-based risk and recommended daily routine</li>
        <li>Downloadable/text sharable report</li>
        </ul>
        <i>For educational/awareness purposes. Not for diagnosis. Consult your doctor for health decisions.</i>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1511174511562-5f7f18b874f8?auto=format&fit=crop&w=400&q=80",
            caption="Example", width=400
            #use_container_width=True
        )
    st.info("Go to **Prediction** tab to start.")

# ============ ABOUT PAGE ============
elif page == "💡 About":
    st.markdown('<div class="header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="glass">
        <b>How does this work?</b><br>
        The tool uses a machine learning model trained on clinical data to estimate diabetes risk for adults. It is based on key biomedical and lifestyle inputs you provide.
        <br><br>
        <b>Disclaimer:</b><br>
        This service is for informational use only and does not substitute for professional medical advice. Model accuracy may vary for individuals outside the generalized population for which it was trained.<br>
        </div>
        """, unsafe_allow_html=True)

# ============ RESOURCES PAGE ============
elif page == "🌐 Resources":
    st.markdown('<div class="header">📝 Resources & Lifestyle Tips</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="glass">
        <h4>🩺 Preventive Health</h4>
        <ul>
          <li><a href="https://www.cdc.gov/diabetes/prevention/index.html" target="_blank">CDC Diabetes Prevention</a></li>
          <li><a href="https://www.who.int/news-room/fact-sheets/detail/diabetes" target="_blank">WHO Facts on Diabetes</a></li>
        </ul>
        <h4>🥗 Healthy Routine</h4>
        <ul>
        <li>Eat more fiber and less sugar</li>
        <li>Move your body every day</li>
        <li>Get enough sleep & hydration</li>
        </ul>
        <h4>🚨 If High Risk:</h4>
        Please show your report to your medical provider.<br>
        <i>Stay healthy and informed!</i>
        </div>
    """, unsafe_allow_html=True)

# ============ CONTRIBUTORS PAGE ============
elif page == "🧑‍🤝‍🧑 Contributors":
    st.markdown('<div class="header">🤝 Contributors & Mentors</div>', unsafe_allow_html=True)

    contributors = [
        {
            "name": "Prakhar Pathak",
            "role": "Lead Developer",
            "linkedin": "https://www.linkedin.com/in/prakhar-pathak-83a554256/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"
        },
        {
            "name": "RishiRaj Singh Parmar",
            "role": "ML Engineer",
            "linkedin": "https://www.linkedin.com"
        },
        {
            "name": "Anuj Anand",
            "role": "Frontend Specialist",
            "linkedin": "https://www.linkedin.com/in/anuj-anand-9867b7253?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app"
        },
        {
            "name": "Divyansh Garg",
            "role": "Data Analyst",
            "linkedin": "https://www.linkedin.com"
        },
        {
            "name": "Ajit Singh",
            "role": "Deployment/QA",
            "linkedin": "https://www.linkedin.com"
        }
    ]
    mentors = [
        {
            "name": "Dr. Khursheed Ahmad Bhat",
            "role": "Research Mentor",
            "linkedin": "https://www.linkedin.com"
        },
        {
            "name": "Ms. Pooja",
            "role": "Clinical Advisor",
            "linkedin": "https://www.linkedin.com"
        }
    ]

    st.markdown("""
    <div class="glass" style="margin-top: 26px; margin-bottom: 10px;">
        <h3 style="text-align:center; margin-bottom:4px;">🤝 Contributors</h3>
        <ul style="font-size:17px; margin-bottom:12px;">
    """, unsafe_allow_html=True)
    for member in contributors:
        st.markdown(
            f'<li><a href="{member["linkedin"]}" target="_blank">{member["name"]}</a> – <i>{member["role"]}</i></li>',
            unsafe_allow_html=True
        )
    st.markdown("""
        </ul>
        <h4 style="margin-bottom:3px;">🧑‍🏫 Mentors</h4>
        <ul style="font-size:16px;">
    """, unsafe_allow_html=True)
    for mentor in mentors:
        st.markdown(
            f'<li><a href="{mentor["linkedin"]}" target="_blank">{mentor["name"]}</a> – <i>{mentor["role"]}</i></li>',
            unsafe_allow_html=True
        )
    st.markdown("</ul></div>", unsafe_allow_html=True)

# ============ PREDICTION PAGE ============
elif page == "💉 Prediction":
    st.markdown('<div class="header">💉 Diabetes Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Confidential screening – no data is stored</div>', unsafe_allow_html=True)
    with st.form(key='diabetes_form'):
        c0, c1 = st.columns([2, 1])
        with c0:
            patient_name = st.text_input("👤 Your Name", max_chars=32)
            age = st.slider("🎂 Age", 10, 100, 40)
            bmi = st.slider("⚖️ BMI", 10.0, 45.0, 25.0, step=0.1, help="Body Mass Index (kg/m²)")
            glucose = st.slider("🩸 Glucose (mg/dL)", 50, 200, 110)
            bp = st.slider("💓 Blood Pressure (mmHg)", 50, 120, 78)
        with c1:
            insulin = st.slider("🧬 Insulin (μU/mL)", 10, 300, 100)
            activity = st.slider("🏃 Physical Activity (min/day)", 0, 90, 30)
            family = st.selectbox(
                "👨‍👩‍👧 Family History of Diabetes",
                [1, 0],
                format_func=lambda x: "Yes" if x else "No"
            )
            smoking = st.selectbox(
                "🚬 Smoking",
                [0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
        st.markdown(
            '<div style="display: flex; justify-content: center; margin-top: 18px;">',
            unsafe_allow_html=True
        )
        submit = st.form_submit_button("🚀 Predict My Risk")
        st.markdown('</div>', unsafe_allow_html=True)

    if submit:
        if not patient_name.strip():
            st.error("Please enter your name.")
            st.stop()
        input_data = np.array([[age, bmi, glucose, bp, insulin, activity, family, smoking]])
        with st.spinner("Analyzing your data..."):
            time.sleep(0.5)
            pred = model.predict(input_data)[0]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1]
            else:
                proba = None
        st.session_state.prediction = pred
        st.session_state.proba = proba
        st.session_state.input_data = {
            'Name': patient_name,
            'Age': f"{age} years",
            'BMI': bmi,
            'Glucose': f"{glucose} mg/dL",
            'Blood Pressure': f"{bp} mmHg",
            'Insulin': f"{insulin} μU/mL",
            'Physical Activity': f"{activity} min/day",
            'Family History': 'Yes' if family else 'No',
            'Smoking': 'Yes' if smoking else 'No'
        }
        st.rerun()

    if st.session_state.get('prediction') is not None:
        pred = st.session_state.prediction
        proba = st.session_state.proba
        input_data_dict = st.session_state.input_data
        age_val = int(input_data_dict['Age'].split()[0])
        bmi_val = float(input_data_dict['BMI'])
        glucose_val = int(input_data_dict['Glucose'].split()[0])
        bp_val = int(input_data_dict['Blood Pressure'].split()[0])
        insulin_val = int(input_data_dict['Insulin'].split()[0])
        activity_val = int(input_data_dict['Physical Activity'].split()[0])

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader(f"📊 Prediction for {input_data_dict['Name']}")
        risk_box_class = "high-risk" if pred == 1 else "low-risk"
        risk_text = "⚠️ <b>High Risk of Diabetes</b>" if pred == 1 else "✅ <b>Low Risk of Diabetes</b>"
        if proba is not None:
            percent = int(proba * 100)
            st.markdown(f"""
            <div class="prob-bar">
                <div class="prob-inner" style="width: {percent}%; background: linear-gradient(90deg, #e57373 {percent}%, #38c172 100%);">
                    {proba:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f'<div class="result-box {risk_box_class}">{risk_text}</div>', unsafe_allow_html=True)
        if pred == 1:
            st.warning("It is <b>recommended to consult a healthcare professional</b>.", icon="⚠️")
        else:
            st.success("Maintain your healthy lifestyle 🤗", icon="✅")

        # ---------- Radar Chart ----------
        st.markdown("### 🕹️ Your Health Markers Chart")
        def radar_chart(input_vals):
            labels = ['Age', 'BMI', 'Glucose', 'BP', 'Insulin', 'Activity']
            ref = [35, 22, 90, 75, 80, 45]
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=input_vals, theta=labels, fill='toself', name='You'))
            fig.add_trace(go.Scatterpolar(r=ref, theta=labels, fill='toself', name='Healthy Avg'))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, margin=dict(l=40, r=40, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        radar_chart([age_val, bmi_val, glucose_val, bp_val, insulin_val, activity_val])

        # ---------- Input Summary Table ----------
        st.markdown("### 📄 Input Summary")
        normal_ranges = {
            "Name": "Text",
            "Age": "Years",
            "BMI": "18.5 - 24.9",
            "Glucose": "70-99 mg/dL",
            "Blood Pressure": "70-90 mmHg",
            "Insulin": "50-150 μU/mL",
            "Physical Activity": "30-60 min/day",
            "Family History": "No/Yes",
            "Smoking": "No/Yes"
        }
        df_summary = pd.DataFrame({
            "Marker": list(input_data_dict.keys()),
            "Value": list(input_data_dict.values()),
            "Normal Range": [normal_ranges.get(k, "N/A") for k in input_data_dict.keys()]
        })
        st.table(df_summary)

        # ---------- Downloadable Report ----------
        report = StringIO()
        report.write("🩺 Diabetes Risk Assessment Report\n==============================\n\n")
        for key, value in input_data_dict.items():
            report.write(f"{key}: {value}\n")
        report.write(f"\nPrediction: {'High Risk' if pred == 1 else 'Low Risk'}\n")
        if proba is not None:
            report.write(f"Probability: {proba:.1%}\n")
        report.write("\nGenerated by Early Diabetes Risk Predictor \n")
        st.download_button(
            "📥 Download Result Report",
            report.getvalue(),
            file_name=f"Diabetes_Report_{input_data_dict['Name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )

        # ---------- Personalized Routine ----------
        st.markdown("### 🌅 Personalized Daily Routine Suggestion")
        if pred == 1:
            st.markdown("""
            <div class="glass">
                <ul style="font-size: 18px; line-height: 1.6;">
                    <li>🥗 <b>Start with fiber-rich, low sugar breakfast:</b> e.g., oats, eggs, berries.</li>
                    <li>🚶 <b>30-45 min brisk walk or light workout daily.</b></li>
                    <li>💧 <b>8-10 glasses water/day.</b></li>
                    <li>🍲 <b>Small balanced meals, avoid refined sugar/fats.</b></li>
                    <li>🍵 <b>Green/herbal tea to boost metabolism.</b></li>
                    <li>🛑 <b>Strictly avoid smoking, alcohol.</b></li>
                    <li>😴 <b>7-8 hours quality sleep.</b></li>
                    <li>🩺 <b>Frequent blood sugar checkups.</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass">
                <ul style="font-size: 18px; line-height: 1.6;">
                    <li>🥗 <b>Maintain balanced diet with fruits, whole grains, greens.</b></li>
                    <li>🏃 <b>At least 20-30 min light activity daily.</b></li>
                    <li>💧 <b>7-8 glasses water daily.</b></li>
                    <li>🚫 <b>Limit sugars/snacks.</b></li>
                    <li>🛑 <b>Avoid smoking, minimize alcohol.</b></li>
                    <li>🧘 <b>Manage stress (mindfulness, hobbies).</b></li>
                    <li>😴 <b>Sleep 7-8 hours/night.</b></li>
                    <li>🩺 <b>Routine family screening every 6-12 months.</b></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center; font-size:14px; color: #7f8c8d;">© 2025 Early Diabetes Predictor</div>',
        unsafe_allow_html=True
    )
