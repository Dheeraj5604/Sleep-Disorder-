import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from model import SleepTransformer


try:
    scaler = joblib.load('scaler.pkl')
    le_target = joblib.load('label_encoder.pkl')
    le_gender = joblib.load('gender_encoder.pkl')
    
   
    model = SleepTransformer(input_dim=10, num_classes=len(le_target.classes_))
    model.load_state_dict(torch.load('sleep_model.pth'))
    model.eval()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run train.py first.")

st.set_page_config(page_title="SleepAI: Transformer Diagnostic", layout="wide")
st.title("🌙 Multidimensional Sleep Disorder Analysis")
st.markdown("---")


with st.sidebar:
    st.header("Patient Metrics")
    age = st.number_input("Age", 18, 90, 21) 
    gender = st.selectbox("Gender", ["Male", "Female"])
    duration = st.slider("Sleep Duration (Hours)", 2.0, 12.0, 5.6)
    efficiency = st.slider("Efficiency (%)", 30, 100, 80)
    hr = st.number_input("Avg Heart Rate", 40, 130, 74)
    steps = st.number_input("Daily Steps", 0, 25000, 6800)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)
    caffeine = st.number_input("Caffeine Intake (mg)", 0, 1000, 297)
    alcohol = st.number_input("Alcohol Intake (Units)", 0, 10, 1)
    screen = st.number_input("Screen Time (Hours)", 0.0, 18.0, 4.3)


if st.button("Generate Diagnostic Report"):
    
    g_enc = le_gender.transform([gender])[0]
    feature_names = [
        'age', 'gender', 'sleep_duration_hours', 'sleep_efficiency_percent', 
        'heart_rate_avg', 'steps_per_day', 'stress_level', 
        'caffeine_intake_mg', 'alcohol_intake_units', 'screen_time_hours'
    ]
    raw_data = pd.DataFrame([[age, g_enc, duration, efficiency, hr, steps, stress, caffeine, alcohol, screen]], 
                            columns=feature_names)
    scaled_data = scaler.transform(raw_data)
    
    
    with torch.no_grad():
        output = model(torch.FloatTensor(scaled_data))
        probs = torch.softmax(output, dim=1).numpy()[0] 
        pred_idx = np.argmax(probs)
        condition = le_target.inverse_transform([pred_idx])[0]

    
    st.markdown(f"### Final Diagnosis: :blue[{condition}]")
    st.info(f"The model is {np.max(probs)*100:.2f}% confident in this analysis.")
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        fig_df = pd.DataFrame({
            "Disorder": le_target.classes_,
            "Probability (%)": probs * 100
        }).sort_values(by="Probability (%)", ascending=False)

        
        fig = px.bar(
            fig_df, 
            x="Probability (%)", 
            y="Disorder", 
            orientation='h',
            color="Probability (%)",
            text="Probability (%)",
            title="Transformer Confidence Distribution (Multidimensional Analysis)",
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**Key Indicators Found:**")
        if stress > 6: st.warning("⚠️ High Stress Levels detected")
        if efficiency < 70: st.warning("⚠️ Low Sleep Efficiency detected")
        if caffeine > 300: st.warning("⚠️ High Caffeine Consumption")
        st.write("---")
        st.caption("Deep Representation Learning allows the model to analyze non-linear relationships between your daily habits and sleep quality.")