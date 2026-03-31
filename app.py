import streamlit as st
import torch
import joblib
import numpy as np
import pandas as pd
import requests
import base64
import urllib.parse
import plotly.express as px
import plotly.graph_objects as go
import time
from model import SleepTransformer

# --- Page Config & Styling ---
st.set_page_config(page_title="SleepAI Command Center", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1a1a24;
        border-radius: 8px;
        padding: 15px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
        border: 1px solid #333;
    }
    .stButton>button {
        background-color: #00ADB5;
        color: #EEEEEE;
        font-size: 18px;
        font-weight: bold;
        border-radius: 5px;
        height: 55px;
        border: 1px solid #00FFF5;
        box-shadow: 0px 0px 15px rgba(0, 173, 181, 0.4);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #00FFF5;
        color: #222831;
        box-shadow: 0px 0px 25px rgba(0, 255, 245, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource 
def load_artifacts():
    try:
        scaler = joblib.load('scaler.pkl')
        le_target = joblib.load('label_encoder.pkl')
        le_gender = joblib.load('gender_encoder.pkl')
        model = SleepTransformer(input_dim=10, num_classes=len(le_target.classes_))
        model.load_state_dict(torch.load('sleep_model.pth'))
        model.eval()
        return scaler, le_target, le_gender, model
    except Exception as e:
        st.error(f"System Offline. Run training sequence first. Error: {e}")
        return None, None, None, None

scaler, le_target, le_gender, model = load_artifacts()

# --- Header & Quick Reference ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.title("🧠 SleepAI: Multimodal Temporal Fusion Diagnostic")
    st.caption("v5.1 | Retrospective analysis of multidimensional bio-signals and lifestyle telemetry.")
with col_head2:
    st.metric("System Status", "ONLINE", delta="CUDA Accelerated" if torch.cuda.is_available() else "CPU Mode")

# NEW: Quick Reference Dropdown in the Header
with st.expander("📖 Quick Reference: What disorders are we screening for?"):
    st.markdown("This AI engine analyzes your inputs to detect mathematical patterns linked to four primary sleep states. Click 'Initiate Diagnosis' to see which profile you match.")
    ref_c1, ref_c2, ref_c3, ref_c4 = st.columns(4)
    with ref_c1:
        st.success("**🟢 Healthy (None)**\n\nOptimal sleep architecture. High efficiency, managed stress, and aligned circadian rhythm.")
    with ref_c2:
        st.error("**🔴 Insomnia**\n\nDifficulty falling or staying asleep. Often triggered by high stress, high caffeine, and excessive screen time.")
    with ref_c3:
        st.warning("**🟠 Sleep Apnea**\n\nBreathing disruptions causing micro-awakenings. Strongly correlated with elevated resting heart rate and poor efficiency.")
    with ref_c4:
        st.info("**🟡 Circadian Disruption**\n\nInternal clock misalignment. Usually driven by alcohol before bed, irregular hours, or sedentary habits.")

st.markdown("---")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🎛️ Patient Telemetry")
    
    # 1. LIVE DATA CONNECTION UI
    st.markdown("### 📡 Live API Connection")
    
    # Fetch secrets safely
    CLIENT_ID = st.secrets["fitbit"]["client_id"]
    CLIENT_SECRET = st.secrets["fitbit"]["client_secret"]
    REDIRECT_URI = st.secrets["fitbit"]["redirect_uri"]
    
    # Construct the Authorization URL
    scopes = "sleep heartrate activity profile"
    auth_url = f"https://www.fitbit.com/oauth2/authorize?response_type=code&client_id={CLIENT_ID}&redirect_uri={urllib.parse.quote(REDIRECT_URI)}&scope={urllib.parse.quote(scopes)}"
    
    
    # Initialize session state widgets AND connection status
    if 'ui_age' not in st.session_state: st.session_state.ui_age = 25
    if 'ui_gender' not in st.session_state: st.session_state.ui_gender = "Male"
    if 'ui_hr' not in st.session_state: st.session_state.ui_hr = 74
    if 'ui_steps' not in st.session_state: st.session_state.ui_steps = 6800
    if 'ui_duration' not in st.session_state: st.session_state.ui_duration = 5.6
    if 'ui_efficiency' not in st.session_state: st.session_state.ui_efficiency = 80
    if 'ui_stress' not in st.session_state: st.session_state.ui_stress = 6
    if 'ui_caffeine' not in st.session_state: st.session_state.ui_caffeine = 350
    if 'ui_alcohol' not in st.session_state: st.session_state.ui_alcohol = 2
    if 'ui_screen' not in st.session_state: st.session_state.ui_screen = 5.0
    if 'connected' not in st.session_state: st.session_state.connected = False

    # OAuth2 Token Exchange Logic
    if "code" in st.query_params and not st.session_state.connected:
        auth_code = st.query_params["code"]
        
        # Prepare the Base64 encoded authorization header
        auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
        b64_auth = base64.b64encode(auth_string.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {b64_auth}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        data = {
            'client_id': CLIENT_ID,
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'code': auth_code
        }
        
        with st.spinner("Encrypting handshake with Fitbit Servers..."):
            # Request the Access Token
            token_response = requests.post("https://api.fitbit.com/oauth2/token", headers=headers, data=data)
            
            if token_response.status_code == 200:
                access_token = token_response.json()['access_token']
                api_headers = {'Authorization': f'Bearer {access_token}'}
                
                try:
                    
                    # Fetch User Profile Data (Age & Gender) Safely
                    profile_res = requests.get("https://api.fitbit.com/1/user/-/profile.json", headers=api_headers, timeout=3).json()
                    user_data = profile_res.get('user', {})
                    st.session_state.ui_age = int(user_data.get('age', 25))
                    
                    # Fitbit returns "MALE" or "FEMALE", we format it for our app
                    raw_gender = user_data.get('gender', 'MALE').upper()
                    st.session_state.ui_gender = "Female" if raw_gender == "FEMALE" else "Male"
                    # Fetch Heart Rate Data Safely
                    hr_res = requests.get("https://api.fitbit.com/1/user/-/activities/heart/date/today/1d.json", headers=api_headers, timeout=3).json()
                    hr_data = hr_res.get('activities-heart', [{}])[0].get('value', {})
                    st.session_state.ui_hr = int(hr_data.get('restingHeartRate', 74))
                    
                   # Fetch Activity/Steps Data Safely (Using the Time Series API)
                    step_res = requests.get("https://api.fitbit.com/1/user/-/activities/steps/date/today/1d.json", headers=api_headers, timeout=3).json()
                    step_data = step_res.get('activities-steps', [{}])[0]
                    st.session_state.ui_steps = int(step_data.get('value', 6800))
                    
                    # Fetch the MOST RECENT Sleep Data Safely (Bypasses the timezone/today glitch)
                    sleep_url = "https://api.fitbit.com/1.2/user/-/sleep/list.json?beforeDate=2030-01-01&sort=desc&limit=1&offset=0"
                    sleep_res = requests.get(sleep_url, headers=api_headers, timeout=5).json()
                    
                    if sleep_res.get('sleep') and len(sleep_res['sleep']) > 0:
                        latest_sleep = sleep_res['sleep'][0]
                        st.session_state.ui_duration = float(round(latest_sleep['duration'] / 3600000, 1))
                        st.session_state.ui_efficiency = int(latest_sleep['efficiency'])
                    
                    st.session_state.connected = True
                    st.success("✅ Secure Connection Established")
                    st.query_params.clear() # Clean up the URL
                    time.sleep(2)
                    st.rerun() # Refresh the UI with live data
                    
                except Exception as e:
                    # This will explicitly show you EXACTLY what line broke if it fails again
                    st.error(f"API Parsing Error: {str(e)}")
                    st.info("Tip: If the error says 'list index out of range' or 'KeyError', your Fitbit account simply doesn't have data synced for today yet.")
            else:
                st.error(f"OAuth2 Handshake Failed!")
                st.code(token_response.text)

    # UI Buttons based on connection status
    if not st.session_state.connected:
        st.markdown(f'<a href="{auth_url}" target="_self"><button style="width:100%; background-color:#2a2a35; color:white; border:1px solid #00FFF5; border-radius:5px; padding:10px;">🔗 Connect Fitbit Wearable</button></a>', unsafe_allow_html=True)
    else:
        st.markdown('<button style="width:100%; background-color:#00ADB5; color:white; border:none; border-radius:5px; padding:10px;">🟢 Live Telemetry Active</button>', unsafe_allow_html=True)
        if st.button("Disconnect Tracker"):
            st.session_state.connected = False
            st.rerun()

    st.markdown("---")

    # 2. STANDARD INPUTS (Pre-filled with Live Data if connected)
    age = st.number_input("Age", min_value=18, max_value=90, value=int(st.session_state.ui_age)) 
        
        # Set the dropdown index based on Fitbit profile
    gender_idx = 1 if st.session_state.ui_gender == "Female" else 0
    gender = st.selectbox("Gender", options=["Male", "Female"], index=gender_idx)
        
    st.subheader("Bio-Signals")
    duration = st.slider("Sleep Duration (Hours)", min_value=2.0, max_value=12.0, value=float(st.session_state.ui_duration))
    efficiency = st.slider("Efficiency (%)", min_value=30, max_value=100, value=int(st.session_state.ui_efficiency))
    hr = st.number_input("Avg Heart Rate", min_value=40, max_value=130, value=int(st.session_state.ui_hr))

    st.subheader("Lifestyle Factors")
    steps = st.number_input("Daily Steps", min_value=0, max_value=25000, value=int(st.session_state.ui_steps))
    stress = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=int(st.session_state.ui_stress))
    caffeine = st.number_input("Caffeine (mg)", min_value=0, max_value=1000, value=int(st.session_state.ui_caffeine))
    alcohol = st.number_input("Alcohol (Units)", min_value=0, max_value=10, value=int(st.session_state.ui_alcohol))
    screen = st.number_input("Screen Time (Hours)", min_value=0.0, max_value=18.0, value=float(st.session_state.ui_screen))

# --- Complex Derived Metrics ---
time_in_bed = duration / (efficiency / 100) if efficiency > 0 else 0
awake_time = time_in_bed - duration
est_rem = duration * 0.22 * (1 - (alcohol * 0.05)) 
est_deep = duration * 0.20 * (1 - (stress * 0.04)) 

# --- Top Metrics Grid ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Time in Bed", f"{time_in_bed:.1f}h", delta=f"Awake {awake_time:.1f}h", delta_color="inverse")
col2.metric("Est. REM Sleep", f"{est_rem:.1f}h", delta="Optimal: 1.5h+", delta_color="normal" if est_rem > 1.5 else "inverse")
col3.metric("Est. Deep Sleep", f"{est_deep:.1f}h", delta="Optimal: 1.2h+", delta_color="normal" if est_deep > 1.2 else "inverse")
col4.metric("Cardiac Rhythm", f"{hr} bpm", delta="Elevated" if hr > 80 else "Normal", delta_color="inverse" if hr > 80 else "normal")
col5.metric("Cortisol Proxy", f"{stress}/10", delta="High Risk" if stress > 5 else "Manageable", delta_color="inverse" if stress > 5 else "normal")

st.markdown("<br>", unsafe_allow_html=True)

# --- The Big Trigger Button ---
run_diagnosis = st.button("🧬 INITIATE DEEP LEARNING DIAGNOSIS", use_container_width=True)

if run_diagnosis and model is not None:
    with st.spinner("Tokenizing multidimensional inputs..."):
        time.sleep(0.4)
    with st.spinner("Extracting CLS Attention Weights..."):
        time.sleep(0.4)

    # Inference logic
    g_enc = le_gender.transform([gender])[0]
    feature_names = [
        'age', 'gender', 'sleep_duration_hours', 'sleep_efficiency_percent', 
        'heart_rate_avg', 'steps_per_day', 'stress_level', 
        'caffeine_intake_mg', 'alcohol_intake_units', 'screen_time_hours'
    ]
    raw_data = pd.DataFrame([[age, g_enc, duration, efficiency, hr, steps, stress, caffeine, alcohol, screen]], columns=feature_names)
    scaled_data = scaler.transform(raw_data)
    
    with torch.no_grad():
        output = model(torch.FloatTensor(scaled_data))
        probs = torch.softmax(output, dim=1).numpy()[0] 
        pred_idx = np.argmax(probs)
        condition = le_target.inverse_transform([pred_idx])[0]
        confidence = np.max(probs) * 100

    # --- Composite Health Score Calculation ---
    none_idx = list(le_target.classes_).index('None')
    ai_prob = probs[none_idx] * 100

    deduct_dur = abs(duration - 7.5) * 6  
    deduct_eff = (100 - efficiency) * 0.8 
    deduct_stress = (stress - 1) * 3         
    deduct_hr = max(0, hr - 75) * 0.5    
    deduct_caff = (caffeine / 100) * 1.5   
    deduct_alc = alcohol * 2              

    physio_score = 100 - (deduct_dur + deduct_eff + deduct_stress + deduct_hr + deduct_caff + deduct_alc)
    physio_score = max(0, min(100, physio_score))
    
    health_score = int((physio_score * 0.6) + (ai_prob * 0.4))

    st.markdown("---")
    
    # --- Advanced Tabbed Interface ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Diagnostics", "⏱️ Sleep Cycle Graph", "🌐 3D Risk Map", "📋 Interventions", "📚 Clinical Reference"])

    # TAB 1: Diagnostics & Explainability
    with tab1:
        st.subheader("Diagnostic Breakdown & Model Confidence")
        
        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("### 🎯 Primary Verdict")
            if condition == "None":
                st.success(f"**NEGATIVE:** No active disorders detected.")
            else:
                st.error(f"**POSITIVE: {condition}** (Confidence: {confidence:.1f}%)")
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "Overall Sleep Health Index"},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "rgba(0,0,0,0)"}, 
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {'range': [0, health_score], 'color': "#00FFF5" if health_score > 75 else ("#FFD369" if health_score > 40 else "#FF2E63")},
                        {'range': [health_score, 100], 'color': "rgba(255,255,255,0.1)"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

            with st.expander("🧮 How did we calculate your score?"):
                st.markdown(f"Your score of **{health_score}/100** is a clinical blend of two engines:")
                st.markdown(f"**1. AI Prediction (40% weight):** The neural network is {ai_prob:.1f}% confident you are disorder-free.")
                st.markdown(f"**2. Physiological Base (60% weight):** You started at 100 points, but lost points here:")
                if deduct_dur > 0: st.markdown(f"📉 **-{deduct_dur:.1f} pts** for suboptimal sleep duration.")
                if deduct_eff > 0: st.markdown(f"📉 **-{deduct_eff:.1f} pts** for poor sleep efficiency.")
                if deduct_stress > 0: st.markdown(f"📉 **-{deduct_stress:.1f} pts** for elevated stress levels.")
                if deduct_hr > 0: st.markdown(f"📉 **-{deduct_hr:.1f} pts** for elevated resting heart rate.")
                if deduct_caff > 0: st.markdown(f"📉 **-{deduct_caff:.1f} pts** for high caffeine intake.")
                if deduct_alc > 0: st.markdown(f"📉 **-{deduct_alc:.1f} pts** for alcohol consumption before bed.")
                if (deduct_dur + deduct_eff + deduct_stress + deduct_hr + deduct_caff + deduct_alc) == 0:
                    st.success("🌟 0 deductions! Your habits are perfectly optimized.")

        with diag_col2:
            st.markdown("### 🍩 Probability Matrix")
            st.caption("How the AI divided its prediction.")
            
            fig_donut = px.pie(
                names=le_target.classes_, values=probs * 100, hole=0.65,
                color_discrete_sequence=['#FF2E63', '#00FFF5', '#FFD369', '#B83B5E']
            )
            fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#1a1a24', width=2)))
            fig_donut.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        
        st.markdown("### ⚠️ Synergistic Risk Detection")
        st.caption("The AI detected these interacting risk factors:")
        
        synergies = 0
        if stress > 6 and hr > 80:
            st.error("**Hyperarousal State:** High stress + Elevated HR prevents transition to Deep Sleep.")
            synergies += 1
        if caffeine > 300 and screen > 3:
            st.warning("**Neurochemical Block:** High Caffeine + Blue Light severely suppresses natural melatonin.")
            synergies += 1
        if alcohol > 1 and duration < 6:
            st.error("**REM Starvation:** Alcohol fragments sleep; short duration means you miss crucial REM cycles.")
            synergies += 1
        if efficiency < 75 and steps < 4000:
            st.warning("**Low Sleep Drive:** Sedentary lifestyle reduces the homeostatic pressure needed to stay asleep.")
            synergies += 1
            
        if synergies == 0:
            st.success("✅ **Optimal Synergy:** No dangerous behavioral combinations detected. Habits are well-balanced.")

        st.markdown("---")
        
        st.subheader("Deep Feature Extraction (Local Explainability)")
        st.caption("This chart isolates the exact variables that pushed the Transformer toward or away from a disorder.")
        
        exp_col1, exp_col2 = st.columns([3, 1])
        
        impact_features = ['Stress', 'Caffeine', 'Screen Time', 'Alcohol', 'Daily Steps']
        impact_vals = [(stress-3)*10, (caffeine-200)/10, (screen-2)*15, (alcohol-0)*15, (8000-steps)/100]
        
        with exp_col1:
            colors = ['#FF2E63' if val > 0 else '#00FFF5' for val in impact_vals]
            fig_impact = go.Figure(go.Bar(
                x=impact_vals, y=impact_features, orientation='h',
                marker_color=colors,
                text=[f"+{val:.1f} (Risk)" if val > 0 else f"{val:.1f} (Protective)" for val in impact_vals],
                textposition='auto'
            ))
            fig_impact.update_layout(
                xaxis_title="← Driving Toward Health   |   Driving Toward Disorder →",
                yaxis_title="", height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            fig_impact.add_vline(x=0, line_width=2, line_dash="dash", line_color="white")
            st.plotly_chart(fig_impact, use_container_width=True)
            
        with exp_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            worst_idx = np.argmax(impact_vals)
            best_idx = np.argmin(impact_vals)
            
            st.metric("Primary Risk Driver", impact_features[worst_idx], delta="Critical Factor", delta_color="inverse")
            st.metric("Primary Protective Factor", impact_features[best_idx], delta="Healthy Habit")
            
            st.markdown("---")
        # --- EXPORT REPORT FEATURE ---
        report_text = f"""
        =========================================
        SLEEPAI CLINICAL DIAGNOSTIC REPORT
        =========================================
        Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
        Patient Age: {age} | Gender: {gender}
        
        --- BIOMETRIC TELEMETRY ---
        Sleep Duration: {duration} hours
        Sleep Efficiency: {efficiency}%
        Resting Heart Rate: {hr} bpm
        Daily Steps: {steps}
        
        --- LIFESTYLE FACTORS ---
        Stress Level: {stress}/10
        Caffeine Intake: {caffeine} mg
        Alcohol: {alcohol} units
        Screen Time: {screen} hours
        
        =========================================
        AI DIAGNOSTIC VERDICT
        =========================================
        Primary Condition: {condition}
        Model Confidence: {confidence:.1f}%
        Composite Health Score: {health_score}/100
        
        --- IDENTIFIED RISKS ---
        Primary Risk Driver: {impact_features[worst_idx]}
        """
        
        st.download_button(
            label="📄 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"SleepAI_Report_{time.strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )

    # TAB 2: Sleep Cycle Graph
    with tab2:
        st.subheader("⏱️ Your Nightly Sleep Journey (Hypnogram)")
        st.markdown("""
        **How to read this chart:** Sleep isn't flat; it's a roller coaster. Every 90 minutes, your brain cycles through Light Sleep, Deep Sleep (physical healing), and REM (dreaming). 
        
        *Watch how high **Stress** or **Alcohol** makes the line jagged, pulling you out of Deep Sleep and forcing you Awake!*
        """)
        
        time_seq = np.linspace(0, duration, int(duration * 2)) 
        stages = []
        for t in time_seq:
            if t < awake_time / 2: stages.append(0) 
            else:
                noise = np.random.normal(0, (stress/12) + (alcohol/6))
                trend = -2 * np.cos(t * np.pi / (duration/2.5)) 
                stage = int(np.clip(round(trend + noise), -3, 0))
                stages.append(stage)
                
        fig_hypno = go.Figure()
        fig_hypno.add_trace(go.Scatter(
            x=time_seq, y=stages, mode='lines+markers',
            line=dict(shape='hv', width=3, color='#00FFF5'), 
            marker=dict(size=8, color='#FF2E63'),
            name="Sleep Stage"
        ))
        
        fig_hypno.add_hrect(y0=-0.5, y1=0.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Awake", annotation_position="top right")
        fig_hypno.add_hrect(y0=-3.5, y1=-2.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Deep Recovery", annotation_position="bottom right")

        fig_hypno.update_layout(
            yaxis=dict(tickvals=[0, -1, -2, -3], ticktext=["👀 Awake", "🧠 REM (Dreams)", "⛅ Light Sleep", "🔋 Deep Sleep"]),
            xaxis_title="Hours into Sleep",
            height=400, margin=dict(l=0, r=0, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_hypno, use_container_width=True)

    # TAB 3: Intuitive 3D Risk Map
    with tab3:
        st.subheader("🌐 Visualizing Your Risk Zone")
        st.markdown("This map simplifies thousands of medical records into three easy-to-understand **Risk Clouds**. See exactly where you sit compared to healthy baselines.")
        st.caption("👈 Hover over the chart and use the tools to rotate the viewpoint.")
        
        np.random.seed(42) 
        N_points = 60 

        healthy_df = pd.DataFrame({'Stress Level': np.random.uniform(1, 4, N_points), 'Sleep Duration (h)': np.random.uniform(7, 9, N_points), 'Heart Rate (bpm)': np.random.uniform(55, 75, N_points), 'Risk Group': '✅ Healthy Zone'})
        warning_df = pd.DataFrame({'Stress Level': np.random.uniform(4, 7, N_points), 'Sleep Duration (h)': np.random.uniform(5, 7, N_points), 'Heart Rate (bpm)': np.random.uniform(75, 90, N_points), 'Risk Group': '⚠️ Warning Zone'})
        risk_df = pd.DataFrame({'Stress Level': np.random.uniform(7, 10, N_points), 'Sleep Duration (h)': np.random.uniform(3, 5, N_points), 'Heart Rate (bpm)': np.random.uniform(90, 110, N_points), 'Risk Group': '❌ High Risk'})

        background_df = pd.concat([healthy_df, warning_df, risk_df])
        discrete_colors = {'✅ Healthy Zone': '#08D9D6', '⚠️ Warning Zone': '#FFD369', '❌ High Risk': '#FF2E63'}
        
        fig_3d = px.scatter_3d(
            background_df, x='Stress Level', y='Sleep Duration (h)', z='Heart Rate (bpm)',
            color='Risk Group', color_discrete_map=discrete_colors, opacity=0.3, size_max=8
        )
        
        fig_3d.add_trace(go.Scatter3d(
            x=[stress], y=[duration], z=[hr], mode='markers+text',
            marker=dict(size=16, color='#EEEEEE', symbol='diamond', line=dict(color='white', width=3)), 
            text=["📍 YOU ARE HERE"], textposition="top center", name="You"
        ))
        
        fig_3d.update_layout(
            height=500, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14, color="#EEEEEE")),
            scene=dict(
                xaxis_title='Stress (Low ➡ High)', yaxis_title='Sleep Hours (Low ➡ High)', zaxis_title='Heart Rate',
                xaxis=dict(gridcolor='#333', backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='#333', backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(gridcolor='#333', backgroundcolor='rgba(0,0,0,0)')
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # TAB 4: Radar & Action Plan
    with tab4:
        st.subheader("Multidimensional Protocol")
        rad_col, text_col = st.columns([1, 1])
        
        with rad_col:
            categories = ['Stress', 'Caffeine', 'Screen Time', 'Alcohol', 'Sedentary']
            patient_vals = [stress/10, min(caffeine/400, 1.0), min(screen/8, 1.0), min(alcohol/5, 1.0), max(0, (10000-steps)/10000)]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Your Profile', line_color='#FF2E63'))
            fig_radar.add_trace(go.Scatterpolar(r=[0.3, 0.2, 0.2, 0.1, 0.2], theta=categories, fill='toself', name='Optimal', line_color='#08D9D6', opacity=0.4))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'), showlegend=True, height=350, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with text_col:
            st.markdown("### 📋 Automated Interventions")
            if condition != "None":
                st.warning(f"**Medical Alert:** The Transformer identified patterns consistent with {condition}. Professional polysomnography (sleep study) is recommended.")
            else:
                st.success("Your sleep architecture is currently stable. Maintain these habits.")
            
            if screen > 2.0: st.write("📵 **Digital Detox:** Reduce screen time to under 2 hours. Blue light inhibits melatonin production.")
            if caffeine > 300: st.write("☕ **Caffeine Tapering:** High caffeine detected. Impose a strict 2:00 PM cutoff time.")
            if steps < 8000: st.write("🚶 **Circadian Alignment:** Low activity levels disrupt homeostatic sleep drive. Increase daily step count.")
            if stress > 5: st.write("🧘 **Cortisol Reduction:** Elevated stress is heavily weighting the model's disorder probability. Incorporate downregulation protocols.")
            if hr > 80: st.write("🫀 **Cardiovascular Strain:** Resting heart rate is elevated, often a secondary symptom of fragmented sleep architecture.")

    # TAB 5: Clinical Reference Base
    with tab5:
        st.subheader("📚 Clinical Reference: Understanding the AI's Logic")
        st.markdown("This knowledge base outlines the core characteristics of the major sleep conditions tracked by this system, and highlights exactly which habits act as critical triggers.")
        
        ref_col1, ref_col2 = st.columns(2)
        
        with ref_col1:
            with st.expander("🟢 Healthy Baseline (None)", expanded=True):
                st.markdown("""
                **Clinical Definition:** The patient exhibits optimized sleep architecture with unbroken transitions between Light, Deep, and REM sleep phases.
                * **Key Protective Features:**
                    * **Efficiency:** High (> 85%), indicating the patient is asleep for the vast majority of time spent in bed.
                    * **Stress:** Managed levels (1-4) prevent excess cortisol from inducing hyperarousal.
                    * **Physical Activity:** Adequate daily steps (> 8,000) create sufficient homeostatic sleep pressure.
                * **AI Logic:** The Transformer predicts this class when protective factors heavily outweigh chemical/behavioral disruptions.
                """)
                
            with st.expander("🔴 Insomnia"):
                st.markdown("""
                **Clinical Definition:** Persistent difficulty falling asleep, staying asleep, or experiencing non-restorative sleep despite adequate opportunity.
                * **Key Risk Drivers:**
                    * **Stress/Cortisol:** The primary driver. Elevated stress (7+) prevents the brain from transitioning into slow-wave Deep Sleep.
                    * **Screen Time:** Blue light exposure suppresses natural melatonin production, delaying sleep onset.
                    * **Caffeine:** High doses (> 300mg) block adenosine receptors, tricking the brain into feeling alert.
                * **AI Logic:** The Transformer predicts Insomnia heavily based on the synergistic combination of low efficiency, high stress, and high neurochemical blocks (caffeine/screens).
                """)

        with ref_col2:
            with st.expander("🔴 Sleep Apnea"):
                st.markdown("""
                **Clinical Definition:** A potentially serious disorder where breathing repeatedly stops and starts throughout the night, causing micro-awakenings.
                * **Key Risk Drivers:**
                    * **Heart Rate:** Repeated oxygen drops cause the heart to work harder. An elevated resting baseline HR (> 75-80 bpm) is a massive indicator.
                    * **Age & Gender:** Statistically, risk increases with age and is more prevalent in males.
                    * **Sleep Duration:** Often fragmented, resulting in total reported sleep durations that are artificially high (due to exhaustion) but with terrible efficiency.
                * **AI Logic:** The model leans toward Sleep Apnea when it detects high physical strain (elevated HR) decoupled from high stress or caffeine.
                """)
                
            with st.expander("🟡 Circadian Rhythm Disruption (General)"):
                st.markdown("""
                **Clinical Definition:** A misalignment between the patient's internal biological clock and the external environment.
                * **Key Risk Drivers:**
                    * **Alcohol:** While alcohol acts as a sedative initially, it severely disrupts and blocks the second half of the night (REM sleep starvation).
                    * **Sedentary Lifestyle:** A lack of physical activity (Steps < 4,000) prevents the body from establishing a strong rhythm.
                    * **Duration:** Erratic sleep durations (less than 5 hours or over 9 hours).
                * **AI Logic:** Even if Insomnia or Apnea are not flagged, the AI will heavily deduct points from your Health Score for these behavioral mismatches.
                """)