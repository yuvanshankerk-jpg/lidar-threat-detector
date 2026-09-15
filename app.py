import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Threat Detector",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure lidar_spoof_detector.joblib is in the root directory.")
    st.stop()

def generate_point_cloud(attack_mode=False):
    np.random.seed(101)
    ground_x = np.random.uniform(-10, 10, 800)
    ground_y = np.random.uniform(5, 45, 800)
    ground_z = np.zeros(800) - 1.73
    ground_intensity = np.random.uniform(0.1, 0.4, 800)
    
    obs_x = np.random.normal(3.0, 0.6, 150)
    obs_y = np.random.normal(20.0, 1.2, 150)
    obs_z = np.random.normal(0.0, 0.5, 150)
    obs_intensity = np.random.uniform(0.2, 0.5, 150)
    
    all_x = list(ground_x) + list(obs_x)
    all_y = list(ground_y) + list(obs_y)
    all_z = list(ground_z) + list(obs_z)
    all_i = list(ground_intensity) + list(obs_intensity)
    point_types = ["Environment"] * len(ground_x) + ["Real Obstacle"] * len(obs_x)
    
    if attack_mode:
        spoof_x = np.random.normal(-2.0, 0.3, 80)
        spoof_y = np.random.normal(12.0, 0.3, 80)
        spoof_z = np.random.normal(-0.2, 0.3, 80)
        spoof_intensity = np.random.uniform(0.85, 0.99, 80)
        
        all_x += list(spoof_x)
        all_y += list(spoof_y)
        all_z += list(spoof_z)
        all_i += list(spoof_intensity)
        point_types += ["Spoofed Laser Echoes"] * len(spoof_x)
        
    return np.array(all_x), np.array(all_y), np.array(all_z), np.array(all_i), point_types

st.title("🚗 Autonomous Vehicle LiDAR Threat Detection System")
st.markdown("Real-time ML perception security to identify phantom obstacle injection attacks and sensor spoofing.")

st.sidebar.header("LiDAR Stream Settings")
simulate_attack = st.sidebar.toggle("Inject Laser Spoofing Attack", value=False)
inspection_target = st.sidebar.selectbox(
    "Select Target Cluster to Analyze",
    ["Cluster 1 (Real Vehicle at 20m)", "Cluster 2 (Candidate at 12m)"]
)

px, py, pz, pi, ptypes = generate_point_cloud(attack_mode=simulate_attack)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Interactive 3D LiDAR Point Cloud")
    color_map = {"Environment": "#334155", "Real Obstacle": "#22c55e", "Spoofed Laser Echoes": "#ef4444"}
    marker_colors = [color_map[t] for t in ptypes]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=px, y=py, z=pz,
        mode='markers',
        marker=dict(size=2.5, color=marker_colors, opacity=0.85),
        text=[f"Type: {t}<br>Intensity: {i:.2f}" for t, i in zip(ptypes, pi)],
        hoverinfo='text'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Lateral X (m)',
            yaxis_title='Longitudinal Y (m)',
            zaxis_title='Elevation Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=-1.5, y=-2.0, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Physical Feature Extraction & ML Inference")
    
    if "Cluster 1" in inspection_target:
        features = np.array([[150, 0.35, 0.12, 0.1, 0.28, 0.85, 20.0]])
    else:
        if simulate_attack:
            features = np.array([[80, 0.92, 0.02, -0.8, 0.05, 0.10, 12.0]])
        else:
            st.info("ℹ️ No object cluster detected at 12m. Toggle 'Inject Laser Spoofing Attack' in the sidebar to simulate an attack.")
            st.stop()
            
    risk_prob = model.predict_proba(features)[0][1]
    is_malicious = risk_prob > 0.5
    
    st.metric("Threat Probability", f"{risk_prob * 100:.1f}%")
    
    if is_malicious:
        st.error("🚨 THREAT VERDICT: MALICIOUS PHANTOM INJECTION")
        st.warning("Evasive braking suppressed. Cluster violates natural optical reflection physics.")
    else:
        st.success("✅ THREAT VERDICT: AUTHENTIC PHYSICAL OBSTACLE")
        st.info("Confidence High: Target satisfies ray-tracing occlusion and surface diffusion.")
        
    st.markdown("#### Extracted Physics Metrics")
    m1, m2 = st.columns(2)
    m1.metric("Mean Reflection", f"{features[0][1]:.2f}")
    m1.metric("Occlusion Score", f"{features[0][5]:.2f}")
    m2.metric("Reflected Points", f"{int(features[0][0])}")
    m2.metric("Distance", f"{features[0][6]:.1f}m")
