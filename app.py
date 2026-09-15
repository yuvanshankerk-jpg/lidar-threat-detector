import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Perception Sentinel",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure 'lidar_spoof_detector.joblib' is present in your repo root.")
    st.stop()

# ---------------------------------------------------------
# HEADER & STEP-BY-STEP USER GUIDE
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Interactive Cybersecurity Simulator for Self-Driving Cars")

with st.expander("❓ How do I use this website? (Click to view guide)", expanded=False):
    st.markdown("""
    **What is happening here?**
    You are in the driver's seat of an autonomous vehicle. On the left, you see a 3D digital model of the highway. On the right, an AI security guard inspects every object in front of the car.
    
    **Try these 3 steps:**
    1. **Choose a Scenario** in the left sidebar (click **Normal Driving** or **Simulate Attack**).
    2. Watch the **3D Scene**: If under attack, an attacker fires a red laser from the sidewalk, creating a fake obstacle.
    3. Watch the **Speedometer & AI Verdict**: The AI checks if the obstacle is real. If it is fake, the AI keeps cruising at $100\\text{ km/h}$ instead of crashing by slamming the brakes!
    """)

# ---------------------------------------------------------
# SIDEBAR: ONE-CLICK SCENARIOS & CUSTOM TUNING
# ---------------------------------------------------------
st.sidebar.header("🕹️ Quick-Play Scenarios")

scenario = st.sidebar.radio(
    "Choose a Driving Condition:",
    ["🟢 Scenario A: Safe Highway Cruising", "🔴 Scenario B: Roadside Laser Attack Active"]
)

is_attack = "Scenario B" in scenario

st.sidebar.divider()
st.sidebar.subheader("🎛️ Attacker Custom Controls")

if is_attack:
    laser_power = st.sidebar.slider(
        "Laser Signal Intensity",
        min_value=0.50, max_value=1.00, value=0.92, step=0.02,
        help="Higher values mean the hacker fired a brighter, more concentrated laser pulse."
    )
    attack_distance = st.sidebar.slider(
        "Phantom Obstacle Distance (meters)",
        min_value=6, max_value=25, value=12, step=1,
        help="How far ahead of your car the hacker wants the fake object to appear."
    )
else:
    st.sidebar.caption("Attacker controls are disabled during Safe Cruising mode.")
    laser_power = 0.92
    attack_distance = 12

# ---------------------------------------------------------
# 3D SCENE GENERATION
# ---------------------------------------------------------
def build_vehicle_mesh(center, size, name, color, opacity=0.85):
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    x = [cx-dx, cx+dx, cx+dx, cx-dx, cx-dx, cx+dx, cx+dx, cx-dx]
    y = [cy-dy, cy-dy, cy+dy, cy+dy, cy-dy, cy-dy, cy+dy, cy+dy]
    z = [cz-dz, cz-dz, cz-dz, cz-dz, cz+dz, cz+dz, cz+dz, cz+dz]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name, flatshading=True)

# Generate Scene
fig_scene = go.Figure()

# Road Surface
fig_scene.add_trace(go.Mesh3d(
    x=[-6, 6, 6, -6], y=[0, 0, 50, 50], z=[-1.73, -1.73, -1.73, -1.73],
    i=[0, 0], j=[1, 2], k=[2, 3], color="#1e293b", opacity=0.9, name="Asphalt Highway"
))

# Lane Markers
for lane_y in range(2, 50, 6):
    fig_scene.add_trace(go.Scatter3d(
        x=[0, 0], y=[lane_y, lane_y + 3], z=[-1.70, -1.70],
        mode='lines', line=dict(color="#f8fafc", width=5), showlegend=False, hoverinfo='skip'
    ))

# Ego Vehicle (Blue)
fig_scene.add_trace(build_vehicle_mesh([0, 0, -0.7], [2.0, 4.2, 1.5], "Ego Vehicle (Your Car)", "#38bdf8", 0.9))

# Genuine Vehicle Ahead (Green)
fig_scene.add_trace(build_vehicle_mesh([2.5, 22.0, -0.7], [2.0, 4.2, 1.5], "Genuine Vehicle Ahead", "#22c55e", 0.85))

# Attack elements if active
if is_attack:
    # Phantom Ghost Box
    fig_scene.add_trace(build_vehicle_mesh(
        [-2.0, float(attack_distance), -0.7], [1.8, 3.5, 1.3],
        "Phantom Laser Hallucination", "#ef4444", 0.25
    ))
    # Laser Ray
    fig_scene.add_trace(go.Scatter3d(
        x=[-6.5, -2.0], y=[attack_distance - 4, attack_distance], z=[-0.8, -0.7],
        mode='lines+markers', line=dict(color="#dc2626", width=6, dash='dash'),
        name="Hacker Laser Beam"
    ))

fig_scene.update_layout(
    scene=dict(
        xaxis=dict(title='Lateral (X)', range=[-8, 8]),
        yaxis=dict(title='Distance (Y)', range=[-2, 52]),
        zaxis=dict(title='Elevation (Z)', range=[-3, 5]),
        aspectmode='manual', aspectratio=dict(x=1, y=3, z=0.8),
        camera=dict(eye=dict(x=-1.6, y=-2.0, z=1.6))
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=550,
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
)

# ---------------------------------------------------------
# INTERACTION & INFERENCE DISPLAY
# ---------------------------------------------------------
col_view, col_status = st.columns([3, 2])

with col_view:
    st.subheader("🚗 3D Highway Driving View")
    st.plotly_chart(fig_scene, use_container_width=True)

with col_status:
    st.subheader("🧠 AI Threat Analysis")
    
    if is_attack:
        # Features: [num_points, mean_intensity, std_intensity, skew_intensity, eigen, occlusion, dist]
        features = np.array([[85, float(laser_power), 0.02, -0.75, 0.06, 0.10, float(attack_distance)]])
        target_name = f"Suspicious Object at {attack_distance}m"
    else:
        features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 22.0]])
        target_name = "Genuine Vehicle at 22m"
        
    risk_prob = model.predict_proba(features)[0][1]
    is_threat = risk_prob > 0.5
    
    st.markdown(f"**Target Analyzed:** `{target_name}`")
    st.metric("Adversarial Threat Probability", f"{risk_prob * 100:.1f}%")
    
    if is_threat:
        st.error("🚨 ATTACK DETECTED: FAKE OBSTACLE INJECTED")
        st.markdown("""
        * **Safety Decision:** **Suppress Emergency Braking**
        * **Vehicle State:** Cruising safely at $100\\text{ km/h}$.
        * **Reasoning:** The object casts no ray-tracing shadow and reflects unnatural laser intensity.
        """)
    else:
        st.success("✅ ROAD CLEAR: OBSTACLE IS AUTHENTIC")
        st.markdown("""
        * **Safety Decision:** **Adaptive Cruise Control Active**
        * **Vehicle State:** Maintaining safe following distance.
        * **Reasoning:** Laser reflection exhibits organic material diffusion and valid physical shadows.
        """)
        
    st.markdown("---")
    st.markdown("#### Vehicle Telemetry Status")
    t1, t2 = st.columns(2)
    t1.metric("Cruising Speed", "100 km/h", delta="Stable")
    t2.metric("Emergency Brake", "OFF (Suppressed)" if is_threat else "NORMAL", delta_color="inverse")
