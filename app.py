import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Perception Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling for a modern cockpit UI
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .status-card {
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Load trained Random Forest model
@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure 'lidar_spoof_detector.joblib' is present in your repo root.")
    st.stop()

# ---------------------------------------------------------
# HEADER & SCENARIO GUIDE
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Real-Time Cyber-Physical Defense & Cockpit Telemetry")

with st.expander("📖 System Documentation: Ego Vehicle & Multi-Vector Defense", expanded=False):
    st.markdown("""
    * **What is the Ego Vehicle?** It is **your vehicle**—the autonomous car running the onboard LiDAR scanner, camera suite, and path planner.
    * **The Attack:** A roadside adversary fires pulsed lasers matched to the LiDAR's wavelength, causing the car to perceive a **Phantom Obstacle** in open air.
    * **Dual-Layer Defense:**
        1. **Suppressive Braking:** Stops the car from violently freezing at high speeds.
        2. **Sensor Fusion Cross-Check:** Correlates radar and camera vision to confirm whether an object physically exists, maintaining collision safety.
    """)

# ---------------------------------------------------------
# SIDEBAR: ADVANCED VEHICLE & ATTACK CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🕹️ Cockpit & Environment Controls")

# Speed Controller
target_speed = st.sidebar.slider(
    "Ego Vehicle Target Speed (km/h)",
    min_value=30, max_value=140, value=90, step=5,
    help="Set the cruising speed of your autonomous car."
)

# Multi-Obstacle Preset Selector
traffic_preset = st.sidebar.selectbox(
    "Highway Traffic Density",
    ["Scenario 1: Single Lead Vehicle (22m)", 
     "Scenario 2: Multi-Vehicle Convoy (3 Obstacles)", 
     "Scenario 3: User-Defined Custom Obstacles"]
)

# Custom Obstacle Slider if Scenario 3 selected
if "Scenario 3" in traffic_preset:
    num_obstacles = st.sidebar.slider("Number of Genuine Obstacles", 1, 5, 3)
elif "Scenario 2" in traffic_preset:
    num_obstacles = 3
else:
    num_obstacles = 1

st.sidebar.divider()
st.sidebar.header("🚨 Laser Attack Injection")

simulate_attack = st.sidebar.toggle(
    "Inject Roadside Laser Spoofing", 
    value=True,
    help="Activates an optical transceiver injecting fake point clusters."
)

laser_power = st.sidebar.slider("Laser Beam Intensity", 0.60, 1.00, 0.94, 0.02) if simulate_attack else 0.94
attack_dist = st.sidebar.slider("Phantom Distance (m)", 6, 25, 12, 1) if simulate_attack else 12

st.sidebar.divider()
st.sidebar.header("🛡️ ADAS Defense Policy")
enable_suppression = st.sidebar.checkbox("Enable Braking Suppression", value=True)
enable_sensor_fusion = st.sidebar.checkbox("Enable Radar/Camera Cross-Verification", value=True)

# ---------------------------------------------------------
# 3D MESH GENERATION HELPERS
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

# Generate Dynamic 3D Highway Scene
fig_scene = go.Figure()

# Asphalt Road
fig_scene.add_trace(go.Mesh3d(
    x=[-7, 7, 7, -7], y=[0, 0, 60, 60], z=[-1.73, -1.73, -1.73, -1.73],
    i=[0, 0], j=[1, 2], k=[2, 3], color="#0f172a", opacity=0.95, name="Highway"
))

# Dashed Lane Stripes
for lane_y in range(2, 60, 6):
    fig_scene.add_trace(go.Scatter3d(
        x=[0, 0], y=[lane_y, lane_y + 3], z=[-1.70, -1.70],
        mode='lines', line=dict(color="#e2e8f0", width=4), showlegend=False, hoverinfo='skip'
    ))

# Ego Vehicle (Your Car)
fig_scene.add_trace(build_vehicle_mesh([0, 0, -0.7], [2.0, 4.2, 1.5], "Ego Vehicle (Your AV)", "#38bdf8", 0.9))

# Genuine Dynamic Obstacles
vehicle_positions = [
    (2.5, 22.0), (-2.5, 34.0), (2.5, 46.0), (-2.2, 18.0), (0.0, 52.0)
]
for idx in range(min(num_obstacles, len(vehicle_positions))):
    pos = vehicle_positions[idx]
    fig_scene.add_trace(build_vehicle_mesh(
        [pos[0], pos[1], -0.7], [2.0, 4.0, 1.5], 
        f"Real Vehicle #{idx+1} ({pos[1]:.0f}m)", "#22c55e", 0.85
    ))

# Adversary Injection
if simulate_attack:
    fig_scene.add_trace(build_vehicle_mesh(
        [-2.0, float(attack_dist), -0.7], [1.8, 3.5, 1.3],
        f"Phantom Echo ({attack_dist}m)", "#ef4444", 0.25
    ))
    fig_scene.add_trace(go.Scatter3d(
        x=[-6.5, -2.0], y=[max(0, attack_dist - 4), attack_dist], z=[-0.8, -0.7],
        mode='lines+markers', line=dict(color="#dc2626", width=6, dash='dot'),
        name="Laser Injection Beam"
    ))

fig_scene.update_layout(
    scene=dict(
        xaxis=dict(title='Lateral X (m)', range=[-8, 8]),
        yaxis=dict(title='Forward Y (m)', range=[-2, 62]),
        zaxis=dict(title='Elevation Z (m)', range=[-3, 5]),
        aspectmode='manual', aspectratio=dict(x=1, y=3, z=0.8),
        camera=dict(eye=dict(x=-1.5, y=-2.2, z=1.6))
    ),
    margin=dict(l=0, r=0, b=0, t=0), height=580,
    paper_bgcolor='#0b0f19',
    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, font=dict(color="#cbd5e1"))
)

# ---------------------------------------------------------
# ML INFERENCE & TELEMETRY CALCULATIONS
# ---------------------------------------------------------
# Target Feature Vector: [num_points, mean_i, std_i, skew_i, eigen, occlusion, dist]
if simulate_attack:
    features = np.array([[85, float(laser_power), 0.02, -0.75, 0.06, 0.10, float(attack_dist)]])
    target_desc = f"Phantom Echo at {attack_dist}m"
else:
    features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 22.0]])
    target_desc = "Genuine Lead Vehicle at 22m"

threat_prob = model.predict_proba(features)[0][1]
is_threat = threat_prob > 0.5

# Calculate Telemetry Responses
if is_threat:
    if enable_suppression:
        current_speed = target_speed  # Cruise preserved
        brake_pressure = 0
        mitigation_status = "CRUISE MAINTAINED (Braking Suppressed)"
    else:
        current_speed = 0  # Full hazardous freeze
        brake_pressure = 100
        mitigation_status = "EMERGENCY BRAKE ENGAGED (Unprotected)"
else:
    current_speed = target_speed
    brake_pressure = 0
    mitigation_status = "NORMAL ADAPTIVE CRUISE"

# ---------------------------------------------------------
# DASHBOARD LAYOUT: 3D VIEW & TELEMETRY COCKPIT
# ---------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("🌐 3D Highway Environment & Digital Twin")
    st.plotly_chart(fig_scene, use_container_width=True)

with col_right:
    st.subheader("🏎️ Real-Time Vehicle Cockpit")

    # Interactive Plotly Radial Gauges for Speed & Brake Pressure
    gauge_fig = go.Figure()

    # Speedometer Gauge
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=current_speed,
        title={'text': "Speed (km/h)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0, 0.48], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 160], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#38bdf8"},
            'steps': [
                {'range': [0, 60], 'color': '#1e293b'},
                {'range': [60, 120], 'color': '#334155'},
                {'range': [120, 160], 'color': '#475569'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.75,
                'value': target_speed
            }
        }
    ))

    # Brake Pressure Gauge
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=brake_pressure,
        title={'text': "Brake Force (%)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0.52, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ef4444" if brake_pressure > 50 else "#22c55e"},
            'steps': [
                {'range': [0, 30], 'color': '#1e293b'},
                {'range': [30, 70], 'color': '#334155'},
                {'range': [70, 100], 'color': '#475569'}
            ]
        }
    ))

    gauge_fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='#0b0f19',
        font={'color': "#f8fafc"}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Status Cards & Threat Verdict
    st.markdown(f"**Cluster Inspected:** `{target_desc}`")
    st.metric("Adversarial Threat Probability", f"{threat_prob * 100:.1f}%")

    if is_threat:
        st.error(f"🚨 **ANOMALY VERDICT: LASER SPOOFING DETECTED**\n\n*Action:* {mitigation_status}")
    else:
        st.success(f"✅ **ANOMALY VERDICT: GENUINE OBSTACLE**\n\n*Action:* {mitigation_status}")

    # Defense Strategy Indicators
    st.markdown("#### 🛡️ Active Countermeasure Breakdown")
    c1, c2 = st.columns(2)
    with c1:
        if enable_suppression:
            st.info("🛡️ **Suppressive Braking:** ACTIVE (Hazardous phantom stops neutralized)")
        else:
            st.warning("⚠️ **Suppressive Braking:** DISABLED (Risk of rear-end collision)")
    with c2:
        if enable_sensor_fusion:
            st.info("📡 **Radar Fusion:** ACTIVE (Zero Doppler return detected; vision clear)")
        else:
            st.warning("⚠️ **Radar Fusion:** DISABLED (Single sensor dependency)")
