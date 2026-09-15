import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Perception Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Load trained anomaly model
@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure 'lidar_spoof_detector.joblib' is present in your repo root.")
    st.stop()

# ---------------------------------------------------------
# UI HEADER & MISSION CONTEXT
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Physical Anomaly Detection for LiDAR Laser Spoofing Attacks")

with st.expander("📖 What is this system and how does it work? (Click to read)", expanded=True):
    st.markdown("""
    **The Scenario:**
    Autonomous Vehicles (AVs) scan the road ahead in 3D using **LiDAR** pulses. If an adversary fires synchronized laser pulses at the car from the sidewalk, they can manipulate the sensor's Time-of-Flight (ToF) clock. This creates a **phantom obstacle**—fake returns that fool the perception system into believing an emergency obstacle exists ahead, triggering dangerous phantom braking at highway speeds.
    
    **What this AI does:**
    This dashboard inspects 3D spatial clusters and tests them against optical physics laws (ray occlusion shadows and surface diffusion). It determines whether an obstacle is a **real physical vehicle** or an **adversarial laser hallucination**.
    
    **How to test the system:**
    1. Look at the **Real-World 3D Scene** tab to see your vehicle and the road environment.
    2. Toggle **"Inject Laser Spoofing Attack"** in the sidebar to simulate an attacker firing a laser spoofer at the vehicle.
    3. Select **Cluster 2 (Candidate at 12m)** in the dropdown to run the ML inspection and view the verdict.
    """)

# ---------------------------------------------------------
# SYNTHETIC DATA GENERATORS
# ---------------------------------------------------------
def generate_point_cloud(attack_mode=False):
    np.random.seed(101)
    
    # Ground plane (road)
    ground_x = np.random.uniform(-7, 7, 900)
    ground_y = np.random.uniform(3, 45, 900)
    ground_z = np.zeros(900) - 1.73
    ground_intensity = np.random.uniform(0.1, 0.35, 900)
    
    # Genuine vehicle at 20m ahead
    obs_x = np.random.normal(2.5, 0.5, 160)
    obs_y = np.random.normal(20.0, 1.0, 160)
    obs_z = np.random.normal(-0.2, 0.4, 160)
    obs_intensity = np.random.uniform(0.2, 0.55, 160)
    
    all_x = list(ground_x) + list(obs_x)
    all_y = list(ground_y) + list(obs_y)
    all_z = list(ground_z) + list(obs_z)
    all_i = list(ground_intensity) + list(obs_intensity)
    point_types = ["Road Surface"] * len(ground_x) + ["Real Vehicle Echoes"] * len(obs_x)
    
    # Injected laser attack at 12m
    if attack_mode:
        spoof_x = np.random.normal(-2.0, 0.35, 90)
        spoof_y = np.random.normal(12.0, 0.35, 90)
        spoof_z = np.random.normal(-0.3, 0.3, 90)
        spoof_intensity = np.random.uniform(0.88, 0.99, 90)
        
        all_x += list(spoof_x)
        all_y += list(spoof_y)
        all_z += list(spoof_z)
        all_i += list(spoof_intensity)
        point_types += ["Spoofed Laser Returns"] * len(spoof_x)
        
    return np.array(all_x), np.array(all_y), np.array(all_z), np.array(all_i), point_types

def create_3d_box_traces(center, size, name, color, opacity=0.8):
    """Utility to build solid 3D bounding boxes representing vehicles."""
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    
    # 8 corner vertices of the cuboid
    x = [cx-dx, cx+dx, cx+dx, cx-dx, cx-dx, cx+dx, cx+dx, cx-dx]
    y = [cy-dy, cy-dy, cy+dy, cy+dy, cy-dy, cy-dy, cy+dy, cy+dy]
    z = [cz-dz, cz-dz, cz-dz, cz-dz, cz+dz, cz+dz, cz+dz, cz+dz]
    
    # 12 triangular facets
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    
    mesh = go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity,
        name=name, flatshading=True
    )
    return mesh

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🕹️ Perception Scenario Controls")

simulate_attack = st.sidebar.toggle(
    "Inject Laser Spoofing Attack",
    value=False,
    help="Simulates an attacker aiming a laser at your LiDAR transceiver from the sidewalk."
)

inspection_target = st.sidebar.selectbox(
    "Target Cluster to Classify",
    ["Cluster 1 (Genuine Vehicle at 20m)", "Cluster 2 (Candidate at 12m)"],
    help="Choose which 3D object to send to the ML physics verification model."
)

st.sidebar.divider()
st.sidebar.markdown("""
**Ground Truth Legend:**
* 🔵 **Ego Vehicle:** Your self-driving car equipped with LiDAR.
* 🟢 **Genuine Vehicle:** Solid physical vehicle ahead ($20\\text{m}$).
* 🔴 **Laser Spoofed Echo:** Optical phantom target ($12\\text{m}$).
""")

# ---------------------------------------------------------
# MAIN INTERFACE: 3D DUAL-VIEW & CLASSIFIER
# ---------------------------------------------------------
px, py, pz, pi, ptypes = generate_point_cloud(attack_mode=simulate_attack)

col_vis, col_pred = st.columns([3, 2])

with col_vis:
    tab_sim, tab_raw = st.tabs(["🚗 Real-World 3D Digital Twin", "📡 Raw LiDAR Point Cloud"])
    
    # TAB 1: REAL-WORLD SCENE
    with tab_sim:
        st.caption("Macroscopic digital-twin representation showing physical objects vs empty space.")
        fig_scene = go.Figure()
        
        # 1. Road Surface Ribbon
        fig_scene.add_trace(go.Mesh3d(
            x=[-6, 6, 6, -6],
            y=[0, 0, 50, 50],
            z=[-1.73, -1.73, -1.73, -1.73],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color="#1e293b", opacity=0.9, name="Asphalt Highway"
        ))
        
        # 2. Highway Lane Dividers
        for lane_y in range(2, 50, 6):
            fig_scene.add_trace(go.Scatter3d(
                x=[0, 0], y=[lane_y, lane_y + 3], z=[-1.70, -1.70],
                mode='lines', line=dict(color="#f8fafc", width=5),
                showlegend=False, hoverinfo='skip'
            ))
            
        # 3. Ego Vehicle (The Autonomous Car)
        fig_scene.add_trace(create_3d_box_traces(
            center=[0, 0, -0.7], size=[2.0, 4.2, 1.5],
            name="Ego Vehicle (AV)", color="#38bdf8", opacity=0.85
        ))
        
        # 4. Genuine Vehicle Ahead
        fig_scene.add_trace(create_3d_box_traces(
            center=[2.5, 20.0, -0.7], size=[2.0, 4.2, 1.5],
            name="Genuine Vehicle (Ahead)", color="#22c55e", opacity=0.85
        ))
        
        # 5. Phantom Obstacle Injection (Only visible when attack is toggled)
        if simulate_attack:
            # Rendered as a translucent red ghost wireframe to show physical emptiness
            fig_scene.add_trace(create_3d_box_traces(
                center=[-2.0, 12.0, -0.7], size=[1.8, 3.5, 1.3],
                name="Phantom Echo (Empty Space)", color="#ef4444", opacity=0.25
            ))
            
            # Laser trajectory ray from roadside
            fig_scene.add_trace(go.Scatter3d(
                x=[-6.5, -2.0], y=[8.0, 12.0], z=[-1.0, -0.7],
                mode='lines+markers',
                line=dict(color="#dc2626", width=6, dash='dash'),
                name="Adversary Laser Trajectory"
            ))
            
        fig_scene.update_layout(
            scene=dict(
                xaxis=dict(title='Lateral X (m)', range=[-8, 8]),
                yaxis=dict(title='Longitudinal Y (m)', range=[-2, 52]),
                zaxis=dict(title='Elevation Z (m)', range=[-3, 5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=3, z=0.8),
                camera=dict(eye=dict(x=-1.6, y=-2.0, z=1.6))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=580,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
        )
        st.plotly_chart(fig_scene, use_container_width=True)

    # TAB 2: RAW SENSOR POINT CLOUD
    with tab_raw:
        st.caption("Raw time-of-flight return pulses visualized by laser reflection intensity.")
        color_map = {
            "Road Surface": "#475569",
            "Real Vehicle Echoes": "#22c55e",
            "Spoofed Laser Returns": "#ef4444"
        }
        marker_colors = [color_map[t] for t in ptypes]
        
        fig_raw = go.Figure(data=[go.Scatter3d(
            x=px, y=py, z=pz,
            mode='markers',
            marker=dict(size=2.5, color=marker_colors, opacity=0.85),
            text=[f"Echo Type: {t}<br>Intensity: {i:.2f}" for t, i in zip(ptypes, pi)],
            hoverinfo='text'
        )])
        
        fig_raw.update_layout(
            scene=dict(
                xaxis_title='Lateral X (m)',
                yaxis_title='Longitudinal Y (m)',
                zaxis_title='Elevation Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=-1.5, y=-2.0, z=1.5))
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=580
        )
        st.plotly_chart(fig_raw, use_container_width=True)

# ---------------------------------------------------------
# RIGHT PANEL: MODEL PREDICTION & REASONING
# ---------------------------------------------------------
with col_pred:
    st.subheader("🧪 ML Perception Threat Analysis")
    
    if "Cluster 1" in inspection_target:
        # Physical vehicle feature vector: [num_points, mean_i, std_i, skew_i, eigen, occlusion, dist]
        features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 20.0]])
        target_name = "Cluster 1: Genuine Vehicle (20m)"
    else:
        if simulate_attack:
            # Spoofed cluster feature vector
            features = np.array([[90, 0.94, 0.02, -0.75, 0.06, 0.08, 12.0]])
            target_name = "Cluster 2: Adversarial Candidate (12m)"
        else:
            st.info("ℹ️ No object exists in the Cluster 2 coordinate space. Toggle **'Inject Laser Spoofing Attack'** in the sidebar to simulate an adversarial attack.")
            st.stop()
            
    # Run Random Forest Inference
    risk_prob = model.predict_proba(features)[0][1]
    is_threat = risk_prob > 0.5
    
    st.markdown(f"**Inspecting:** `{target_name}`")
    st.metric("Threat Anomaly Probability", f"{risk_prob * 100:.1f}%")
    
    if is_threat:
        st.error("🚨 CRITICAL ANOMALY: PHANTOM OBSTACLE DETECTED")
        st.warning("⚠️ Action: **Suppress Evasive Braking**. Maintain cruising speed and log optical intrusion alert.")
    else:
        st.success("✅ AUTHENTIC DETECTED: REAL PHYSICAL OBSTACLE")
        st.info("🛡️ Action: **Allow ADAS Trajectory Planning**. Track object in adaptive cruise control (ACC).")
        
    st.markdown("---")
    st.markdown("#### Physical Feature Invariants")
    st.caption("How the ML model separates real obstacles from laser spoofs:")
    
    col_f1, col_f2 = st.columns(2)
    col_f1.metric(
        "Laser Return Intensity",
        f"{features[0][1]:.2f}",
        delta="Abnormally High" if features[0][1] > 0.7 else "Normal Diffuse",
        delta_color="inverse"
    )
    col_f1.metric(
        "Occlusion Shadow Score",
        f"{features[0][5]:.2f}",
        delta="Violates Physics (No Shadow)" if features[0][5] < 0.4 else "Valid Shadow",
        delta_color="normal"
    )
    col_f2.metric("Point Count", f"{int(features[0][0])}")
    col_f2.metric("Longitudinal Distance", f"{features[0][6]:.1f}m")
