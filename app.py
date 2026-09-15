import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Perception Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Custom Cockpit Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure 'lidar_spoof_detector.joblib' is in your repo root.")
    st.stop()

# ---------------------------------------------------------
# CONTEXT & INSTRUCTIONS
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Interactive 3D Cyber-Physical Attack Simulator")

with st.expander("📖 Understand the Roles & Highway Scenario (Click to Expand)", expanded=False):
    st.markdown("""
    * **🔵 Ego Vehicle (Your Car):** The blue car in the foreground running the roof LiDAR sensor and machine learning firewall.
    * **🟢 Lead Vehicle (Real Obstacle):** The authentic green sedan ahead. It casts an authentic ray-tracing shadow.
    * **🔴 Phantom Echo (Laser Attack):** A red floating silhouette created by a roadside attacker firing synchronized pulses. It lacks physical substance.
    """)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🕹️ Scenario & Perspective")

scenario = st.sidebar.radio(
    "Select Driving Scenario:",
    ["🟢 Scenario A: Normal Highway Cruising (Clean)",
     "🔴 Scenario B: Roadside Laser Spoofing Attack"]
)
is_attack = "Scenario B" in scenario

cam_view = st.sidebar.selectbox(
    "Camera Perspective",
    ["Chase Cam (Third Person)", "Cockpit / Dashcam", "Bird's-Eye (Top-Down)"]
)

st.sidebar.divider()
st.sidebar.header("⚙️ Vehicle & Attack Tuning")
target_speed = st.sidebar.slider("Cruising Speed (km/h)", 50, 130, 95, 5)

if is_attack:
    laser_power = st.sidebar.slider("Attacker Laser Intensity", 0.60, 1.00, 0.94, 0.02)
    attack_y_dist = st.sidebar.slider("Phantom Distance (m ahead)", 8, 22, 14, 1)
    enable_suppression = st.sidebar.checkbox("Enable ML Braking Suppression", value=True)
else:
    laser_power = 0.35
    attack_y_dist = 14
    enable_suppression = True

# ---------------------------------------------------------
# REALISTIC 3D VEHICLE MESH BUILDER
# ---------------------------------------------------------
def create_cuboid_mesh(center, size, color, opacity=1.0, name="Part"):
    """Builds a closed 3D triangular mesh for chassis and cabins."""
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    x = [cx-dx, cx+dx, cx+dx, cx-dx, cx-dx, cx+dx, cx+dx, cx-dx]
    y = [cy-dy, cy-dy, cy+dy, cy+dy, cy-dy, cy-dy, cy+dy, cy+dy]
    z = [cz-dz, cz-dz, cz-dz, cz-dz, cz+dz, cz+dz, cz+dz, cz+dz]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity, name=name, flatshading=True)

def build_car_model_traces(center, body_color, is_ego=False, is_ghost=False):
    """Assembles a detailed multi-component vehicle with chassis, cabin, lights, and wheels."""
    cx, cy, cz = center
    opacity = 0.30 if is_ghost else 0.95
    traces = []
    
    # 1. Lower Chassis
    traces.append(create_cuboid_mesh([cx, cy, cz], [1.9, 4.3, 0.65], body_color, opacity, "Chassis"))
    
    # 2. Tapered Cabin & Windshield
    cabin_color = "#334155" if not is_ghost else body_color
    traces.append(create_cuboid_mesh([cx, cy - 0.2, cz + 0.6], [1.5, 2.3, 0.55], cabin_color, opacity, "Cabin"))
    
    # 3. Headlights and Taillights
    if not is_ghost:
        # Front headlights (facing +Y)
        traces.append(create_cuboid_mesh([cx - 0.65, cy + 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#fef08a", 1.0, "Headlight"))
        traces.append(create_cuboid_mesh([cx + 0.65, cy + 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#fef08a", 1.0, "Headlight"))
        # Rear taillights (facing -Y)
        traces.append(create_cuboid_mesh([cx - 0.65, cy - 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#dc2626", 1.0, "Taillight"))
        traces.append(create_cuboid_mesh([cx + 0.65, cy - 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#dc2626", 1.0, "Taillight"))
        
        # 4. Wheels (4 black corners)
        wheel_offsets = [(-0.95, -1.2), (0.95, -1.2), (-0.95, 1.2), (0.95, 1.2)]
        for wx, wy in wheel_offsets:
            traces.append(create_cuboid_mesh([cx + wx, cy + wy, cz - 0.25], [0.3, 0.7, 0.5], "#0f172a", 1.0, "Wheel"))
            
    # 5. Roof LiDAR Sensor (only on Ego vehicle)
    if is_ego:
        traces.append(create_cuboid_mesh([cx, cy - 0.2, cz + 0.95], [0.4, 0.4, 0.25], "#e2e8f0", 1.0, "Roof LiDAR"))
        
    return traces

# ---------------------------------------------------------
# ANIMATION TIMELINE GENERATION
# ---------------------------------------------------------
num_frames = 14
frames = []
road_len = 65

for step in range(num_frames):
    t = step / (num_frames - 1)
    ego_y = 2.0 + t * 14.0
    lead_y = 25.0 + t * 14.0
    
    frame_traces = []
    
    # 1. Road Surface Ribbon
    frame_traces.append(go.Mesh3d(
        x=[-6.5, 6.5, 6.5, -6.5], y=[-2, -2, road_len, road_len],
        z=[-1.73, -1.73, -1.73, -1.73],
        i=[0, 0], j=[1, 2], k=[2, 3], color="#0f172a", opacity=0.95, name="Highway"
    ))
    
    # 2. Scrolling Lane Markers
    lane_x, lane_y, lane_z = [], [], []
    offset = (step * 2.2) % 6.0
    for ly_base in np.arange(-2 + offset, road_len, 6.0):
        lane_x.extend([0, 0, None])
        lane_y.extend([ly_base, ly_base + 3.0, None])
        lane_z.extend([-1.70, -1.70, None])
    frame_traces.append(go.Scatter3d(
        x=lane_x, y=lane_y, z=lane_z, mode='lines',
        line=dict(color="#f8fafc", width=5), name="Lane Stripes", showlegend=False
    ))
    
    # 3. Ego Vehicle (Blue AV)
    frame_traces.extend(build_car_model_traces([0.0, ego_y, -0.6], "#38bdf8", is_ego=True))
    
    # 4. Lead Car (Green Authentic Vehicle)
    frame_traces.extend(build_car_model_traces([2.3, lead_y, -0.6], "#22c55e", is_ego=False))
    
    # 5. Attack Injection (Appears mid-run if attack scenario selected)
    attack_triggered = is_attack and (step >= 5)
    if attack_triggered:
        phantom_y = ego_y + attack_y_dist
        frame_traces.extend(build_car_model_traces([-2.2, phantom_y, -0.6], "#ef4444", is_ghost=True))
        # Laser ray
        frame_traces.append(go.Scatter3d(
            x=[-6.5, -2.2], y=[phantom_y - 3.5, phantom_y], z=[-0.8, -0.6],
            mode='lines+markers', line=dict(color="#dc2626", width=6, dash='dot'),
            name="Laser Beam"
        ))
        
    frames.append(go.Frame(data=frame_traces, name=f"f_{step}"))

# Camera Perspective Matrix
cam_dict = {
    "Chase Cam (Third Person)": dict(eye=dict(x=0.0, y=-2.4, z=1.4)),
    "Cockpit / Dashcam": dict(eye=dict(x=0.0, y=-0.4, z=0.6)),
    "Bird's-Eye (Top-Down)": dict(eye=dict(x=0.0, y=-0.1, z=3.2))
}

fig_anim = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        scene=dict(
            xaxis=dict(title='Lateral X', range=[-8, 8]),
            yaxis=dict(title='Forward Y', range=[-2, road_len]),
            zaxis=dict(title='Elevation Z', range=[-2, 5]),
            aspectmode='manual', aspectratio=dict(x=1, y=3.2, z=0.8),
            camera=cam_dict[cam_view]
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=580,
        paper_bgcolor='#0b0f19',
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True}], "label": "▶ Play Cruise", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 10}, "type": "buttons",
            "x": 0.05, "y": 0.05, "bgcolor": "#1e293b", "font": {"color": "#f8fafc"}
        }],
        sliders=[{
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"} for k, f in enumerate(frames)],
            "x": 0.05, "y": -0.05, "len": 0.9,
            "currentvalue": {"prefix": "Time Step: ", "font": {"color": "#94a3b8"}},
            "font": {"color": "#94a3b8"}
        }]
    ),
    frames=frames
)

# ---------------------------------------------------------
# ML INFERENCE & TELEMETRY
# ---------------------------------------------------------
if is_attack:
    features = np.array([[85, float(laser_power), 0.02, -0.75, 0.06, 0.10, float(attack_y_dist)]])
    target_info = f"Phantom Echo at {attack_y_dist}m"
else:
    features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 25.0]])
    target_info = "Lead Vehicle (Green Sedan at 25m)"

threat_prob = model.predict_proba(features)[0][1]
is_threat = threat_prob > 0.5

if is_threat:
    if enable_suppression:
        current_speed = target_speed
        brake_force = 0
        action_msg = "Cruise Maintained (Phantom Braking Suppressed)"
    else:
        current_speed = 0
        brake_force = 100
        action_msg = "HAZARDOUS BRAKE APPLIED (Unprotected)"
else:
    current_speed = target_speed
    brake_force = 0
    action_msg = "Adaptive Cruise Active (Safe Headway)"

# ---------------------------------------------------------
# DASHBOARD LAYOUT
# ---------------------------------------------------------
col_3d, col_panel = st.columns([3, 2])

with col_3d:
    st.subheader("🌐 3D Highway Driving Scene")
    st.caption("Click '▶ Play Cruise' below the canvas to start animated driving.")
    st.plotly_chart(fig_anim, use_container_width=True)

with col_panel:
    st.subheader("🏎️ Cockpit Telemetry")
    
    # Gauges for Speed & Brake Pressure
    gauge_fig = go.Figure()
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number", value=current_speed,
        title={'text': "Speed (km/h)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0, 0.48], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 160], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#38bdf8"},
            'steps': [{'range': [0, 60], 'color': '#1e293b'}, {'range': [60, 120], 'color': '#334155'}, {'range': [120, 160], 'color': '#475569'}]
        }
    ))
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number", value=brake_force,
        title={'text': "Brake Pressure (%)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0.52, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ef4444" if brake_force > 50 else "#22c55e"},
            'steps': [{'range': [0, 30], 'color': '#1e293b'}, {'range': [30, 70], 'color': '#334155'}, {'range': [70, 100], 'color': '#475569'}]
        }
    ))
    gauge_fig.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='#0b0f19', font={'color': "#f8fafc"})
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Threat Analysis Card
    st.markdown(f"**Target Inspected:** `{target_info}`")
    st.metric("Adversarial Threat Probability", f"{threat_prob * 100:.1f}%")

    if is_threat:
        st.error(f"🚨 **ANOMALY DETECTED: LASER PHANTOM ATTACK**\n\n*Action:* {action_msg}")
    else:
        st.success(f"✅ **NORMAL: AUTHENTIC LEAD VEHICLE**\n\n*Action:* {action_msg}")

    st.markdown("#### 🛡️ Optical Invariants Inspected")
    f1, f2 = st.columns(2)
    f1.metric("Reflected Intensity", f"{features[0][1]:.2f}", delta="Laser Peak" if features[0][1] > 0.7 else "Normal Surface", delta_color="inverse")
    f2.metric("Occlusion Shadow", f"{features[0][5]:.2f}", delta="Missing (Empty Space)" if features[0][5] < 0.4 else "Solid Shadow", delta_color="normal")
