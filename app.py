import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="AV LiDAR Perception Sentinel",
    page_icon="🛡️",
    layout="wide"
)

# Dark cockpit aesthetics
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
# HEADER & SCENARIO OVERVIEW
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Interactive 3D Cyber-Physical Attack & Defense Simulator")

with st.expander("📖 Interactive Instructions (How to test vehicle reactions)", expanded=False):
    st.markdown("""
    1. Select **Scenario B: Roadside Laser Attack Sequence**.
    2. Choose an **AV Defensive Reaction** in the sidebar:
       * **Unprotected (Aggressive Panic Stop):** Watch the car physically lock its wheels and stop at 0 km/h on the highway!
       * **Evasive Lane Change:** Watch the car physically steer into the right lane around the phantom!
       * **Suppressive Braking (Passive Cruise):** The ML firewall ignores the fake obstacle and drives straight through safely.
    3. Click **'▶ Play Cruise'** under the 3D window to run the animated sequence.
    """)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🕹️ Simulation Setup")

scenario = st.sidebar.radio(
    "Choose Highway Mode:",
    ["🟢 Scenario A: Normal Highway Cruising (Clean)",
     "🔴 Scenario B: Roadside Laser Attack Sequence"]
)
is_attack = "Scenario B" in scenario

cam_view = st.sidebar.selectbox(
    "Camera Perspective",
    ["Chase Cam (Behind Car)", "Cockpit / Dashcam", "Bird's-Eye (Overhead)"]
)

st.sidebar.divider()
st.sidebar.header("⚙️ Vehicle Dynamics")
target_speed = st.sidebar.slider("Cruising Speed (km/h)", 50, 130, 95, 5)

if is_attack:
    st.sidebar.divider()
    st.sidebar.header("🛡️ AV Defense & Reaction Policy")
    defense_reaction = st.sidebar.selectbox(
        "Reaction to Perceived Obstacle:",
        ["1. Unprotected (Aggressive Panic Stop)",
         "2. ML Defensive Lane Change (Evasive Maneuver)",
         "3. ML Suppressive Braking (Passive Cruise - Safe)"]
    )
    laser_power = st.sidebar.slider("Attacker Laser Intensity", 0.60, 1.00, 0.94, 0.02)
    attack_y_dist = st.sidebar.slider("Phantom Distance (m ahead)", 8, 20, 12, 1)
else:
    defense_reaction = "Normal Cruising"
    laser_power = 0.35
    attack_y_dist = 12

# ---------------------------------------------------------
# REALISTIC 3D VEHICLE MESH BUILDER
# ---------------------------------------------------------
def create_cuboid_mesh(center, size, color, opacity=1.0, name="Part"):
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
    cx, cy, cz = center
    opacity = 0.25 if is_ghost else 0.95
    traces = []
    
    # 1. Lower Chassis
    traces.append(create_cuboid_mesh([cx, cy, cz], [1.9, 4.3, 0.65], body_color, opacity, "Chassis"))
    
    # 2. Tapered Cabin & Windshield
    cabin_color = "#334155" if not is_ghost else body_color
    traces.append(create_cuboid_mesh([cx, cy - 0.2, cz + 0.6], [1.5, 2.3, 0.55], cabin_color, opacity, "Cabin"))
    
    # 3. Headlights, Taillights & Wheels
    if not is_ghost:
        traces.append(create_cuboid_mesh([cx - 0.65, cy + 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#fef08a", 1.0, "Headlight"))
        traces.append(create_cuboid_mesh([cx + 0.65, cy + 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#fef08a", 1.0, "Headlight"))
        traces.append(create_cuboid_mesh([cx - 0.65, cy - 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#dc2626", 1.0, "Taillight"))
        traces.append(create_cuboid_mesh([cx + 0.65, cy - 2.15, cz + 0.1], [0.35, 0.1, 0.2], "#dc2626", 1.0, "Taillight"))
        
        wheel_offsets = [(-0.95, -1.2), (0.95, -1.2), (-0.95, 1.2), (0.95, 1.2)]
        for wx, wy in wheel_offsets:
            traces.append(create_cuboid_mesh([cx + wx, cy + wy, cz - 0.25], [0.3, 0.7, 0.5], "#0f172a", 1.0, "Wheel"))
            
    # 4. Roof LiDAR on Ego car
    if is_ego:
        traces.append(create_cuboid_mesh([cx, cy - 0.2, cz + 0.95], [0.4, 0.4, 0.25], "#e2e8f0", 1.0, "Roof LiDAR"))
        
    return traces

# ---------------------------------------------------------
# INTERACTIVE ANIMATION TIMELINE GENERATOR
# ---------------------------------------------------------
num_frames = 16
frames = []
road_len = 65

attacker_x, attacker_y, attacker_z = -6.0, 16.0, -0.6

# Trajectory computation based on chosen reaction
ego_x_coords = []
ego_y_coords = []

current_y = 1.0
current_x = 0.0

for step in range(num_frames):
    attack_active = is_attack and (step >= 5)
    
    if not attack_active:
        # Standard cruising progression
        current_y = 1.0 + (step / (num_frames - 1)) * 14.0
        current_x = 0.0
    else:
        # What happens after laser hits at step >= 5:
        if "Aggressive Panic Stop" in defense_reaction:
            # FREEZE VEHICLE IN PLACE (sudden stop at step 5 position)
            current_y = ego_y_coords[4] 
            current_x = 0.0
        elif "Evasive Lane Change" in defense_reaction:
            # Shift smoothly into the right lane (X goes from 0.0 to 2.8)
            lane_shift_t = min(1.0, (step - 5) / 4.0)
            current_x = lane_shift_t * 2.8
            current_y = 1.0 + (step / (num_frames - 1)) * 14.0
        else:
            # Suppressive Braking: Continue straight through safely
            current_y = 1.0 + (step / (num_frames - 1)) * 14.0
            current_x = 0.0
            
    ego_x_coords.append(current_x)
    ego_y_coords.append(current_y)

for step in range(num_frames):
    t = step / (num_frames - 1)
    ego_x = ego_x_coords[step]
    ego_y = ego_y_coords[step]
    lead_y = 26.0 + t * 14.0
    
    frame_traces = []
    
    # 1. Road Surface (2 Lanes)
    frame_traces.append(go.Mesh3d(
        x=[-6.5, 6.5, 6.5, -6.5], y=[-2, -2, road_len, road_len],
        z=[-1.73, -1.73, -1.73, -1.73],
        i=[0, 0], j=[1, 2], k=[2, 3], color="#0f172a", opacity=0.95, name="Highway"
    ))
    
    # 2. Road Markings (scroll only if vehicle is moving)
    is_stopped = is_attack and ("Aggressive Panic Stop" in defense_reaction) and (step >= 5)
    lane_offset = 0 if is_stopped else (step * 2.2) % 6.0
    lane_x, lane_y, lane_z = [], [], []
    for ly_base in np.arange(-2 + lane_offset, road_len, 6.0):
        lane_x.extend([0, 0, None])
        lane_y.extend([ly_base, ly_base + 3.0, None])
        lane_z.extend([-1.70, -1.70, None])
    frame_traces.append(go.Scatter3d(
        x=lane_x, y=lane_y, z=lane_z, mode='lines',
        line=dict(color="#f8fafc", width=5), name="Lane Stripes", showlegend=False
    ))
    
    # 3. Ego Vehicle (Blue AV)
    frame_traces.extend(build_car_model_traces([ego_x, ego_y, -0.6], "#38bdf8", is_ego=True))
    
    # 4. Lead Car (Green Car ahead)
    frame_traces.extend(build_car_model_traces([0.0, lead_y, -0.6], "#22c55e", is_ego=False))
    
    # 5. Attacker Rig on Sidewalk
    if is_attack:
        frame_traces.append(create_cuboid_mesh(
            [attacker_x, attacker_y, attacker_z], [0.8, 1.2, 1.2], "#475569", 0.9, "Roadside Laser Rig"
        ))
        
    # 6. DYNAMIC ATTACK ACTIVATION (step >= 5)
    laser_active = is_attack and (step >= 5)
    if laser_active:
        phantom_y = ego_y_coords[4] + attack_y_dist  # Spawn relative to attack point
        
        # Red Laser Beam
        frame_traces.append(go.Scatter3d(
            x=[attacker_x, ego_x],
            y=[attacker_y, ego_y - 0.2],
            z=[attacker_z + 0.4, -0.6 + 0.95],
            mode='lines+markers',
            line=dict(color="#ff0000", width=8, dash='solid'),
            marker=dict(size=4, color="#ff0000"),
            name="⚡ Laser Beam"
        ))
        
        # Red Phantom Ghost Car
        frame_traces.extend(build_car_model_traces([0.0, phantom_y, -0.6], "#ef4444", is_ghost=True))
        
    frames.append(go.Frame(data=frame_traces, name=f"step_{step}"))

cam_angles = {
    "Chase Cam (Behind Car)": dict(eye=dict(x=0.0, y=-2.4, z=1.4)),
    "Cockpit / Dashcam": dict(eye=dict(x=0.0, y=-0.4, z=0.6)),
    "Bird's-Eye (Overhead)": dict(eye=dict(x=0.0, y=-0.1, z=3.2))
}

fig_anim = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        scene=dict(
            xaxis=dict(title='Lateral X', range=[-8, 8]),
            yaxis=dict(title='Forward Y', range=[-2, road_len]),
            zaxis=dict(title='Elevation Z', range=[-2, 5]),
            aspectmode='manual', aspectratio=dict(x=1, y=3.2, z=0.8),
            camera=cam_angles[cam_view]
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=580,
        paper_bgcolor='#0b0f19',
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 220, "redraw": True}, "fromcurrent": True}], "label": "▶ Play Cruise", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": "⏸ Pause", "method": "animate"}
            ],
            "direction": "left", "pad": {"r": 10, "t": 10}, "type": "buttons",
            "x": 0.05, "y": 0.05, "bgcolor": "#1e293b", "font": {"color": "#f8fafc"}
        }],
        sliders=[{
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"} for k, f in enumerate(frames)],
            "x": 0.05, "y": -0.05, "len": 0.9,
            "currentvalue": {"prefix": "Timeline Step: ", "font": {"color": "#94a3b8"}},
            "font": {"color": "#94a3b8"}
        }]
    ),
    frames=frames
)

# ---------------------------------------------------------
# ML INFERENCE & DYNAMIC COCKPIT TELEMETRY
# ---------------------------------------------------------
if is_attack:
    features = np.array([[85, float(laser_power), 0.02, -0.75, 0.06, 0.10, float(attack_y_dist)]])
    target_info = f"Adversarial Laser Injection at {attack_y_dist}m"
else:
    features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 26.0]])
    target_info = "Lead Vehicle (Green Car at 26m)"

threat_prob = model.predict_proba(features)[0][1]
is_threat = threat_prob > 0.5

# Dynamically set gauges and messages based on user's reaction choice
if is_threat:
    if "Aggressive Panic Stop" in defense_reaction:
        current_speed = 0
        brake_force = 100
        hud_badge = "🚨 AGGRESSIVE BRAKING APPLIED (PANIC STOP)"
        status_desc = "Car violently skidded to 0 km/h on highway. Extreme risk of rear-end collision from traffic behind!"
        badge_type = "error"
    elif "Evasive Lane Change" in defense_reaction:
        current_speed = target_speed - 10
        brake_force = 20
        hud_badge = "↪️ EVASIVE LANE CHANGE INITIATED"
        status_desc = "Car detected optical anomaly and executed a controlled lane change to bypass phantom obstacle."
        badge_type = "warning"
    else:
        current_speed = target_speed
        brake_force = 0
        hud_badge = "🛡️ SUPPRESSIVE BRAKING: PHANTOM IGNORED"
        status_desc = "Laser spoofing identified. Emergency braking suppressed. Cruising safely at target speed."
        badge_type = "info"
else:
    current_speed = target_speed
    brake_force = 0
    hud_badge = "✅ NORMAL ADAPTIVE CRUISE"
    status_desc = "Authentic lead car verified. Cruising with safe following distance."
    badge_type = "success"

# ---------------------------------------------------------
# DASHBOARD LAYOUT
# ---------------------------------------------------------
col_3d, col_panel = st.columns([3, 2])

with col_3d:
    st.subheader("🌐 3D Highway Driving Simulation")
    st.caption("Click **'▶ Play Cruise'** below to observe the car's dynamic response to the laser!")
    st.plotly_chart(fig_anim, use_container_width=True)

with col_panel:
    st.subheader("🏎️ Real-Time Cockpit HUD & Telemetry")
    
    # Speed & Brake Gauges
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
        title={'text': "Brake Force (%)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0.52, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ef4444" if brake_force > 50 else ("#f59e0b" if brake_force > 0 else "#22c55e")},
            'steps': [{'range': [0, 30], 'color': '#1e293b'}, {'range': [30, 70], 'color': '#334155'}, {'range': [70, 100], 'color': '#475569'}]
        }
    ))
    gauge_fig.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='#0b0f19', font={'color': "#f8fafc"})
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Reaction Alert Badge
    if badge_type == "error":
        st.error(f"**{hud_badge}**")
    elif badge_type == "warning":
        st.warning(f"**{hud_badge}**")
    elif badge_type == "info":
        st.info(f"**{hud_badge}**")
    else:
        st.success(f"**{hud_badge}**")
        
    st.caption(f"**Vehicle Action:** {status_desc}")

    st.markdown("---")
    st.markdown(f"**Cluster Evaluated:** `{target_info}`")
    st.metric("Adversarial Anomaly Score", f"{threat_prob * 100:.1f}%")

    st.markdown("#### 🔬 Real-Time Optical Physics Checks")
    p1, p2 = st.columns(2)
    p1.metric("Reflected Intensity", f"{features[0][1]:.2f}", delta="Laser Peak" if features[0][1] > 0.7 else "Diffuse Light", delta_color="inverse")
    p2.metric("Occlusion Shadow", f"{features[0][5]:.2f}", delta="Violates Physics" if features[0][5] < 0.4 else "Valid Shadow", delta_color="normal")
