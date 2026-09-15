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

# Load trained Random Forest model
@st.cache_resource
def load_model():
    return joblib.load('lidar_spoof_detector.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}. Ensure 'lidar_spoof_detector.joblib' is in your repo root.")
    st.stop()

# ---------------------------------------------------------
# HEADER & SCENARIO GUIDE
# ---------------------------------------------------------
st.title("🛡️ Autonomous Vehicle Perception Sentinel")
st.markdown("### Interactive 3D Cyber-Physical Driving Simulator & Telemetry")

# ---------------------------------------------------------
# SIDEBAR: SCENARIO & PLAYBACK CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🕹️ Simulation Scenarios")

scenario = st.sidebar.radio(
    "Select Scenario:",
    ["🟢 Scenario A: Normal Highway Cruising (Clean)",
     "🔴 Scenario B: Dynamic Roadside Laser Attack"]
)
is_attack = "Scenario B" in scenario

st.sidebar.divider()
st.sidebar.header("⚙️ Vehicle Dynamics")
target_speed = st.sidebar.slider("Cruising Speed (km/h)", 40, 130, 90, 5)

if is_attack:
    st.sidebar.divider()
    st.sidebar.header("🚨 Laser Attack Parameters")
    laser_power = st.sidebar.slider("Laser Beam Intensity", 0.60, 1.00, 0.94, 0.02)
    attack_y_dist = st.sidebar.slider("Attack Trigger Point Ahead (m)", 8, 22, 14, 1)
    
    st.sidebar.divider()
    st.sidebar.header("🛡️ ADAS Safety Logic")
    enable_suppression = st.sidebar.checkbox("Enable Suppressive Braking (ML Guard)", value=True)
else:
    laser_power = 0.35
    attack_y_dist = 14
    enable_suppression = True

# ---------------------------------------------------------
# ANIMATION ENGINE GENERATOR
# ---------------------------------------------------------
def create_box_points(center, size):
    """Generates wireframe corner coordinates for animated 3D boxes."""
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    # Outer box trace lines
    bx = [cx-dx, cx+dx, cx+dx, cx-dx, cx-dx, cx-dx, cx+dx, cx+dx, cx-dx, cx-dx, None,
          cx-dx, cx-dx, None, cx+dx, cx+dx, None, cx+dx, cx+dx]
    by = [cy-dy, cy-dy, cy+dy, cy+dy, cy-dy, cy-dy, cy-dy, cy+dy, cy+dy, cy-dy, None,
          cy-dy, cy+dy, None, cy-dy, cy+dy, None, cy-dy, cy+dy]
    bz = [cz-dz, cz-dz, cz-dz, cz-dz, cz-dz, cz+dz, cz+dz, cz+dz, cz+dz, cz-dz, None,
          cz+dz, cz+dz, None, cz+dz, cz+dz, None, cz-dz, cz+dz]
    return bx, by, bz

# Generate 15 simulation timeline frames
num_frames = 15
frames = []
road_y_max = 65

for step in range(num_frames):
    t_ratio = step / (num_frames - 1)
    
    # Forward advancement of ego vehicle
    ego_y = 2.0 + t_ratio * 15.0
    lead_car_y = 24.0 + t_ratio * 15.0
    
    # 1. Ego Vehicle Box
    ex, ey, ez = create_box_points([0, ego_y, -0.7], [2.0, 4.2, 1.4])
    
    # 2. Lead Authentic Car Box
    lx, ly, lz = create_box_points([2.4, lead_car_y, -0.7], [2.0, 4.0, 1.4])
    
    # 3. Dynamic Lane Stripes (scroll backward to convey speed)
    lane_x, lane_y, lane_z = [], [], []
    offset = (step * 2.2) % 6.0
    for ly_base in np.arange(-2 + offset, road_y_max, 6.0):
        lane_x.extend([0, 0, None])
        lane_y.extend([ly_base, ly_base + 3.0, None])
        lane_z.extend([-1.70, -1.70, None])
        
    # 4. Attack elements appear halfway through simulation (step >= 6)
    attack_visible = is_attack and (step >= 6)
    if attack_visible:
        phantom_y = ego_y + attack_y_dist
        px, py, pz = create_box_points([-2.2, phantom_y, -0.7], [1.8, 3.4, 1.2])
        lx_ray = [-6.5, -2.2]
        ly_ray = [phantom_y - 3, phantom_y]
        lz_ray = [-0.8, -0.7]
    else:
        px, py, pz = [None], [None], [None]
        lx_ray, ly_ray, lz_ray = [None], [None], [None]
        
    frame_data = [
        # Trace 0: Road asphalt base
        go.Mesh3d(
            x=[-6.5, 6.5, 6.5, -6.5], y=[-2, -2, road_y_max, road_y_max],
            z=[-1.73, -1.73, -1.73, -1.73],
            i=[0, 0], j=[1, 2], k=[2, 3], color="#0f172a", opacity=0.95, name="Highway"
        ),
        # Trace 1: Moving dashed lane markings
        go.Scatter3d(
            x=lane_x, y=lane_y, z=lane_z,
            mode='lines', line=dict(color="#f8fafc", width=4), name="Lane Stripes"
        ),
        # Trace 2: Ego Vehicle (AV)
        go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode='lines', line=dict(color="#38bdf8", width=5), name="Ego Vehicle (AV)"
        ),
        # Trace 3: Genuine Traffic Car Ahead
        go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode='lines', line=dict(color="#22c55e", width=5), name="Real Lead Car"
        ),
        # Trace 4: Phantom Laser Target
        go.Scatter3d(
            x=px, y=py, z=pz,
            mode='lines', line=dict(color="#ef4444", width=4, dash='dash'), name="Phantom Echo (Fake)"
        ),
        # Trace 5: Hacker Laser Trajectory
        go.Scatter3d(
            x=lx_ray, y=ly_ray, z=lz_ray,
            mode='lines+markers', line=dict(color="#dc2626", width=6, dash='dot'), name="Active Laser Beam"
        )
    ]
    frames.append(go.Frame(data=frame_data, name=f"frame_{step}"))

# Construct Main Animated Figure
fig_anim = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        scene=dict(
            xaxis=dict(title='Lateral X (m)', range=[-8, 8]),
            yaxis=dict(title='Forward Y (m)', range=[-2, road_y_max]),
            zaxis=dict(title='Elevation Z (m)', range=[-3, 5]),
            aspectmode='manual', aspectratio=dict(x=1, y=3.2, z=0.8),
            camera=dict(eye=dict(x=-1.5, y=-2.1, z=1.6))
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=580,
        paper_bgcolor='#0b0f19',
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02, font=dict(color="#cbd5e1")),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True}],
                    "label": "▶ Play Cruise",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": "⏸ Pause",
                    "method": "animate"
                }
            ],
            "direction": "left", "pad": {"r": 10, "t": 10}, "type": "buttons",
            "x": 0.05, "y": 0.05, "showactive": True,
            "bgcolor": "#1e293b", "font": {"color": "#f8fafc"}
        }],
        sliders=[{
            "steps": [{"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                       "label": str(k), "method": "animate"} for k, f in enumerate(frames)],
            "transition": {"duration": 0},
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
    target_info = f"Adversarial Candidate at {attack_y_dist}m"
else:
    features = np.array([[160, 0.36, 0.13, 0.12, 0.31, 0.88, 22.0]])
    target_info = "Genuine Lead Vehicle at 22m"

threat_prob = model.predict_proba(features)[0][1]
is_threat = threat_prob > 0.5

# Calculate telemetry outcomes
if is_threat:
    if enable_suppression:
        current_speed = target_speed
        brake_force = 0
        action_msg = "Cruise Maintained (Braking Suppressed)"
    else:
        current_speed = 0
        brake_force = 100
        action_msg = "HAZARDOUS BRAKE ENGAGED (Unprotected Panic Stop)"
else:
    current_speed = target_speed
    brake_force = 0
    action_msg = "Cruising Normally (Adaptive Headway)"

# ---------------------------------------------------------
# DASHBOARD LAYOUT
# ---------------------------------------------------------
col_3d, col_panel = st.columns([3, 2])

with col_3d:
    st.subheader("🌐 Dynamic Highway 3D Digital Twin")
    st.caption("Click '▶ Play Cruise' below the 3D window to start highway animation.")
    st.plotly_chart(fig_anim, use_container_width=True)

with col_panel:
    st.subheader("🏎️ Real-Time Cockpit Telemetry")
    
    # Gauges for Speed & Brake Force
    gauge_fig = go.Figure()
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number", value=current_speed,
        title={'text': "Speed (km/h)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0, 0.48], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 160], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#38bdf8"},
            'steps': [
                {'range': [0, 60], 'color': '#1e293b'},
                {'range': [60, 120], 'color': '#334155'},
                {'range': [120, 160], 'color': '#475569'}
            ]
        }
    ))
    gauge_fig.add_trace(go.Indicator(
        mode="gauge+number", value=brake_force,
        title={'text': "Brake Pressure (%)", 'font': {'color': '#f8fafc', 'size': 14}},
        domain={'x': [0.52, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
            'bar': {'color': "#ef4444" if brake_force > 50 else "#22c55e"},
            'steps': [
                {'range': [0, 30], 'color': '#1e293b'},
                {'range': [30, 70], 'color': '#334155'},
                {'range': [70, 100], 'color': '#475569'}
            ]
        }
    ))
    gauge_fig.update_layout(
        height=210, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='#0b0f19', font={'color': "#f8fafc"}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Threat Breakdown
    st.markdown(f"**Target Analyzed:** `{target_info}`")
    st.metric("Adversarial Threat Probability", f"{threat_prob * 100:.1f}%")

    if is_threat:
        st.error(f"🚨 **ANOMALY DETECTED: LASER PHANTOM ATTACK**\n\n*Safety Action:* {action_msg}")
    else:
        st.success(f"✅ **ALL CLEAR: AUTHENTIC DRIVING ENVIRONMENT**\n\n*Safety Action:* {action_msg}")

    st.markdown("#### 🛡️ Active Safeguards")
    c1, c2 = st.columns(2)
    with c1:
        st.info("📡 **Optical Physics:** " + ("VIOLATION" if is_threat else "NORMAL"))
    with c2:
        st.info("🛡️ **Defense Guard:** " + ("ACTIVE" if enable_suppression else "OFF"))
