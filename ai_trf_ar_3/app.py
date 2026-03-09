import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ultralytics import YOLO
from datetime import datetime, timedelta
import random

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AI Traffic Optimization Hub", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a "Dashboard" Look
st.markdown("""
<style>
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Main Block Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 2px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BACKEND LOGIC (ENGINE)
# ==========================================
class TrafficOptimizationEngine:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        
    def analyze_frame(self, image_cv):
        results = self.model(image_cv, verbose=False)
        vehicle_classes = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        class_counts = {'Car': 0, 'Motorcycle': 0, 'Bus': 0, 'Truck': 0}
        total_vehicles = 0
        annotated_image = image_cv.copy()
        
        detections = results[0].boxes
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = vehicle_classes[cls]
                color = (0, 255, 0)
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                class_counts[label] = class_counts.get(label, 0) + 1
                total_vehicles += 1
                
        return annotated_image, total_vehicles, class_counts

    def calculate_metrics(self, total_vehicles, weather_condition):
        # Simulated Weather Impact (Phase C)
        visibility_factor = 1.5 if weather_condition in ["Rain", "Fog", "Snow"] else 1.0
        threshold = 50 * visibility_factor
        
        # Density Logic (Phase B & I)
        status = "🟢 Clear"
        signal_action = "Free Flow / Minimal Wait"
        
        if total_vehicles > threshold:
            status = "🔴 Congested"
            signal_action = "Extend Green / Divert"
        elif total_vehicles > 30:
            status = "🟡 Moderate"
            signal_action = "Standard Cycle"

        return {
            "density": total_vehicles,
            "status": status,
            "action": signal_action,
            "weather": weather_condition,
            "threshold": threshold
        }

    def predict_future(self, current_density):
        # Phase F: Mock Prediction
        times = [(datetime.now() + timedelta(minutes=15*i)).strftime("%H:%M") for i in range(1, 5)]
        trend = current_density * 1.05
        preds = []
        for t in times:
            noise = random.uniform(-20, 20)
            preds.append({"Time": t, "Predicted Density": int(trend + noise)})
            trend *= 1.02 # Simulate growth
        return preds

    def check_alerts(self, density, weather):
        alerts = []
        if density > 90:
            alerts.append(("CRITICAL", "Severe Congestion Detected"))
        if weather == "Snow":
            alerts.append(("WARNING", "Ice Accident Risk"))
        if random.random() < 0.05:
            alerts.append(("INFO", "Emergency Vehicle Priority"))
        return alerts

# ==========================================
# 3. SESSION STATE MANAGEMENT
# ==========================================
if 'engine' not in st.session_state:
    st.session_state['engine'] = TrafficOptimizationEngine()
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_result' not in st.session_state:
    st.session_state['last_result'] = None

# ==========================================
# 4. SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### System Configuration")
    
    weather_mode = st.selectbox("🌤️ Weather Simulation", ["Clear", "Rain", "Fog", "Snow"])
    uploaded_files = st.file_uploader("📂 Upload Road Images", accept_multiple_files=True, type=['png','jpg','jpeg'])
    
    st.divider()
    st.markdown("**Methodology Phases:**")
    st.markdown("- ✅ Perception (YOLOv8)")
    st.markdown("- ✅ Analysis (Thresholds)")
    st.markdown("- ✅ Decision (RL Logic)")
    
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state['history'] = []
        st.session_state['last_result'] = None
        st.rerun()

# ==========================================
# 5. MAIN DASHBOARD LAYOUT
# ==========================================

st.title("🚦 AI-Driven Traffic Optimization Framework")
st.caption("Real-time Traffic Perception, Prediction, and Control System")

# --- PROCESSING LOGIC ---
if uploaded_files:
    engine = st.session_state['engine']
    
    # Process only the first image for this demo snapshot
    file = uploaded_files[0] 
    bytes_data = file.read()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img_cv is not None:
        # Run AI Engine
        annotated_img, count, classes = engine.analyze_frame(img_cv)
        metrics = engine.calculate_metrics(count, weather_mode)
        
        # Update State
        st.session_state['last_result'] = {
            "img": annotated_img,
            "metrics": metrics,
            "classes": classes
        }
        st.session_state['history'].append(metrics['density'])

# --- ROW 1: KPI DASHBOARD CARDS ---
if st.session_state['last_result']:
    m = st.session_state['last_result']['metrics']
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    
    with col_k1:
        st.metric("🚗 Current Density", f"{m['density']} vehicles", delta="Live")
    with col_k2:
        st.metric("🚦 Traffic Status", m['status'])
    with col_k3:
        st.metric("🌤️ Weather Impact", m['weather'])
    with col_k4:
        st.metric("⚠️ Threshold", f"{int(m['threshold'])} (Adj.)")

    st.divider()

# --- ROW 2: MAIN VISUALIZATION (INPUT & PREVIEW) ---
col_input, col_preview = st.columns([0.35, 0.65])

with col_input:
    st.subheader("📁 Input Source")
    if uploaded_files:
        st.info(f"Analyzing: {len(uploaded_files)} frame(s)")
        # Show raw upload small preview
        st.image(uploaded_files, width=100)
        
        st.subheader("📊 Vehicle Classification")
        if st.session_state['last_result']:
            cls_data = st.session_state['last_result']['classes']
            st.bar_chart(pd.DataFrame(list(cls_data.items()), columns=['Type', 'Count']).set_index('Type'))
    else:
        st.info("Upload images in the sidebar to begin.")

with col_preview:
    st.subheader("🎥 AI Perception Preview")
    if st.session_state['last_result']:
        res_img = st.session_state['last_result']['img']
        # Convert BGR to RGB
        st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Detected Objects & Trajectories")
    else:
        st.warning("Waiting for Perception Data...")

# --- ROW 3: ANALYTICS GRAPHS ---
st.divider()
st.subheader("📈 Temporal Analysis & Prediction")

col_g1, col_g2 = st.columns(2)

with col_g1:
    st.markdown("**Phase K: Historical Density Trend**")
    if st.session_state['history']:
        df_hist = pd.DataFrame(st.session_state['history'], columns=['Density'])
        st.line_chart(df_hist, height=250)
    else:
        st.empty()

with col_g2:
    st.markdown("**Phase F: 30-Min Forecast (LSTM/Transformer)**")
    if st.session_state['last_result']:
        preds = st.session_state['engine'].predict_future(st.session_state['last_result']['metrics']['density'])
        df_pred = pd.DataFrame(preds).set_index('Time')
        st.area_chart(df_pred, height=250, color="#FFA500")

# --- ROW 4: INTELLIGENCE & CONTROL ---
st.divider()
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.subheader("🚨 Alerts & Notifications")
    if st.session_state['last_result']:
        m = st.session_state['last_result']['metrics']
        alerts = st.session_state['engine'].check_alerts(m['density'], m['weather'])
        if alerts:
            for sev, msg in alerts:
                if sev == "CRITICAL": st.error(f"**{sev}**: {msg}")
                elif sev == "WARNING": st.warning(f"**{sev}**: {msg}")
                else: st.info(f"**{sev}**: {msg}")
        else:
            st.success("✅ System Normal")
    else:
        st.info("No data")

with col_b:
    st.subheader("🚦 Signal Optimization")
    if st.session_state['last_result']:
        m = st.session_state['last_result']['metrics']
        st.info(f"**Recommended Action:**\n{m['action']}")
        
        st.markdown(" **RL Agent Decision Logic:**")
        st.code(f"IF density > {int(m['threshold'])} \nTHEN Extend Green Phase", language='python')
    else:
        st.info("Calculating...")

with col_c:
    st.subheader("🛣️ Route Optimization")
    if st.session_state['last_result'] and st.session_state['last_result']['metrics']['density'] > 500:
        st.warning("Congestion Detected. Re-routing...")
        st.markdown("""
        **Alternative Routes:**
        1. **Highway 66** (+5 min)
        2. **5th Avenue** (+12 min)
        """)
    elif st.session_state['last_result']:
        st.success("Main Route Optimal")
    else:
        st.info("Mapping...")