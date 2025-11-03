import streamlit as st
import pandas as pd
import joblib

st.header("🔮 Battery Capacity Prediction")

# --- Load ML model ---
@st.cache_resource
def load_ml_model():
    return joblib.load("models (2).pkl")

rf = load_ml_model()

st.markdown("Use the sliders below to test how charging behaviour affects **remaining battery capacity**:")

# --- Input form ---
colA, colB = st.columns(2)
with colA:
    duration = st.slider("Charging Duration (hours)", 0.0, 1.0, 0.3)
    current = st.number_input("Mean Charging Current (mA)", 100, 1000, 400)
    temperature = st.slider("Mean Temperature (°C)", 20, 50, 32)
with colB:
    energy = st.number_input("Total Energy Consumed (kWh)", 0.0, 0.1, 0.02)
    cycle = st.slider("Cycle Progress (0–1)", 0.0, 1.0, 0.4)

# --- Create test case ---
test_case = pd.DataFrame({
    'Charging_Duration': [duration],
    'Mean_Charging_Current': [current],
    'Mean_Temperature': [temperature],
    'Total_Energy_Consumed': [energy],
    'Cycle_Progress': [cycle]
})

# --- Predict button ---
if st.button("🔍 Predict Remaining Capacity"):
    try:
        prediction = rf.predict(test_case)
        st.success(f"**Predicted Remaining Capacity:** {prediction[0]:.2f}%")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
