import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="ChargeWise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .main-header {
            background-color: #519748;
            padding: 1rem 2rem;
            border-radius: 10px;
            color: white;
            font-family: 'Times New Roman', serif;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
        }
    </style>

    <div class="main-header">
        <h1>🔋 ChargeWise Dashboard</h1>
    </div>
""", unsafe_allow_html=True)

# --- CACHE LOADERS ---
@st.cache_resource
def load_ml_model(path="models (2).pkl"):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}")
        return None

@st.cache_data
def load_data(file_path='feature_engineered_battery.csv'):
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {file_path}")
        return None

# Load everything
rf_model = load_ml_model()
final_df = load_data()

if final_df is None or rf_model is None:
    st.stop()

ML_FEATURES = ['Charging_Duration', 'Mean_Charging_Current', 'Mean_Temperature',
               'Total_Energy_Consumed', 'Cycle_Progress']

# ----------------------------------------------------------------
# 🧭 SIDEBAR FILTERS
# ----------------------------------------------------------------
st.sidebar.header("🔍 Data Filters")

# Device filter
device_options = final_df['Device_ID'].unique()
selected_devices = st.sidebar.multiselect("Select Device(s):", device_options, default=device_options)

# Temperature filter
min_temp, max_temp = int(final_df['Battery_Operating_Temperature'].min()), int(final_df['Battery_Operating_Temperature'].max())
selected_temp = st.sidebar.slider("Select Temperature Range (°C):", min_temp, max_temp, (min_temp, max_temp))

# Cycle Progress filter
min_cycle, max_cycle = round(final_df['Cycle_Progress'].min(), 2), round(final_df['Cycle_Progress'].max(), 2)
selected_cycle = st.sidebar.slider("Select Cycle Progress Range:", min_cycle, max_cycle, (min_cycle, max_cycle))

# Apply filters
filtered_df = final_df[
    (final_df['Device_ID'].isin(selected_devices)) &
    (final_df['Battery_Operating_Temperature'].between(*selected_temp)) &
    (final_df['Cycle_Progress'].between(*selected_cycle))
]

st.sidebar.success(f"✅ Showing {len(filtered_df)} records")

# ----------------------------------------------------------------
# 🔍 EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------
st.header("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Average Charging Duration by Device")
    avg_duration = filtered_df.groupby('Device_ID')['Charging_Duration'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_duration.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel('Duration (seconds)')
    st.pyplot(fig)

with col2:
    st.subheader("Total Energy Consumed by Device (Wh)")
    energy = filtered_df.groupby('Device_ID')['Total_Energy_Consumed'].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    energy.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_ylabel('Energy (Wh)')
    st.pyplot(fig)

# ----------------------------------------------------------------
# 🌡️ TEMPERATURE INSIGHTS
# ----------------------------------------------------------------
st.header("🌡️ Battery Health Diagnostics")

col3, col4 = st.columns(2)
with col3:
    st.subheader('Temperature vs Battery Percent')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(filtered_df['Battery_Percentage'], filtered_df['Battery_Operating_Temperature'], alpha=0.4)
    ax.set_xlabel('Battery Percentage (%)')
    ax.set_ylabel('Temperature (°C)')
    st.pyplot(fig)

with col4:
    st.subheader('Distribution of Battery Operating Temperature')
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(filtered_df['Battery_Operating_Temperature'], bins=30, kde=True, color='orange', ax=ax)
    ax.set_xlabel('Temperature (°C)')
    st.pyplot(fig)

# ----------------------------------------------------------------
# 🧠 PREDICTION SECTION
# ----------------------------------------------------------------
st.header("🔮 Predict Battery Remaining Capacity")

st.markdown("Adjust the sliders below to simulate different charging scenarios:")

# Input form for prediction
colA, colB = st.columns(2)
with colA:
    duration = st.slider("Charging Duration", 0.0, 1.0, 0.3)
    current = st.number_input("Mean Charging Current (mA)", 100, 1000, 400)
    temperature = st.slider("Mean Temperature (°C)", 20, 50, 32)
with colB:
    energy = st.number_input("Total Energy Consumed (Wh)", 0.001, 0.05, 0.01)
    cycle = st.slider("Cycle Progress", 0.0, 1.0, 0.5)

sample_input = pd.DataFrame({
    'Charging_Duration': [duration],
    'Mean_Charging_Current': [current],
    'Mean_Temperature': [temperature],
    'Total_Energy_Consumed': [energy],
    'Cycle_Progress': [cycle]
})

# Prediction button
if st.button("Predict Battery Capacity"):
    prediction_result = rf_model.predict(sample_input[ML_FEATURES])[0]
    st.metric("Predicted Remaining Capacity", f"{prediction_result:.2f}%")

