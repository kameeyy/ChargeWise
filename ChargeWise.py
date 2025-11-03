import streamlit as st
import pandas as pd
import joblib
import altair as alt
from helper_functions import load_data, load_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="ChargeWise Dashboard", layout="wide")

# --- PAGE HEADER ---
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

st.sidebar.success("Select a page above")

# --- LOAD MODEL + DATA ---
model = load_model("Models (2).pkl")
df = load_data("feature_engineered_battery.csv")   # renamed from 'data' → 'df' for clarity

# --- FILTER SECTION ---
st.header("⚙️ Filter Options")
device_options = df['Device_ID'].unique()
selected_devices = st.multiselect("Select Device(s):", device_options, default=list(device_options))
filtered_df = df[df['Device_ID'].isin(selected_devices)]

# --- OVERVIEW SECTION ---
st.header("📊 Overview Dashboard")

col1, col2 = st.columns(2)

# Chart 1 – Average Charging Duration
avg_duration = filtered_df.groupby('Device_ID')['Charging_Duration'].mean().reset_index()
chart1 = (
    alt.Chart(avg_duration)
    .mark_bar(color="#76b7b2")
    .encode(
        x=alt.X('Device_ID:N', title="Device ID"),
        y=alt.Y('Charging_Duration:Q', title="Average Duration (s)"),
        tooltip=['Device_ID', 'Charging_Duration']
    )
)
col1.altair_chart(chart1, use_container_width=True)

# Chart 2 – Total Energy Consumed
energy = filtered_df.groupby('Device_ID')['Total_Energy_Consumed'].sum().reset_index()
chart2 = (
    alt.Chart(energy)
    .mark_bar(color="#f28e2b")
    .encode(
        x=alt.X('Device_ID:N', title="Device ID"),
        y=alt.Y('Total_Energy_Consumed:Q', title="Energy (Wh)"),
        tooltip=['Device_ID', 'Total_Energy_Consumed']
    )
)
col2.altair_chart(chart2, use_container_width=True)



