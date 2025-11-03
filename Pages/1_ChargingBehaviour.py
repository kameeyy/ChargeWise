import streamlit as st
import altair as alt
from helper_functions import load_data, load_model

st.header("⚡ Charging Behaviour Analysis")

# --- Load model and data ---
model = load_model("models (2).pkl")
df = load_data("feature_engineered_battery.csv")

# --- Chart: Charging Duration vs Energy Consumed ---
chart3 = (
    alt.Chart(df)
    .mark_circle(size=60, opacity=0.5, color="#59a14f")
    .encode(
        x=alt.X('Charging_Duration:Q', title="Charging Duration (s)"),
        y=alt.Y('Total_Energy_Consumed:Q', title="Total Energy Consumed (Wh)"),
        color=alt.Color('Device_ID:N', title="Device ID"),
        tooltip=['Device_ID', 'Charging_Duration', 'Total_Energy_Consumed']
    )
    .interactive()
)

st.altair_chart(chart3, use_container_width=True)
