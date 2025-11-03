import streamlit as st
import altair as alt
from helper_functions import load_data, load_model

st.header("🌡️ Temperature Insights")

# --- Load model and data ---
model = load_model("models (2).pkl")
df = load_data("feature_engineered_battery.csv")

# --- Use all devices (no filtering) ---
filtered_df = df  # Display all devices

# --- Chart 1 — Temperature trend over time ---
st.subheader("Device Temperature Over Time")
chart_temp = (
    alt.Chart(filtered_df)
    .mark_line()
    .encode(
        x=alt.X('Timestamp:T', title="Time"),
        y=alt.Y('Mean_Temperature:Q', title="Mean Temperature (°C)"),
        color=alt.Color('Device_ID:N', title="Device ID"),
        tooltip=['Device_ID', 'Mean_Temperature', 'Timestamp']
    )
    .interactive()
)
st.altair_chart(chart_temp, use_container_width=True)

# --- Chart 2 — Temperature vs Remaining Capacity ---
st.subheader("Temperature vs Remaining Capacity")
if 'Remaining_Capacity' in filtered_df.columns:
    chart_corr = (
        alt.Chart(filtered_df)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X('Mean_Temperature:Q', title="Mean Temperature (°C)"),
            y=alt.Y('Remaining_Capacity:Q', title="Remaining Capacity (%)"),
            color=alt.Color('Device_ID:N', title="Device ID"),
            tooltip=['Device_ID', 'Mean_Temperature', 'Remaining_Capacity']
        )
        .interactive()
    )
    st.altair_chart(chart_corr, use_container_width=True)
else:
    st.info("Column 'Remaining_Capacity' not found in dataset — skipping capacity comparison.")

