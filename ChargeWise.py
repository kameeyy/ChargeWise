import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt

# ----------------------------------------------------------------
# ⚙️ STREAMLIT CONFIGURATION
# ----------------------------------------------------------------
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


# ----------------------------------------------------------------
# 🧠 CACHED FUNCTIONS (LOAD MODEL + DATA)
# ----------------------------------------------------------------
@st.cache_resource
def load_ml_model(path="models (2).pkl"):
    """Load pre-trained ML model"""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {path}")
        return None


@st.cache_data
def load_data(file_path='feature_engineered_battery.csv'):
    """Load dataset"""
    try:
        df = pd.read_csv(file_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found at: {file_path}")
        return None


# ----------------------------------------------------------------
# 📦 LOAD EVERYTHING
# ----------------------------------------------------------------
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

st.sidebar.success(f"✅ Showing {len(filtered_df)} records after filtering.")


# ----------------------------------------------------------------
# 🗂️ TABS LAYOUT
# ----------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Charging Behaviour Analysis",
    "Temperature & Performance Insights",
    "Model Prediction Results",
    "Sustainability and Recommendations"
])


# ----------------------------------------------------------------
# 📈 TAB 1 – OVERVIEW
# ----------------------------------------------------------------
with tab1:
    st.header("📊 Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Charging Duration by Device (Altair)")
        avg_duration = filtered_df.groupby('Device_ID')['Charging_Duration'].mean().reset_index()

        chart1 = (
            alt.Chart(avg_duration)
            .mark_bar(color="#76b7b2")
            .encode(
                x=alt.X('Device_ID:N', title="Device ID"),
                y=alt.Y('Charging_Duration:Q', title="Average Duration (s)"),
                tooltip=['Device_ID', 'Charging_Duration']
            )
            .properties(width=350, height=300)
        )
        st.altair_chart(chart1, use_container_width=True)

    with col2:
        st.subheader("Total Energy Consumed by Device (Altair)")
        energy = filtered_df.groupby('Device_ID')['Total_Energy_Consumed'].sum().reset_index()

        chart2 = (
            alt.Chart(energy)
            .mark_bar(color="#f28e2b")
            .encode(
                x=alt.X('Device_ID:N', title="Device ID"),
                y=alt.Y('Total_Energy_Consumed:Q', title="Energy (Wh)"),
                tooltip=['Device_ID', 'Total_Energy_Consumed']
            )
            .properties(width=350, height=300)
        )
        st.altair_chart(chart2, use_container_width=True)


# ----------------------------------------------------------------
# ⚡ TAB 2 – CHARGING BEHAVIOUR ANALYSIS
# ----------------------------------------------------------------
with tab2:
    st.header("⚡ Charging Behaviour Analysis")

    st.subheader("Charging Duration vs Total Energy (Altair)")
    chart3 = (
        alt.Chart(filtered_df)
        .mark_circle(size=60, opacity=0.5, color="#59a14f")
        .encode(
            x=alt.X('Charging_Duration:Q', title="Charging Duration"),
            y=alt.Y('Total_Energy_Consumed:Q', title="Energy Consumed (Wh)"),
            color='Device_ID:N',
            tooltip=['Device_ID', 'Charging_Duration', 'Total_Energy_Consumed']
        )
        .interactive()
        .properties(height=400)
    )
    st.altair_chart(chart3, use_container_width=True)


# ----------------------------------------------------------------
# 🌡️ TAB 3 – TEMPERATURE & PERFORMANCE INSIGHTS
# ----------------------------------------------------------------
with tab3:
    st.header("🌡️ Temperature & Performance Insights")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Temperature vs Battery Percentage (Altair)")
        chart4 = (
            alt.Chart(filtered_df)
            .mark_circle(size=60, opacity=0.4, color="#4e79a7")
            .encode(
                x=alt.X('Battery_Percentage:Q', title="Battery Percentage (%)"),
                y=alt.Y('Battery_Operating_Temperature:Q', title="Temperature (°C)"),
                tooltip=['Device_ID', 'Battery_Percentage', 'Battery_Operating_Temperature']
            )
            .interactive()
            .properties(height=350)
        )
        st.altair_chart(chart4, use_container_width=True)

    with col4:
        st.subheader("Distribution of Battery Operating Temperature (Altair)")
        chart5 = (
            alt.Chart(filtered_df)
            .mark_bar(color="#e15759", opacity=0.8)
            .encode(
                alt.X('Battery_Operating_Temperature:Q', bin=alt.Bin(maxbins=30), title="Temperature (°C)"),
                alt.Y('count()', title='Count')
            )
            .properties(height=350)
        )
        st.altair_chart(chart5, use_container_width=True)


# ----------------------------------------------------------------
# 🔮 TAB 4 – MODEL PREDICTION RESULTS
# ----------------------------------------------------------------
with tab4:
    st.header("🔮 Predict Battery Remaining Capacity")

    st.markdown("Adjust the sliders below to simulate different charging scenarios:")

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

    if st.button("Predict Battery Capacity"):
        prediction_result = rf_model.predict(sample_input[ML_FEATURES])[0]
        st.metric("Predicted Remaining Capacity", f"{prediction_result:.2f}%")

        # Example trend visualization (optional)
        st.subheader("Predicted vs Actual Trend (Example)")
        trend_data = filtered_df.sample(50).copy()
        trend_data['Predicted'] = rf_model.predict(trend_data[ML_FEATURES])

        chart6 = (
            alt.Chart(trend_data)
            .transform_fold(['Battery_Capacity', 'Predicted'], as_=['Type', 'Value'])
            .mark_line(point=True)
            .encode(
                x='Cycle_Progress:Q',
                y='Value:Q',
                color='Type:N',
                tooltip=['Cycle_Progress', 'Value', 'Type']
            )
            .properties(height=350)
        )
        st.altair_chart(chart6, use_container_width=True)


# ----------------------------------------------------------------
# 🌱 TAB 5 – SUSTAINABILITY AND RECOMMENDATIONS
# ----------------------------------------------------------------
with tab5:
    st.header("🌱 Sustainability and Recommendations")
    st.write("""
    **Recommendations for Optimal Battery Performance:**
    - Avoid charging at very high or low temperatures.
    - Maintain moderate charging current to preserve capacity.
    - Minimize full discharge cycles.
    - Use smart charging strategies to reduce total energy loss.
    - Implement predictive maintenance using the ML model insights.
    """)

