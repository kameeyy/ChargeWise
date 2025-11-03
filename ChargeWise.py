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

@st.cache_resource
def load_ml_model(path="models (2).pkl"):
    """Loads the pre-trained ML model only once."""
    try:
        # Load the model using joblib as specified in your imports
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {path}. Please check the path.")
        return None

@st.cache_data
def load_data(file_path='feature_engineered_battery.csv'):
    """Loads the final feature-engineered data only once."""
    try:
        # We use the feature_engineered file as it's the final cleaned set
        df = pd.read_csv(file_path)
        # Convert Timestamp after loading
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {file_path}. Please check the path.")
        return None

# Execute cached functions
rf_model = load_ml_model()
final_df = load_data()

# Define the features list used in your model (based on your original code)
ML_FEATURES = ['Charging_Duration', 'Mean_Charging_Current', 'Mean_Temperature',
               'Total_Energy_Consumed', 'Cycle_Progress']

# --- 2. DASHBOARD BODY ---

if final_df is None or rf_model is None:
    st.stop() # Stop if files failed to load

st.header("🔍 Exploratory Data Analysis (EDA)")

# --- ROW 1: Charging Duration and Energy Consumption ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Charging Duration by Device")
    avg_duration = final_df.groupby('Device_ID')['Charging_Duration'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    avg_duration.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel('Duration (seconds)')
    st.pyplot(fig)

with col2:
    st.subheader("Total Energy Consumed by Device (Wh)")
    energy = final_df.groupby('Device_ID')['Total_Energy_Consumed'].sum().sort_values()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    energy.plot(kind='bar', color='lightcoral', ax=ax)
    ax.set_ylabel('Energy (Wh)')
    st.pyplot(fig)

# --- ROW 2: Temperature Diagnostics ---
st.header("🌡️ Battery Health Diagnostics")
col3, col4 = st.columns(2)

with col3:
    st.subheader('Temperature vs Battery Percent')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(final_df['Battery_Percentage'], final_df['Battery_Operating_Temperature'], alpha=0.4)
    ax.set_xlabel('Battery Percentage (%)')
    ax.set_ylabel('Temperature (°C)')
    st.pyplot(fig)

with col4:
    st.subheader('Distribution of Battery Operating Temperature')
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(final_df['Battery_Operating_Temperature'], bins=30, kde=True, color='orange', ax=ax)
    ax.set_xlabel('Temperature (°C)')
    st.pyplot(fig)

st.divider()

# --- 3. PREDICTION SECTION ---
st.header("🧠 Predictive Capacity Test")

# Define the sample input for prediction (based on your original code)
sample_input = pd.DataFrame({
    'Charging_Duration': [0.3], 
    'Mean_Charging_Current': [400], 
    'Mean_Temperature': [32], 
    'Total_Energy_Consumed': [0.01], 
    'Cycle_Progress': [0.5] 
})

# Run the prediction
try:
    # Ensure the DataFrame only contains the features the model expects
    prediction_result = rf_model.predict(sample_input[ML_FEATURES])[0]
    
    st.metric(
        label="Predicted Remaining Capacity for Sample Scenario",
        value=f"{prediction_result:.2f}%",
        help="Input: 50% cycle progress, 32°C temp, moderate current/duration."
    )
    
    st.caption("This prediction demonstrates the model's ability to estimate capacity based on charging habits and life cycle.")

except Exception as e:
    st.error(f"Prediction failed. Please check feature columns and model integrity. Error: {e}")