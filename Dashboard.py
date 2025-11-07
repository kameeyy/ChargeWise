import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np
import os

# STREAMLIT CONFIGURATION
st.set_page_config(
    page_title="ChargeWise Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.image(os.path.join(os.path.dirname(__file__), "Logo(RemoveBg).png"), use_container_width=True)


# PAGE STYLING
def add_styles():
    st.markdown("""
    <style>
        /* Overall page background */
        .main {
            background-color: #cccdcc;
            color: #272726;
            font-family: 'Roboto', sans-serif;
            //font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }

        /* Header */
        .main-header {
            background: linear-gradient(90deg, #519748 0%, #3f7d3a 100%);
            padding: 1.2rem 2rem;
            border-radius: 15px;
            color: white;
            font-family: 'Roboto', sans-serif;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.3rem;
            text-align: center;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
        background-color: #519748 !important;   /* green background */
        color: white !important;                /* white text */
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin: 5px;
    }

    [data-testid="stMetricLabel"] {
        color: white !important;                /* metric label color */
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;                /* metric value color */
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        color: white !important;                /* up/down delta text */
    }

        /* Chart containers (Altair, Plotly, etc.) */
        [data-testid="stVegaLiteChart"], [data-testid="stPlotlyChart"] {
        background-color: white !important;
        border-radius: 20px !important;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        overflow: hidden !important;      /* ensures inner iframe doesn’t exceed the curve */
        clip-path: inset(0 round 20px);   /* cleanly clips chart content to rounded edges */
        }


        /* Tabs */
        [data-baseweb="tab"] {
            background-color: white !important;
            border-radius: 0x !important;
            color: #272726 !important;
            font-weight: 600;
            margin-right: 5px;
            padding: 10px 25px !important;
            transition: all 0.2s ease-in-out;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stTabs"] {
            padding: 0px 50px !important;   /* top/bottom | left/right */
            margin-top: 10px !important;
            margin-bottom: 20px !important;
        }
        [data-baseweb="tab"]:hover {
            background-color: #519748 !important;
            color: white !important;
            border: none !important;
        }
        [aria-selected="true"][data-baseweb="tab"] {
            background-color: #519748 !important;
            color: white !important;
            border: none !important;
            outline: none !important;
        }
        
        /* Optional: container spacing for visual clarity */
        [data-testid="stTabs"] {
        margin-bottom: 15px !important;
        }
        
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #cccdcc !important;
        }        


        /* Dataset preview */
        .dataframe {
            border-radius: 15px !important;
            overflow: hidden !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #519748;
            border-radius: 10px;
        }
    </style>
    <div class="main-header">
        <h1>ChargeWise Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)



add_styles()
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)





# DATASET PREPARATION
@st.cache_resource(show_spinner=False)
def ensure_parquet():
    """Create a Parquet version if not already available (handles mixed types safely)."""
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "feature_engineered_battery.csv")
    parquet_path = os.path.join(base_dir, "feature_engineered_battery.parquet")

    # Only convert once — saves time and lag
    if not os.path.exists(parquet_path):
        # st.info("Converting CSV to Parquet for faster loading...")
        df = pd.read_csv(csv_path)

        # Fix mixed-type columns that cause Arrow conversion errors
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)

        df.to_parquet(parquet_path, index=False)
        st.success("Parquet file created successfully!")
    else:
        st.info("File Existed")
        
    return parquet_path




#MODEL & DATA LOADERS
@st.cache_resource(show_spinner=False)
def load_model(path="models (2).pkl"):
    model_path = os.path.join(os.path.dirname(__file__), path)
    return joblib.load(model_path)

@st.cache_data(show_spinner=False)
def load_data(path):
    file_path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_parquet(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Device_ID'] = df['Device_ID'].astype('category')
    return df


# LOAD FILES
parquet_path = ensure_parquet()
rf_model = load_model()
df = load_data(parquet_path)


# Denormalize (Unscale) Charging Duration
max_normalised_value = 0.7        # max normalized value from dataset
assumed_max_minutes = 120         # assumed max real charging time in minutes

scaling_factor = assumed_max_minutes / max_normalised_value
df['Unscaled_Charging_Duration'] = df['Charging_Duration'] * scaling_factor

ML_FEATURES = ['Charging_Duration', 'Mean_Charging_Current',
               'Mean_Temperature', 'Total_Energy_Consumed', 'Cycle_Progress']



#PRE-COMPUTE PREDICTIONS (cached)
@st.cache_data(show_spinner=False)
def add_predictions(df, _model, features):
    df = df.copy()
    df['Predicted_Capacity'] = _model.predict(df[features])
    return df

predicted_df = add_predictions(df, rf_model, ML_FEATURES)




# SIDEBAR FILTER
st.sidebar.header("Filter Options")
device_options = sorted(predicted_df['Device_ID'].unique())
selected_devices = st.sidebar.multiselect("Select Device(s):", device_options, default=device_options)

@st.cache_data(show_spinner=False)
def filter_data(df, selected_devices):
    return df[df['Device_ID'].isin(selected_devices)]

filtered_df = filter_data(predicted_df, selected_devices)

MAX_POINTS = 3000
if len(filtered_df) > MAX_POINTS:
    filtered_df = filtered_df.sample(MAX_POINTS, random_state=42)

st.sidebar.success(f"Showing {len(filtered_df)} records.")



# MAIN PAGE TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview Dashboard",
    "Charging & Temperature Insights",
    "Model & Predictions",
    "Recommendations & Sustainability"
])




# PAGE 1: OVERVIEW DASHBOARD
with tab1:
    st.header("Overview Dashboard")
    st.divider()
    


    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Charging Duration", f"{int((m:=filtered_df['Unscaled_Charging_Duration'].mean()))}m {int(round((m % 1)*60))}s") #Changes by Daniel
    #col1.metric("Avg Charging Duration", f"{filtered_df['Unscaled_Charging_Duration'].mean():.2f} min")
    col2.metric("Avg Operating Temp", f"{filtered_df['Battery_Operating_Temperature'].mean():.2f} °C")
    col3.metric("Avg Energy Consumed", (lambda x: f"{x * 1000:.3f} mWh" if x < 0.001 else f"{x / 1000:.3f} kWh" if x >= 1000 else f"{x:.3f} Wh")(filtered_df['Total_Energy_Consumed'].mean())) #Changes by Daniel
    #col3.metric("Avg Energy Consumed", f"{filtered_df['Total_Energy_Consumed'].mean():.4f} Wh")
    col4.metric("Predicted Capacity", f"{filtered_df['Predicted_Capacity'].mean():.2f} %")

    st.divider()
    st.subheader("Device Comparison")

    col1, col2 = st.columns(2)
    with col1:
        avg_duration_device = filtered_df.groupby('Device_ID')['Unscaled_Charging_Duration'].mean().reset_index()
        chart1 = alt.Chart(avg_duration_device).mark_bar(color='#6EC5E9').encode(
        x=alt.X('Device_ID:N', title='Device ID'),
        y=alt.Y('Unscaled_Charging_Duration:Q', title='Avg Charging Duration (min)'),
        tooltip=['Device_ID', 'Unscaled_Charging_Duration']
    )

        st.altair_chart(chart1, use_container_width=True)

    
    with col2:
        energy_device = df.groupby('Device_ID')['Total_Energy_Consumed'].sum().reset_index()
        energy_device = energy_device[energy_device['Total_Energy_Consumed'] > 0]

        chart_energy = (
            alt.Chart(energy_device)
            .mark_bar(color='#E97171')
            .encode(
                x=alt.X('Device_ID:N', title='Device ID'),
                y=alt.Y('Total_Energy_Consumed:Q', title='Total Energy Consumed (Wh)'),
                tooltip=['Device_ID', 'Total_Energy_Consumed']
            )
        )


        st.altair_chart(chart_energy, use_container_width=True)

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(10))

# PAGE 2: CHARGING & TEMPERATURE INSIGHTS
with tab2:
    st.header("🌡 Charging & Temperature Insights")

    col1, col2 = st.columns(2)
    with col1:
        scatter1 = alt.Chart(filtered_df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('Unscaled_Charging_Duration:Q', title='Charging Duration (min)'),
            y=alt.Y('Mean_Temperature:Q', title='Mean Temperature (°C)'),
            color='Device_ID:N',
            tooltip=['Device_ID', 'Charging_Duration', 'Mean_Temperature']
        ).interactive()
        st.altair_chart(scatter1, use_container_width=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.altair_chart(chart1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    with col2:
        hist_temp = alt.Chart(filtered_df).mark_bar(opacity=0.7, color='orange').encode(
            alt.X('Battery_Operating_Temperature:Q', bin=alt.Bin(maxbins=30), title='Battery Operating Temperature (°C)'),
            y=alt.Y('count()', title='Count')
        )
        st.altair_chart(hist_temp, use_container_width=True)

    st.divider()
    st.subheader("🔗 Correlation Heatmap")
    corr = filtered_df.select_dtypes(include=[np.number]).corr()
    st.dataframe(corr.style.background_gradient(cmap='RdYlGn', axis=None))



# PAGE 3: MODEL & PREDICTIONS
with tab3:
    st.header("Model & Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importance")
        feat_importances = pd.DataFrame({
            'Feature': ML_FEATURES,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=True)
                # Clean feature names
        feat_importances['Feature'] = feat_importances['Feature'].str.replace('_', ' ')

        # Create feature importance chart (improved aesthetics)
        chart_feat = (
            alt.Chart(feat_importances)
            .mark_bar(color='#519748', cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                x=alt.X(
                    'Importance:Q',
                    title='Importance (%)',
                    scale=alt.Scale(domain=[0, (float(feat_importances['Importance'].max())*100) + 0.05]),  # tighter x-axis
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13, grid=False)
                ),
                y=alt.Y(
                    'Feature:N',
                    sort='x',
                    title='Feature',
                    axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelLimit=160)  # ensures full label visible
                ),
                tooltip=['Feature', 'Importance']
            )
            .properties(
                width=400,   # shorter width for compact look
                height=220,  # balanced height for 5 bars
            )
            .configure_axis(
                grid=False,
                domainColor='#aaaaaa'
            )
            .configure_view(
                strokeWidth=0  # removes chart border
            )
        )

        st.altair_chart(chart_feat, use_container_width=True)

    with col2:
        st.subheader("Predicted vs Actual (Sample)")
        sample = filtered_df.sample(min(100, len(filtered_df)))
        scatter_pred = alt.Chart(sample).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X('Battery_Percentage:Q', title='Actual Battery Capacity (%)'),
            y=alt.Y('Predicted_Capacity:Q', title='Predicted Capacity (%)'),
            color='Device_ID:N',
            tooltip=['Device_ID', 'Battery_Percentage', 'Predicted_Capacity']
        ).interactive()
        st.altair_chart(scatter_pred, use_container_width=True)



# PAGE 4: RECOMMENDATIONS & SUSTAINABILITY
with tab4:
    st.header("Recommendations & Sustainability")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
         **Best Practices**
        - Avoid overnight charging  
        - Keep charge between 20–80% 
        - Maintain temperature under 35°C 
        - Use adaptive charging if supported 
        """)

    with col2:
        st.success("""
        **Predicted Benefits**
        - Up to 30–40% longer battery lifespan  
        - Reduced energy waste  
        - Fewer premature replacements  
        - Lower environmental impact 
        """)
