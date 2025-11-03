# helper_functions.py
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)
