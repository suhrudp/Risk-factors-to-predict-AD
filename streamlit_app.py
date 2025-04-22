import streamlit as st
import numpy as np
import pickle

# Load the model (skip loading scaler)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Title
st.title("Risk Factors to Predict Alzheimer's Disease")

# Define your inputs
st.header("Patient Information")

# Example: 9 features (customize as needed)
NACCAGE = st.number_input("Age", min_value=0, max_value=120, value=60)
BPDIAS = st.number_input("Diastolic BP", min_value=0, max_value=200, value=80)
BPSYS = st.number_input("Systolic BP", min_value=0, max_value=300, value=120)
WEIGHT = st.number_input("Weight (kg)", min_value=0, max_value=300, value=70)
HRATE = st.number_input("Heart Rate", min_value=0, max_value=200, value=70)
PACKSPER = st.number_input("Packs of cigarettes per day", min_value=0, max_value=100, value=0)
DIABETES = st.selectbox("Diabetes", options=[0, 1])
CBSTROKE = st.selectbox("History of Stroke", options=[0, 1])
SLEEPAP = st.selectbox("Sleep Apnea", options=[0, 1])

# Collect into array
input_data = [NACCAGE, BPDIAS, BPSYS, WEIGHT, HRATE, PACKSPER, DIABETES, CBSTROKE, SLEEPAP]

# Predict button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    
    result = "Risk of developing Alzheimer's" if prediction[0] == 1 else "No significant risk"
    st.success(f"Prediction: {result}")
