import streamlit as st
import numpy as np
import pickle
import shap
import pandas as pd

# Load your pre-trained model
model_path = 'model.pkl'  # Replace with the actual model file path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app UI
st.title("Alzheimer's Risk Prediction")
st.write("Enter medical and lifestyle details to estimate the risk of Alzheimer's.")

# Feature Inputs (matching new features provided in the HTML form)
ARTSPIN = st.number_input("Spinal Arthritis (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
CBSTROKE = st.number_input("History of Stroke (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
NACCAGE = st.number_input("Age (in years)", min_value=0, value=0)
PDOTHR = st.number_input("Other Parkinson's Disease Symptoms (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
HEIGHT = st.number_input("Height (cm)", min_value=0.0, value=0.0, step=0.1)
SLEEPAP = st.number_input("Sleep Apnea Diagnosis (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
HRATE = st.number_input("Heart Rate (BPM)", value=0.00, format="%.2f")
SEIZURES = st.number_input("Seizure Episodes (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
PSYCDIS = st.number_input("Psychiatric Disorders (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
PACKSPER = st.number_input("Cigarettes Smoked Per Day", min_value=0, value=0)
ARTH = st.number_input("Arthritis Diagnosis (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
DEP2YRS = st.number_input("Depression in Last 2 Years (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
BPDIAS = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, value=0)
REMDIS = st.number_input("REM Sleep Behavior Disorder (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
CVOTHR = st.number_input("Other Cardiovascular Conditions (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
DIABETES = st.number_input("Diabetes Diagnosis (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
NACCBMI = st.number_input("Body Mass Index (kg/m²)", min_value=0.0, value=0.0, step=0.1)
HYPOSOM = st.number_input("Insomnia/Hyposomnia (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
B12DEF_PREFINAL = st.number_input("Vitamin B12 Deficiency (0: No, 1: Yes)", min_value=0, max_value=1, value=0)

# Collect inputs into a list (19 features here)
input_data = [
    ARTSPIN, CBSTROKE, NACCAGE, PDOTHR, HEIGHT, SLEEPAP, HRATE, SEIZURES, PSYCDIS,
    PACKSPER, ARTH, DEP2YRS, BPDIAS, REMDIS, CVOTHR, DIABETES, NACCBMI, HYPOSOM, B12DEF_PREFINAL
]

# Prediction button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)  # Ensure correct shape (1 row, 19 features)
    
    # Debugging: Display the input array
    st.write(f"Input data shape: {input_array.shape}")
    st.write(f"Input data: {input_array}")

    # Prediction using the loaded model
    try:
        # Use predict_proba to get the probability for class 1 (Alzheimer's risk)
        probability = model.predict_proba(input_array)[0][1]  # Probability of class 1

        # Format probability as percentage
        prob_percent = round(probability * 100, 2)

        # Display the result based on the prediction
        st.success(f"Estimated risk of developing Alzheimer's: {prob_percent}%")

        # SHAP values visualization
        explainer = shap.TreeExplainer(model)  # Create SHAP explainer
        shap_values = explainer.shap_values(input_array)  # Calculate SHAP values
        
        # Check the structure of shap_values
        st.write("SHAP Values Structure")
        st.write(f"shap_values[0] (Class 0): {shap_values[0]}")
        st.write(f"shap_values[1] (Class 1): {shap_values[1]}")

        # Extract SHAP values for class 1 (Alzheimer's risk)
        shap_df = pd.DataFrame(shap_values[1], columns=[
            "Spinal Arthritis", "History of Stroke", "Age", "Other Parkinson's Disease Symptoms", "Height", 
            "Sleep Apnea Diagnosis", "Heart Rate", "Seizure Episodes", "Psychiatric Disorders", "Cigarettes Smoked Per Day", 
            "Arthritis Diagnosis", "Depression in Last 2 Years", "Diastolic Blood Pressure", "REM Sleep Behavior Disorder", 
            "Other Cardiovascular Conditions", "Diabetes Diagnosis", "Body Mass Index", "Insomnia/Hyposomnia", 
            "Vitamin B12 Deficiency"
        ])
        
        # Display SHAP values numerically in a table
        st.subheader("Numerical SHAP Values for Class 1 (Alzheimer's Risk)")
        st.write(shap_df)  # Display SHAP values as a table
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
