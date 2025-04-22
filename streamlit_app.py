import streamlit as st
import numpy as np
import pickle

# Load your pre-trained model
model_path = 'lightgbm_model_22-04-2025.pkl'  # Replace with the actual model file path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app UI
st.title("Alzheimer's Risk Prediction")
st.write("Enter medical and lifestyle details to estimate the risk of Alzheimer's.")

# Feature Inputs (same as before)
NACCAGE = st.number_input("NACCAGE (Age)", min_value=0, value=0)
ARTSPIN = st.number_input("ARTSPIN (Spin Rate)", min_value=0, value=0)
CBSTROKE = st.number_input("CBSTROKE (History of Stroke)", min_value=0, value=0)
PDOTHR = st.number_input("PDOTHR (Other Disorders)", min_value=0, value=0)
SLEEPAP = st.number_input("SLEEPAP (Sleep Apnea)", min_value=0, value=0)
SEIZURES = st.number_input("SEIZURES (History of Seizures)", min_value=0, value=0)
ARTH = st.number_input("ARTH (Arthritis)", min_value=0, value=0)
PSYCDIS = st.number_input("PSYCDIS (Psychiatric Disorder)", min_value=0, value=0)
HRATE = st.number_input("HRATE (Heart Rate)", value=0.00, format="%.2f")
REMDIS = st.number_input("REMDIS (Recent Disorders)", min_value=0, value=0)
PACKSPER = st.number_input("PACKSPER (Pack Years of Smoking)", min_value=0, value=0)
BPDIAS = st.number_input("BPDIAS (Diastolic Blood Pressure)", min_value=0, value=0)
WEIGHT = st.number_input("WEIGHT (Weight in kg)", value=0.00, format="%.2f")
DEP2YRS = st.number_input("DEP2YRS (Depression Duration in years)", min_value=0, value=0)
DIABETES = st.number_input("DIABETES (Diabetes)", min_value=0, value=0)
DEPOTHR = st.number_input("DEPOTHR (Other Depression)", min_value=0, value=0)
B12DEF_PREFINAL = st.number_input("B12DEF_PREFINAL (B12 Deficiency Final)", min_value=0, value=0)
CVOTHR = st.number_input("CVOTHR (Cardiovascular Disease)", min_value=0, value=0)
ARTLOEX = st.number_input("ARTLOEX (Other ART History)", min_value=0, value=0)
HYPOSOM = st.number_input("HYPOSOM (Hypo-somnolence)", min_value=0, value=0)
NACCAMD = st.number_input("NACCAMD (NACC Alzheimer's Disease)", min_value=0, value=0)
CBTIA = st.number_input("CBTIA (Cognitive Decline)", min_value=0, value=0)
BPSYS = st.number_input("BPSYS (Systolic Blood Pressure)", min_value=0, value=0)
TOBAC100 = st.number_input("TOBAC100 (Tobacco Use, 100+ Cigarettes)", min_value=0, value=0)

# Collect inputs into a list
input_data = [
    NACCAGE, ARTSPIN, CBSTROKE, PDOTHR, SLEEPAP, SEIZURES, ARTH, PSYCDIS,
    HRATE, REMDIS, PACKSPER, BPDIAS, WEIGHT, DEP2YRS, DIABETES, DEPOTHR,
    B12DEF_PREFINAL, CVOTHR, ARTLOEX, HYPOSOM, NACCAMD, CBTIA, BPSYS, TOBAC100
]

# Prediction button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)  # Ensure correct shape (1 row, N features)
    
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
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
