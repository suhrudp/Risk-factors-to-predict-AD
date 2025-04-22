import streamlit as st
import pickle
import numpy as np

# Load the model
with open('lightgbm_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.set_page_config(page_title="Alzheimer's Risk Prediction", page_icon="🧠")
st.title("Alzheimer's Risk Prediction")

st.write("Enter medical and lifestyle details to estimate the risk of Alzheimer's.")

# List of input fields
fields = [
    'NACCAGE', 'ARTSPIN', 'CBSTROKE', 'PDOTHR', 'SLEEPAP', 'SEIZURES', 'ARTH', 'PSYCDIS', 'HRATE',
    'REMDIS', 'PACKSPER', 'BPDIAS', 'WEIGHT', 'DEP2YRS', 'DIABETES', 'DEPOTHR', 'B12DEF_PREFINAL',
    'CVOTHR', 'ARTLOEX', 'HYPOSOM', 'NACCAMD', 'CBTIA', 'BPSYS', 'TOBAC100'
]

# Initialize input storage
input_data = []

# Create form
with st.form("prediction_form"):
    for field in fields:
        if field in ['HRATE', 'WEIGHT']:
            value = st.number_input(f"{field}:", step=0.01, format="%.2f")
        else:
            value = st.number_input(f"{field}:", step=1)
        input_data.append(value)

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    result = "Risk of developing Alzheimer's" if prediction[0] == 1 else "No risk of developing Alzheimer's"
    st.success(f"Prediction: {result}")
