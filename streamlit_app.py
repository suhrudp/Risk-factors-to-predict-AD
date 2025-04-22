import streamlit as st
import numpy as np
import pickle

# Load your model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Alzheimer's Risk Prediction")
st.write("Enter medical and lifestyle details to estimate the risk of Alzheimer's.")

# Collect user input
data = {}
features = [
    'NACCAGE', 'ARTSPIN', 'CBSTROKE', 'PDOTHR', 'SLEEPAP', 'SEIZURES', 'ARTH', 'PSYCDIS', 'HRATE', 'REMDIS',
    'PACKSPER', 'BPDIAS', 'WEIGHT', 'DEP2YRS', 'DIABETES', 'DEPOTHR', 'B12DEF_PREFINAL', 'CVOTHR',
    'ARTLOEX', 'HYPOSOM', 'NACCAMD', 'CBTIA', 'BPSYS', 'TOBAC100'
]

for feature in features:
    if feature in ['HRATE', 'WEIGHT']:
        data[feature] = st.number_input(f"{feature}:", value=0.0)
    else:
        data[feature] = st.number_input(f"{feature}:", value=0)

# Submit button
if st.button('Predict'):
    input_array = np.array(list(data.values())).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Risk of developing Alzheimer's" if prediction[0] == 1 else "Low risk"
    st.success(f"Prediction: {result}")
