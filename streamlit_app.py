import streamlit as st
import numpy as np
import pickle

# Load your model
model = pickle.load(open("model.pkl", "rb"))  # make sure model.pkl is in your project folder

# Streamlit app
st.title("Alzheimer's Risk Prediction")
st.write("Enter medical and lifestyle details to estimate the risk of Alzheimer's.")

# Feature Inputs
NACCAGE = st.number_input("NACCAGE", min_value=0, value=0)
ARTSPIN = st.number_input("ARTSPIN", min_value=0, value=0)
CBSTROKE = st.number_input("CBSTROKE", min_value=0, value=0)
PDOTHR = st.number_input("PDOTHR", min_value=0, value=0)
SLEEPAP = st.number_input("SLEEPAP", min_value=0, value=0)
SEIZURES = st.number_input("SEIZURES", min_value=0, value=0)
ARTH = st.number_input("ARTH", min_value=0, value=0)
PSYCDIS = st.number_input("PSYCDIS", min_value=0, value=0)
HRATE = st.number_input("HRATE", value=0.00, format="%.2f")
REMDIS = st.number_input("REMDIS", min_value=0, value=0)
PACKSPER = st.number_input("PACKSPER", min_value=0, value=0)
BPDIAS = st.number_input("BPDIAS", min_value=0, value=0)
WEIGHT = st.number_input("WEIGHT", value=0.00, format="%.2f")
DEP2YRS = st.number_input("DEP2YRS", min_value=0, value=0)
DIABETES = st.number_input("DIABETES", min_value=0, value=0)
DEPOTHR = st.number_input("DEPOTHR", min_value=0, value=0)
B12DEF_PREFINAL = st.number_input("B12DEF_PREFINAL", min_value=0, value=0)
CVOTHR = st.number_input("CVOTHR", min_value=0, value=0)
ARTLOEX = st.number_input("ARTLOEX", min_value=0, value=0)
HYPOSOM = st.number_input("HYPOSOM", min_value=0, value=0)
NACCAMD = st.number_input("NACCAMD", min_value=0, value=0)
CBTIA = st.number_input("CBTIA", min_value=0, value=0)
BPSYS = st.number_input("BPSYS", min_value=0, value=0)
TOBAC100 = st.number_input("TOBAC100", min_value=0, value=0)

# Collect inputs
input_data = [
    NACCAGE, ARTSPIN, CBSTROKE, PDOTHR, SLEEPAP, SEIZURES, ARTH, PSYCDIS,
    HRATE, REMDIS, PACKSPER, BPDIAS, WEIGHT, DEP2YRS, DIABETES, DEPOTHR,
    B12DEF_PREFINAL, CVOTHR, ARTLOEX, HYPOSOM, NACCAMD, CBTIA, BPSYS, TOBAC100
]

# Prediction button
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "Risk of developing Alzheimer's" if prediction[0] == 1 else "No significant risk detected"
    st.success(f"Prediction: {result}")