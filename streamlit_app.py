import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# 1. Function to manually input data for each feature
def get_input():
    st.title('Enter Patient Data')

    spinal_arthritis = st.selectbox("Spinal Arthritis (0 or 1):", [0, 1])
    history_of_stroke = st.selectbox("History of Stroke (0 or 1):", [0, 1])
    age = st.slider("Age (40-80):", 40, 80, 60)
    other_parkinsons_symptoms = st.selectbox("Other Parkinson's Disease Symptoms (0 or 1):", [0, 1])
    height = st.slider("Height (150-190 cm):", 150, 190, 170)
    sleep_apnea_diagnosis = st.selectbox("Sleep Apnea Diagnosis (0 or 1):", [0, 1])
    heart_rate = st.slider("Heart Rate (60-100):", 60, 100, 75)
    seizure_episodes = st.selectbox("Seizure Episodes (0 or 1):", [0, 1])
    psychiatric_disorders = st.selectbox("Psychiatric Disorders (0 or 1):", [0, 1])
    cigarettes_smoked = st.slider("Cigarettes Smoked Per Day (0-30):", 0, 30, 10)
    arthritis_diagnosis = st.selectbox("Arthritis Diagnosis (0 or 1):", [0, 1])
    depression_last_2_years = st.selectbox("Depression in Last 2 Years (0 or 1):", [0, 1])
    diastolic_blood_pressure = st.slider("Diastolic Blood Pressure (70-90):", 70, 90, 80)
    rem_sleep_behavior_disorder = st.selectbox("REM Sleep Behavior Disorder (0 or 1):", [0, 1])
    cardiovascular_conditions = st.selectbox("Other Cardiovascular Conditions (0 or 1):", [0, 1])
    diabetes_diagnosis = st.selectbox("Diabetes Diagnosis (0 or 1):", [0, 1])
    bmi = st.slider("Body Mass Index (18.5-35):", 18.5, 35.0, 25.0)
    insomnia_hyposomnia = st.selectbox("Insomnia/Hyposomnia (0 or 1):", [0, 1])
    vitamin_b12_deficiency = st.selectbox("Vitamin B12 Deficiency (0 or 1):", [0, 1])

    # Creating a dictionary from user input
    data = {
        "Spinal Arthritis": spinal_arthritis,
        "History of Stroke": history_of_stroke,
        "Age": age,
        "Other Parkinson's Disease Symptoms": other_parkinsons_symptoms,
        "Height": height,
        "Sleep Apnea Diagnosis": sleep_apnea_diagnosis,
        "Heart Rate": heart_rate,
        "Seizure Episodes": seizure_episodes,
        "Psychiatric Disorders": psychiatric_disorders,
        "Cigarettes Smoked Per Day": cigarettes_smoked,
        "Arthritis Diagnosis": arthritis_diagnosis,
        "Depression in Last 2 Years": depression_last_2_years,
        "Diastolic Blood Pressure": diastolic_blood_pressure,
        "REM Sleep Behavior Disorder": rem_sleep_behavior_disorder,
        "Other Cardiovascular Conditions": cardiovascular_conditions,
        "Diabetes Diagnosis": diabetes_diagnosis,
        "Body Mass Index": bmi,
        "Insomnia/Hyposomnia": insomnia_hyposomnia,
        "Vitamin B12 Deficiency": vitamin_b12_deficiency
    }

    # Convert the data dictionary to DataFrame
    return pd.DataFrame([data])

# 2. Load the pre-trained model
model = joblib.load('/path/to/your/model.pkl')  # Change this path accordingly

# 3. Get the data from the user
user_data = get_input()

# 4. Make predictions using the model
if st.button("Predict"):
    prediction = model.predict(user_data)
    st.write("Prediction:", prediction)

    # 5. SHAP Explanation
    # Create a SHAP explainer using the model
    explainer = shap.TreeExplainer(model)  # Use the appropriate explainer (e.g., TreeExplainer for tree-based models)

    # Calculate SHAP values for the user input
    shap_values = explainer.shap_values(user_data)

    # Visualize SHAP values
    st.subheader("SHAP Summary Plot")
    shap.initjs()  # This initializes the SHAP visualization tools
    shap.summary_plot(shap_values, user_data)
