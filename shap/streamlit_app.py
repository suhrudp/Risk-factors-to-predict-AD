import shap
import numpy as np
import pandas as pd
import torch
from torch import nn

# Example LSTM model definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_patients, output_dim=2, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.patient_embedding = nn.Embedding(num_patients, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, patient_ids):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        patient_intercepts = self.patient_embedding(patient_ids)
        out = out + patient_intercepts
        out = self.fc(out)
        return out

# Example input features and synthetic data generation
features = [
    "Spinal Arthritis", "History of Stroke", "Age", "Other Parkinson's Disease Symptoms", "Height", 
    "Sleep Apnea Diagnosis", "Heart Rate", "Seizure Episodes", "Psychiatric Disorders", "Cigarettes Smoked Per Day", 
    "Arthritis Diagnosis", "Depression in Last 2 Years", "Diastolic Blood Pressure", "REM Sleep Behavior Disorder", 
    "Other Cardiovascular Conditions", "Diabetes Diagnosis", "Body Mass Index", "Insomnia/Hyposomnia", 
    "Vitamin B12 Deficiency"
]

def generate_synthetic_data():
    # Generating random synthetic data similar to input range
    synthetic_data = {
        "Spinal Arthritis": np.random.randint(0, 2),
        "History of Stroke": np.random.randint(0, 2),
        "Age": np.random.randint(40, 80),
        "Other Parkinson's Disease Symptoms": np.random.randint(0, 2),
        "Height": np.random.uniform(150, 190),
        "Sleep Apnea Diagnosis": np.random.randint(0, 2),
        "Heart Rate": np.random.uniform(60, 100),
        "Seizure Episodes": np.random.randint(0, 2),
        "Psychiatric Disorders": np.random.randint(0, 2),
        "Cigarettes Smoked Per Day": np.random.randint(0, 30),
        "Arthritis Diagnosis": np.random.randint(0, 2),
        "Depression in Last 2 Years": np.random.randint(0, 2),
        "Diastolic Blood Pressure": np.random.randint(70, 90),
        "REM Sleep Behavior Disorder": np.random.randint(0, 2),
        "Other Cardiovascular Conditions": np.random.randint(0, 2),
        "Diabetes Diagnosis": np.random.randint(0, 2),
        "Body Mass Index": np.random.uniform(18.5, 35.0),
        "Insomnia/Hyposomnia": np.random.randint(0, 2),
        "Vitamin B12 Deficiency": np.random.randint(0, 2),
    }
    return pd.DataFrame([synthetic_data])

# Example LSTM model instance
input_dim = 20  # Number of features (adjust accordingly)
hidden_dim = 64
model = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_patients=1)

# Generate synthetic data
synthetic_data_df = generate_synthetic_data()

# Model prediction function for SHAP
def model_predict(data):
    data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    patient_ids = torch.tensor([0], dtype=torch.long)  # Using dummy patient id 0
    with torch.no_grad():
        outputs = model(data_tensor, patient_ids)
    return outputs.numpy()

# Generate SHAP values
explainer = shap.Explainer(model_predict, synthetic_data_df)
shap_values = explainer(synthetic_data_df)

# Extract SHAP values as numerical data
shap_df = pd.DataFrame(shap_values.values, columns=features)

# Calculate the mean absolute SHAP value for each feature
shap_df["Mean Absolute SHAP"] = shap_df.abs().mean(axis=0)

# Display the SHAP values as a numerical table
import ace_tools as tools; tools.display_dataframe_to_user(name="SHAP Values", dataframe=shap_df)
