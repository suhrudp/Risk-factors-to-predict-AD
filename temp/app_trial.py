from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'  # Path to the saved StandardScaler

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

app = Flask(__name__)

# Define feature names (same order as training)
FEATURES = [
    'PDOTHR', 'HYPOSOM', 'NACCAGE', 'NACCFAM', 'EDUC', 'NACCLIVS', 'NACCADEP', 
    'HRATE', 'NACCBMI', 'NACCPDMD', 'PD', 'REMDIS', 'NACCAMD', 'ARTH', 'ALCOHOL', 
    'OTHCOND', 'BPSYS', 'VISWCORR', 'ALCOCCAS', 'NACCOM', 'CBSTROKE', 'NACCAPSY', 
    'INEDUC', 'SEIZURES', 'B12DEF', 'NACCAHTN', 'BPDIAS', 'HEARWAID', 'NACCMOM', 
    'DIABET', 'PACKSPER', 'HEARING', 'NACCNINR'
]

# Define continuous variables that need scaling
CONTINUOUS_FEATURES = [
    "NACCAGE", "EDUC", "INEDUC", "NACCAMD", "PACKSPER", 
    "NACCBMI", "BPSYS", "BPDIAS", "HRATE"
]

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert inputs to float
        input_values = {feature: float(request.form[feature]) for feature in FEATURES}
        
        # Separate continuous and categorical variables
        continuous_values = np.array([[input_values[feature] for feature in CONTINUOUS_FEATURES]])  # 2D for scaler
        categorical_values = [input_values[feature] for feature in FEATURES if feature not in CONTINUOUS_FEATURES]

        # Apply Standard Scaling **only** to continuous variables
        scaled_continuous = scaler.transform(continuous_values)[0]  # Flatten after transformation
        
        # Reconstruct the final input in the correct order
        final_features = []
        for feature in FEATURES:
            if feature in CONTINUOUS_FEATURES:
                final_features.append(scaled_continuous[list(CONTINUOUS_FEATURES).index(feature)])
            else:
                final_features.append(input_values[feature])

        # Convert to NumPy array and reshape for prediction
        final_features = np.array(final_features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)
        output = 'High Risk' if prediction[0] == 1 else 'Low Risk'

        return render_template('index.html', features=FEATURES, prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', features=FEATURES, prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)