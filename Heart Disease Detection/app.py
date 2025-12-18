from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load Model and Scaler
MODEL_PATH = os.path.join("model", "heart_disease_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")

model = None
scaler = None

def load_model():
    global model, scaler
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading model/scaler: {e}")
        model = None
        scaler = None

# Load immediately on import/start
load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('result.html', error="Model not loaded properly.")

    try:
        # Extract features from form
        # Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar, Rest ECG, Max Heart Rate, Exercise Induced Angina, Oldpeak, ST Slope
        
        features = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope'])
        ]

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1] # Probability of being positive

        # Result Logic
        result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        result_class = "danger" if prediction == 1 else "success" # CSS class for color
        
        return render_template('result.html', 
                               prediction=result_text, 
                               result_class=result_class,
                               probability=round(probability * 100, 2))

    except Exception as e:
        return render_template('result.html', error=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
