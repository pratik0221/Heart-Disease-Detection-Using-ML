import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# 1. Load Dataset
# URL for the Heart Disease Dataset (Cleveland)
url = "https://raw.githubusercontent.com/kb22/Heart-Disease-Prediction/master/dataset.csv"

print(f"Downloading dataset from {url}...")
try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully.")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
except Exception as e:
    print(f"Error downloading dataset: {e}")
    # Fallback to synthetic data if download fails (to ensure app works for demo)
    print("Generating synthetic data for demonstration...")
    # Generate 300 samples with required columns
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = np.random.rand(300, 14)
    df = pd.DataFrame(data, columns=columns)
    # Adjust ranges roughly to reality
    df['age'] = (df['age'] * 50 + 30).astype(int)
    df['sex'] = (df['sex'] > 0.5).astype(int)
    df['cp'] = (df['cp'] * 4).astype(int)
    df['trestbps'] = (df['trestbps'] * 50 + 100).astype(int)
    df['chol'] = (df['chol'] * 200 + 150).astype(int)
    df['fbs'] = (df['fbs'] > 0.85).astype(int)
    df['restecg'] = (df['restecg'] * 3).astype(int)
    df['thalach'] = (df['thalach'] * 100 + 100).astype(int)
    df['exang'] = (df['exang'] > 0.7).astype(int)
    df['oldpeak'] = df['oldpeak'] * 4
    df['slope'] = (df['slope'] * 3).astype(int)
    df['target'] = (df['target'] > 0.5).astype(int)
    print("Synthetic data generated.")

# Canonicalize column names
# The dataset from kb22/Heart-Disease-Prediction often has proper names or slightly different ones.
# We need to map them to: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope
# Let's inspect columns dynamically or force rename if we are sure of the order.
# The standard order is usually 14 columns.
# If columns are named differently, we try to rename.
expected_cols_ordered = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

if len(df.columns) == 14:
    df.columns = expected_cols_ordered
else:
    # If not 14, we might be in trouble or it's a subset.
    # We will assume the user inputs map to what we have.
    # For this task, strict mapping is needed.
    pass

required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope']
target = 'target'


# 2. Preprocessing
X = df[required_features]
y = df[target]

print(f"Feature shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5. Save Model and Scaler
model_path = os.path.join("model", "heart_disease_model.pkl")
scaler_path = os.path.join("model", "scaler.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
