# ❤️ Heart Disease Detection Web Application

An AI-powered web application that predicts heart disease risk using machine learning. Built with **Flask** and a modern **glassmorphism UI**.

---

## 🎯 Features

* Machine Learning–based heart disease prediction
* Real-time risk analysis
* Clean and modern UI
* Responsive design
* Uses **11 medical parameters**

---

## 🧠 Medical Parameters Used

* Age, Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise Angina
* Oldpeak (ST Depression)
* ST Slope

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash
git clone https://github.com/pratik0221/heart-disease-detection.git
cd heart-disease-detection
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open browser and visit:
👉 `http://127.0.0.1:5000`

---

## 📁 Project Structure

```
heart-disease-detection/
├── app.py
├── requirements.txt
├── README.md
├── model/
│   ├── heart_disease_model.pkl
│   ├── scaler.pkl
│   └── train_model.py
├── static/
│   ├── css/style.css
│   └── images/
└── templates/
    ├── index.html
    └── result.html
```

---

## 🤖 Model Details

* Algorithm: Random Forest / Logistic Regression
* Input: 11 medical features
* Output: Disease Detected / No Disease Detected

---

## 🛠️ Technologies Used

* Flask (Python)
* scikit-learn
* pandas, numpy
* HTML5, CSS3
* Pickle

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used for medical diagnosis.

---

## 👨‍💻 Author

**Pratik Raju Mohite**

* GitHub: [https://github.com/pratik0221](https://github.com/pratik0221)

