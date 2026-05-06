# 🩺 Smart HealthCare (AI for Social Good)

## 📌 Problem Statement

Early detection of diseases like diabetes and heart disease is challenging without automated screening systems.
This project aims to develop an AI-based system to predict disease risk using patient medical data and enable early diagnosis.

---

## 🚀 Project Overview

**Smart HealthCare** is a machine learning–based web application that predicts diseases based on user-input symptoms.
It uses a deep learning model trained on medical symptom data to provide **quick and preliminary disease predictions**.

This system helps in:

* Early detection of diseases
* Reducing dependency on manual screening
* Supporting healthcare decision-making

---

## 🧠 Technologies Used

* Python 🐍
* TensorFlow / Keras
* Pandas & NumPy
* Scikit-learn
* Streamlit (Web App)
* Power BI (Data Visualization)

---

## 📂 Dataset

The dataset used for this project is from Kaggle:

🔗 https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

### Dataset Features:

* 132 symptom columns (binary values: 0 or 1)
* 1 target column: **Prognosis (Disease)**
* 40+ diseases

---

## 🏗️ Project Structure

```
Smart-Healthcare/
│── data/
│   ├── Training.csv
│   ├── Testing.csv
│
│── Model/
│   ├── train_model.py
│   ├── predict.py
│   ├── model.h5
│   ├── encoder.pkl
│
│── app.py


## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/smart-healthcare.git
cd smart-healthcare
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model

```bash
python Model/train_model.py
```
---

## ▶️ Run the Application

### 🔹 Run Streamlit Web App

```bash
python -m streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```
---

## 🖥️ Features

* ✔ Disease prediction using symptoms
* ✔ Deep learning model (ANN)
* ✔ Interactive web interface (Streamlit)
* ✔ Fast and user-friendly UI
* ✔ Supports multiple disease classes

---

## 📊 Visualization (Power BI)

Power BI is used to:

* Analyze disease distribution
* Identify common symptoms
* Visualize dataset insights

---

## 🧪 Model Details

* Model Type: Artificial Neural Network (ANN)
* Input: Symptom vector (0/1)
* Output: Predicted disease
* Loss Function: Categorical Crossentropy
* Optimizer: Adam

---

## ⚠️ Disclaimer
This project is for **educational purposes only**.
It is not a substitute for professional medical advice, diagnosis, or treatment.
---

## 🌍 Impact
This project supports **AI for Social Good** by:

* Providing accessible preliminary health screening
* Promoting awareness about disease symptoms
* Assisting in early-stage detection
---
## 🙌 Future Improvements

* Add real patient datasets
* Improve accuracy with advanced models
* Deploy on cloud (AWS / Streamlit Cloud)
* Add chatbot for medical assistance
---
