# 🚀 AI-Based Oilfield Predictive Maintenance System

## 📌 Project Overview

The AI-Based Oilfield Predictive Maintenance System is an end-to-end intelligent maintenance platform designed to predict equipment failures, estimate Remaining Useful Life (RUL), diagnose bearing faults, and provide explainable maintenance recommendations for critical oilfield equipment.

The system combines Machine Learning, Explainable AI (SHAP), Digital Twin simulation, and real-time monitoring dashboards to support proactive maintenance decisions and reduce operational downtime.

---

## 🌐 Live Demo

**Streamlit Deployment:**
https://oilfield-predictive-maintenance-niptebitbmrxmxbog93gyx.streamlit.app/

---

## 🎯 Objectives

* Predict equipment failures before breakdowns occur.
* Estimate Remaining Useful Life (RUL) of assets.
* Monitor equipment health in real time.
* Detect bearing faults using AI models.
* Explain model predictions using SHAP Explainable AI.
* Generate maintenance recommendations automatically.
* Visualize asset health through Digital Twin technology.

---

## 🛠 Technologies Used

### Programming Language

* Python

### Machine Learning

* XGBoost
* Scikit-Learn
* SHAP

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* Streamlit

### Deployment

* Streamlit Cloud
* GitHub

---

## 🔥 Key Features

### 1. Failure Prediction

Predicts the probability of equipment failure using an XGBoost classification model.

### 2. Remaining Useful Life (RUL) Estimation

Estimates how long equipment can continue operating before maintenance is required.

### 3. SHAP Explainable AI

Provides feature-level explanations for every prediction, enabling transparent decision-making.

### 4. Digital Twin Simulation

Creates a virtual representation of equipment health and operational status.

### 5. Bearing Fault Diagnosis

Identifies bearing conditions such as:

* Normal
* Inner Race Fault
* Outer Race Fault
* Ball Bearing Fault

### 6. Maintenance Recommendation Engine

Automatically recommends:

* Inspection
* Maintenance Scheduling
* Asset Replacement
* Emergency Shutdown

### 7. Fleet Monitoring Dashboard

Monitors multiple equipment units simultaneously and ranks assets by risk level.

### 8. Risk Assessment System

Calculates asset risk scores using:

* Failure Probability
* Equipment Health
* Digital Twin Health
* Asset Aging

### 9. Maintenance Cost Estimation

Provides estimated maintenance costs based on equipment type and predicted risk.

### 10. Real-Time Dashboard

Interactive Streamlit dashboard with:

* Telemetry Monitoring
* Failure Probability Trends
* Health Metrics
* Forecast Charts
* Event History

---

## 📊 Equipment Supported

* Pump
* Compressor
* Drilling Motor
* Bearing Assembly

---

## 🏗 System Architecture

Sensor Data / Telemetry

↓

Feature Engineering

↓

Data Preprocessing & Scaling

↓

XGBoost Failure Prediction Model

↓

SHAP Explainability

↓

Maintenance Recommendation Engine

↓

Digital Twin Simulation

↓

Interactive Streamlit Dashboard

---

## 📈 Dashboard Modules

* Equipment Aging Analysis
* Asset Health Monitoring
* Failure Probability Gauge
* Maintenance Scheduler
* Bearing Fault Diagnosis
* SHAP Explainability
* Fleet Monitoring
* Risk Ranking
* Digital Twin Dashboard
* Failure Forecasting
* Event Logging

---

## 📂 Project Structure

```text
Oil_Field_AI_Predictive_Maintainace_system/
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── xgboost_failure_model.pkl
│   ├── scaler.pkl
│   ├── bearing_fault_model.pkl
│   └── bearing_label_encoder.pkl
│
├── logs/
│   └── events.csv
│
├── requirements.txt
├── README.md
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run dashboard/app.py
```

## 📊 AI Models Used

### Failure Prediction Model

* Algorithm: XGBoost Classifier
* Output: Failure Probability

### Bearing Fault Diagnosis Model

* Classification Model
* Predicts Bearing Health Condition

### Explainability

* SHAP Tree Explainer

---

## 💡 Real-World Applications

* Oil & Gas Industry
* Drilling Operations
* Industrial Manufacturing
* Smart Factories
* Predictive Asset Management
* Industrial IoT Systems

---

## 🎓 Academic Significance

This project demonstrates practical applications of:

* Machine Learning
* Predictive Maintenance
* Explainable AI (XAI)
* Digital Twin Technology
* Industrial Analytics
* AI-Powered Decision Support Systems

---

## 👨‍💻 Author

**Upashtika Bordoloi**
B.Tech – Artificial Intelligence & Machine Learning
