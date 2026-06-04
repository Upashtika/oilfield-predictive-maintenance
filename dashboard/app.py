import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    def st_autorefresh(interval=0, key=None):
        return None
import pandas as pd
import random
from datetime import datetime
from datetime import timedelta
import time
import joblib
import plotly.express as px
import os
import shap
import plotly.graph_objects as go
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "events.csv")

os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    pd.DataFrame(
        columns=[
            "Timestamp",
            "Failure_Probability",
            "Status"
        ]
    ).to_csv(LOG_FILE, index=False)

#Page Configuration
st.set_page_config(page_title="Oilfield Predictive Maintenance System", layout="wide")
#title
st.title("AI Based Predictive Maintenance")
#load saved model and scaler
model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_failure_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
# rul_model = joblib.load(os.path.join(BASE_DIR, "models", "rul_model.pkl"))
bearing_model = joblib.load(os.path.join(BASE_DIR, "models", "bearing_fault_model.pkl"))
bearing_encoder = joblib.load(os.path.join(BASE_DIR, "models", "bearing_label_encoder.pkl"))
explainer = shap.TreeExplainer(model)

#create placeholders
telemetry_placeholder = st.empty()
prediction_placeholder = st.empty()
alert_placeholder = st.empty()

#session state section
if "temperature_history" not in st.session_state:
    st.session_state.temperature_history = []

if "probability_history" not in st.session_state:
    st.session_state.probability_history = []
    
if "equipment_age" not in st.session_state:
    st.session_state.equipment_age = 0

st.session_state.equipment_age += 1
    
#Display Equipment Age
st.subheader("Equipment Aging")
st.metric("Equipment Age", f"{st.session_state.equipment_age:.0f} Hours")

#Calculate aging score
aging_score = max(0, 100 - (st.session_state.equipment_age/10))

#Aging score
if aging_score > 80:
    aging_status = "New"
elif aging_score > 60:
    aging_status = "Moderate wear"
elif aging_score > 30:
    aging_status = "Aged"
else:
    aging_status = "End of Life"
st.metric("Asset Aging Status", aging_status)

#adding gauge for asset health
fig_age = go.Figure(go.Indicator(
    mode="gauge+number",
    value=aging_score,
    title={'text': "Asset Health"},
    gauge={
        'axis': {'range': [0,100]},
        'steps': [
            {'range':[0,30], 'color':'red'},
            {'range':[30,60], 'color':'orange'},
            {'range':[60,100], 'color':'green'}
        ]
    }
))

st.plotly_chart(fig_age, width="stretch")

#Maintenance based on age
if aging_score < 30:
    error_status = "Asset replacement recommended."
elif aging_score < 60:
    error_status = "Major overhaul recommended"
else:
    error_status = "Asset is healthy"
st.metric("Maintenance Based on age", error_status )

# No while True for dashboard auto refresh
st_autorefresh(interval=4000, key="data_refresh")
#simulated telemetry

#sidebar for oil equipment type selection
equipment = st.sidebar.selectbox(
    "Equipment",
    [
        "Pump",
        "Compressor",
        "Drilling Motor",
        "Bearing Assembly"
    ]
)


if equipment == "Pump":
    telemetry = {
        "Type": 0,
        "Air temperature [K]": round(random.uniform(295,305),2),
        "Process temperature [K]": round(random.uniform(305,312),2),
        "Rotational speed [rpm]": random.randint(1200,1600),
        "Torque [Nm]": round(random.uniform(20,50),2),
        "Tool wear [min]": random.randint(0,100),

        "TWF": random.choices([0,1], weights=[98,2])[0],
        "HDF": random.choices([0,1], weights=[99,1])[0],
        "PWF": random.choices([0,1], weights=[99,1])[0],
        "OSF": random.choices([0,1], weights=[99,1])[0],
        "RNF": random.choices([0,1], weights=[99,1])[0]
    }
    st.header("Pump Monitoring")
    st.info("Check seals and impeller wear.")
    st.write("-Mechanical Seal")
    st.write("-Impeller")
    st.write("-Bearings")

elif equipment == "Compressor":
    telemetry = {
    "Type": 1,
    "Air temperature [K]": round(random.uniform(300,310),2),
    "Process temperature [K]": round(random.uniform(310,320),2),
    "Rotational speed [rpm]": random.randint(3000,6000),
    "Torque [Nm]": round(random.uniform(10,40),2),
    "Tool wear [min]": random.randint(0,150),

    "TWF": random.choices([0,1], weights=[97,3])[0],
    "HDF": random.choices([0,1], weights=[98,2])[0],
    "PWF": random.choices([0,1], weights=[98,2])[0],
    "OSF": random.choices([0,1], weights=[97,3])[0],
    "RNF": random.choices([0,1], weights=[99,1])[0]
}
    st.header("Compressor Monitoring system")
    st.info("Inspect cooling system and pressure value.")
    st.write("-Air Filter")
    st.write("-Valve kit")
    st.write("-Bearings Set")

elif equipment == "Drilling Motor":
    telemetry = {
    "Type": 2,
    "Air temperature [K]": round(random.uniform(305,315),2),
    "Process temperature [K]": round(random.uniform(315,325),2),
    "Rotational speed [rpm]": random.randint(100,500),
    "Torque [Nm]": round(random.uniform(50,120),2),
    "Tool wear [min]": random.randint(100,250),

    "TWF": random.choices([0,1], weights=[90,10])[0],
    "HDF": random.choices([0,1], weights=[92,8])[0],
    "PWF": random.choices([0,1], weights=[94,6])[0],
    "OSF": random.choices([0,1], weights=[92,8])[0],
    "RNF": random.choices([0,1], weights=[98,2])[0]
}
    st.header("Drilling Motor Monitoring")
    st.info("Check drill bit condition and torque fluctuations.")
    st.write("-Drilling Bit")
    st.write("-Rotor")
    st.write("-Stator")

else:
    telemetry = {
    "Type": 3,
    "Air temperature [K]": round(random.uniform(295,305),2),
    "Process temperature [K]": round(random.uniform(305,315),2),
    "Rotational speed [rpm]": random.randint(1500,2500),
    "Torque [Nm]": round(random.uniform(10,30),2),
    "Tool wear [min]": random.randint(0,80),

    "TWF": random.choices([0,1], weights=[99,1])[0],
    "HDF": random.choices([0,1], weights=[99,1])[0],
    "PWF": random.choices([0,1], weights=[99,1])[0],
    "OSF": random.choices([0,1], weights=[99,1])[0],
    "RNF": random.choices([0,1], weights=[99,1])[0]
}
    st.header("Bearing Assembly Monitoring")
    st.info("Inspect bearing lubrication and vibration levels.")
    st.write("-Bearing Kit")
    st.write("-Lubrication Pack")
    
#Equipment ranking dashboard
assets = pd.DataFrame({
    "Equipment": ["Pump", "Compressor", "Motor"],
    "Risk Score": [
        random.randint(10,90),
        random.randint(10,90),
        random.randint(10,90)
    ]
})

st.dataframe(
    assets.sort_values(
        by="Risk Score",
        ascending=False
    )
)


 
#convert to dataframe
df = pd.DataFrame([telemetry])

st.write("Current Columns:", list(df.columns))

#Scale Data
scaled_data = scaler.transform(df)
#Prediction
prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0][1]
    
rul_columns = [
    'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4',
    'sensor_5', 'sensor_6', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_10', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
    'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]

rul_input = pd.DataFrame(
    [[0] * len(rul_columns)],
    columns=rul_columns
)

predicted_rul = random.randint(120, 300)

#Calculate days remaining
estimated_days = max(1, int(predicted_rul/10))

#Calculate maintenance date
next_maintenance_date = (datetime.now() + timedelta(days=estimated_days))

bearing_sample = pd.DataFrame({
    "RMS": [round(random.uniform(0.05, 0.25), 4)],
    "Peak": [round(random.uniform(0.2, 1.5), 4)],
    "Kurtosis": [round(random.uniform(-1, 12), 4)],
    "Skewness": [round(random.uniform(-1, 2), 4)]
})

bearing_prediction = bearing_model.predict(bearing_sample)
bearing_condition = (bearing_encoder.inverse_transform(bearing_prediction))[0]

#displaying the maintenance data
st.subheader("Maintenance Scheduler")
st.metric("Days Until Service", estimated_days)
st.metric("Next maintenance Date", next_maintenance_date.strftime("%Y-%m-%d"))

#Maintenance recommendations
recommendations = []

if probability > 0.7:
    recommendations.append(
        "Inspect equipment immediately."
    )

if aging_score < 50:
    recommendations.append(
        "Plan overhaul or replacement."
    )

if bearing_condition != "Normal":
    recommendations.append(
        f"Check bearing: {bearing_condition}"
    )

for rec in recommendations:
    st.write("•", rec)

#Priority levels for maintenance
if estimated_days < 7:
    warning_status = "Urgent Maintenance required"
elif estimated_days < 30:
    warning_status = "Maintenance due soon"
else:
    warning_status = "Maintenance Scheduled Normal"
st.metric("Priority levels for maintenance", warning_status)

# Maintenance Cost Estimation
if equipment == "Pump":
    base_cost = 5000

elif equipment == "Compressor":
    base_cost = 12000

elif equipment == "Drilling Motor":
    base_cost = 20000

else:  # Bearing Assembly
    base_cost = 3000

risk_multiplier = 1 + (probability * 2)
cost = int(base_cost * risk_multiplier)

st.metric(
    "Estimated Maintenance Cost",
    f"${cost:,}"
)

st.subheader("Bearing Signal Features")
st.dataframe(bearing_sample)

shap_values = explainer.shap_values(scaled_data)
feature_names = df.columns.tolist()
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": abs(shap_values[0])
})
importance_df = importance_df.sort_values(
    by="Impact",
    ascending=False
)

top_features = importance_df.head(3)
    
health_placeholder = st.empty()
    
health_score = max(0, 100 - probability * 100)

st.subheader("Digital Twin")
    
if health_score > 80:
    st.success(" Healthy Equipment")
    twin_status = "Healthy"
elif health_score > 50:
    st.warning("Maintenance Recommended")
    twin_status = "Degrading"
else:
    st.error("Immediate Inspection Required")
    twin_status = "Critical"
    
st.metric("Digital Twin Status", twin_status)

st.metric("Predicted Remaining Useful Life", f"{predicted_rul:.0f} cycles")
    
st.subheader("AI Explanation")
for _, row in top_features.iterrows():
    st.write(
        f"{row['Feature']} (Impact: {row['Impact']:.4f})"
    )
    
st.session_state.temperature_history.append(
    telemetry["Process temperature [K]"]
)

st.session_state.probability_history.append(
    probability
)
    
#last 20 values
temperature_history = st.session_state.temperature_history[-20:]
probability_history = st.session_state.probability_history[-20:]
    
#Telemetry table
telemetry_placeholder.subheader("Live Telemetry")
telemetry_placeholder.dataframe(df)
#Prediction section
prediction_placeholder.subheader("Failure Prediction")
prediction_placeholder.metric(
    label="Failure Probality",
    value=f"{probability:2%}"
)
    
event_placeholder = st.empty()
    
st.metric(label = "Equipment Health Score", value=f"{health_score:.1f}%")
if os.path.exists(LOG_FILE):
    logs_df = pd.read_csv(LOG_FILE)
    critical_count = len(logs_df[logs_df["Status"] == "CRITICAL"])
    warning_count = len(logs_df[logs_df["Status"] == "WARNING"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Critical Events", critical_count)
    with col2:
        st.metric("Warning Events", warning_count)
else:
    logs_df = pd.DataFrame(
        columns=[
            "Timestamp",
            "Failure_Probability",
            "Status"
        ]
    )
    
#Alert section
if probability > 0.7:
    alert_placeholder.error("CRITICAL ALERT: Possible Equipment Failure!")
elif probability > 0.4:
    alert_placeholder.warning("WARNING: Abnormal Equipment Condition")
else:
    alert_placeholder.success("Equipment Operating Normally")
    

    
#charts
charts_data = pd.DataFrame({
    "Temperature": st.session_state.temperature_history[-20:],
    "Failure Probability": st.session_state.probability_history[-20:]
})
st.subheader("LIVE MONITORING CHARTS")
fig1 = px.line(charts_data, y="Temperature", title="Process Temperature Trend")
st.plotly_chart(fig1, width="stretch")
fig2 = px.line(charts_data, y="Failure Probability", title="Failure Probability Trend")
st.plotly_chart(fig2, width="stretch")
    
#failure probability gauge
fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value= probability * 100,
    title={'text': "Failure Risk (%)"},
    gauge={'axis': {'range': [0,100]},
           "steps": [
               {"range": [0,40]},
               {"range": [40,70]},
               {"range": [70,100]}]},
))
st.plotly_chart(fig, width="stretch")
    
#Failure probability trend chart
logs_df = pd.read_csv(LOG_FILE)

st.subheader("Failure Probability Trend")

if "Failure_Probability" in logs_df.columns:

    logs_df["Timestamp"] = pd.to_datetime(
        logs_df["Timestamp"]
    )

    st.line_chart(
        logs_df.set_index("Timestamp")[
            "Failure_Probability"
        ]
    )

else:
    st.error(
        f"Column not found. Available columns: {list(logs_df.columns)}"
    )

#Failure Probability Forecast
future_prob = []
current = probability
for i in range(10):
    current += random.uniform(-0.03,0.05)
    current = min(max(current, 0), 1)
    future_prob.append(current)
    
forecast_df = pd.DataFrame({
    "Step": range(1,11),
    "Predicted Failure Probability": future_prob
})

fig = px.line(
    forecast_df,
    x="Step",
    y="Predicted Failure Probability",
    title="Failure Risk Forecast"
)

st.plotly_chart(fig, width="stretch")

#Event history table
st.subheader("Recent Alerts")
st.dataframe(logs_df.tail(20))
    
#download report button
report_csv = logs_df.to_csv(index=False)
st.download_button(
    "Download Event Report",
    report_csv,
    "event_report.csv",
    "text/csv",
    key="download1"
)

maintenance_csv = logs_df.to_csv(index=False)
st.download_button(
    "Download Maintenance Report",
    maintenance_csv,
    "maintenance_report.csv",
    "text/csv",
    key="download2"
) 

#using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Failure Probability", f"{probability:.2%}")

with col2:
    st.metric("Equipment Health", f"{health_score:.1f}%")

with col3:
    st.metric("Predicted RUL", f"{predicted_rul:.0f} cycles")
    
#adding colours to the gauge
gauge={
    "axis": {"range": [0, 100]},
    "steps": [
        {"range": [0, 40], "color": "green"},
        {"range": [40, 70], "color": "yellow"},
        {"range": [70, 100], "color": "red"}
    ]
}

#Maintenance Recommendation Engine
st.subheader("Maintenance Recommendation")

if probability > 0.7 and aging_score < 40:
    st.error("Immediate shutdown and inspection recommended.")
    status = "CRITICAL"
    priority = "P1 - Immediate"
elif probability > 0.4:
    st.warning("Schedule maintenance within 24 hours.")
    status = "WARNING"
    priority = "P2 - Schedule soon"
else:
    st.success("No maintenance required.")
    status = "NORMAL"
    priority = "P3 - Monitor"
st.metric("Maintenance Priority", priority)


#RUL gauge chart
fig_rul = go.Figure(go.Indicator(
    mode="gauge+number",
    value=predicted_rul,
    title={'text': "Remaining Useful Life"},
    gauge={'axis': {'range': [0,300]}}
))

st.plotly_chart(fig_rul, width="stretch")

#Equipment Status Card
if predicted_rul < 50:
    st.error("Equipment Near End-of-Life")
elif predicted_rul < 100:
    st.warning("Maintenance Due Soon")
else:
    st.success("Equipment Life Healthy")
    
#Dashboard Auto-Refresh Timestamp
st.caption(
    f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

#Bearing Fault diagnosis
st.subheader("Bearing Fault Diagnosis")

st.metric("Bearing Condition", bearing_condition,)
fault_score ={"Normal": 0,"Outer": 40,"Ball": 70,"Inner": 100}
fig_bearing = go.Figure(go.Indicator(
    mode="gauge+number",
    value=fault_score[bearing_condition],
    title={'text': "Bearing Health"},
    gauge={
        'axis': {'range': [0,100]},
        'steps': [
            {'range':[0,40]},
            {'range':[40,70]},
            {'range':[70,100]}
        ]
    }
))
st.plotly_chart(fig_bearing, width="stretch")

if bearing_condition == "Normal":
    st.success("Bearing Healthy")

elif bearing_condition == "Inner":
    st.error("Inner Race Fault Detected")

elif bearing_condition == "Outer":
    st.warning("Outer Race Fault Detected")

else:
    st.error("Ball Bearing Fault Detected")
    
#bearing health condition
if bearing_condition == "Normal":
    bearing_health = 100
elif bearing_condition == "Outer":
    bearing_health = 70
elif bearing_condition == "Inner":
    bearing_health = 50
else:
    bearing_health = 30

st.metric("Bearing Health Score",f"{bearing_health}%")



#Live Alarm Counter
st.metric("Total Alarms Today", len(logs_df))

#adding a new warning or critical alert to events.csv
if status != "NORMAL":
    
    new_event = pd.DataFrame([{
        "Timestamp": datetime.now(),
        "Failure_Probability": probability,
        "Status": status
    }])

    if os.path.exists(LOG_FILE):
        old_events = pd.read_csv(LOG_FILE)
        updated_events = pd.concat(
            [old_events, new_event],
            ignore_index=True
        )
    else:
        updated_events = new_event

    updated_events.to_csv(LOG_FILE, index=False)
    
#creating a virtual representation of the equipment(Digital Twin)
st.write("Virtual Asset State")

twin_data = pd.DataFrame({
    "Parameter": [
        "Temperature",
        "RPM",
        "Torque",
        "Tool Wear"
    ],
    "Live Value": [
        telemetry["Process temperature [K]"],
        telemetry["Rotational speed [rpm]"],
        telemetry["Torque [Nm]"],
        telemetry["Tool wear [min]"]
    ]
})

st.dataframe(twin_data)

# Digital Twin Simulation
predicted_temp = telemetry["Process temperature [K]"] + random.uniform(-2, 2)
predicted_torque = telemetry["Torque [Nm]"] + random.uniform(-5, 5)

twin_health = round(
    100 - (
        abs(predicted_temp - telemetry["Process temperature [K]"]) * 2 +
        abs(predicted_torque - telemetry["Torque [Nm]"])
    ),
    2
)

twin_health = max(0, min(100, min(100,(twin_health*0.7)+(aging_score*0.3)))) ##connected aging to digital twin (sensor health, equipment health affects twin health and older assets naturally degrade)

# Asset risk score
risk_score = (
    probability * 40 +
    (100 - health_score) * 0.2 +
    (100 - twin_health) * 0.2 +
    (100 - aging_score) * 0.2
)

risk_score = round(risk_score, 2)

st.metric("Asset Risk Score", f"{risk_score}/100")

#Asset risk score
if risk_score < 30:
    st.success("Low Risk")
elif risk_score < 60:
    st.warning("Medium Risk")
else:
    st.error("High Risk")
    
#Equipment Ranking Dashboard
assets = pd.DataFrame({
    "Equipment": ["Pump", "Compressor", "Motor"],
    "Risk Score": [
        random.randint(10,90),
        random.randint(10,90),
        random.randint(10,90)
    ]
})

st.dataframe(
    assets.sort_values(
        by="Risk Score",
        ascending=False
    )
)

#Digital Digital Twin Metrics
st.subheader("Digital Twin")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Temperature", f"{telemetry['Process temperature [K]']:.1f} K")
with col2:
    st.metric("Twin Predicted Temperature",f"{predicted_temp:.1f} K")
with col3:
    st.metric("Twin Health Score", f"{twin_health:.1f}%")
    
#Add twin Status
if twin_health > 80:
    twin_status = "Healthy"
elif twin_health > 50:
    twin_status = "Warning"
else:
    twin_status = "Critical"
st.metric("Digital Twin Status", twin_status)

#adding a twin comparison chart
twin_df = pd.DataFrame({
    "Actual": [telemetry["Process temperature [K]"], telemetry["Torque [Nm]"]],
    "Predicted": [predicted_temp, predicted_torque]
    }, index=["temperature", "Torque"])
st.subheader("Digital Twin Comparison")
fig_twin = px.bar(twin_df, barmode = "group", title="Actual Vs Digital Twin prediction")
st.plotly_chart(fig_twin, width="stretch")

#Multi-Equipment Fleet Monitoring
fleet_df = pd.DataFrame({
    "Equipment": [
        "Pump-01",
        "Pump-02",
        "Compressor-01",
        "Drill Motor-01",
        "Bearing Unit-01"
    ],
    "Health Score": [
        random.randint(40,100),
        random.randint(40,100),
        random.randint(40,100),
        random.randint(40,100),
        random.randint(40,100)
    ]
})

st.subheader("Fleet Monitoring")
st.dataframe(fleet_df)

#Fleet Risk Ranking
fleet_df["Risk Score"] = 100 - fleet_df["Health Score"]

st.subheader("Highest Risk Assets")

st.dataframe(
    fleet_df.sort_values(
        by="Risk Score",
        ascending=False
    )
)

#Asset map Dashboard
fleet_df["Location"] = [
    "Well A",
    "Well B",
    "Well C",
    "Well D",
    "Well E"
]

#Predict Next Failure
predicted_failure_days = max(
    1,
    int(predicted_rul / 8)
)

st.metric(
    "Estimated Days to Failure",
    predicted_failure_days
)

#Equipment degradation curve(equipment lifecycle dashboard)
fig_age = px.line(
    x=list(range(10)),
    y=[100,95,90,84,79,72,65,58,48,40],
    title="Equipment Degradation Curve"
)

st.plotly_chart(fig_age, width="stretch")