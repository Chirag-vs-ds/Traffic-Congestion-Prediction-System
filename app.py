import streamlit as st
import pandas as pd

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_time_features, add_rain_feature
from src.label_creation import create_congestion_labels
from src.train_model import train_model
from src.predict_congestion import predict_and_explain
from src.traffic_volume_predictor import train_volume_model, predict_traffic_volume
from sklearn.preprocessing import LabelEncoder

st.title("🚦 Smart Traffic Congestion Prediction System")

# Load & train once
df = load_and_clean_data("Traffic_Volume.csv")
df = add_time_features(df)
df = add_rain_feature(df)
df = create_congestion_labels(df)

le = LabelEncoder()
df['congestion_encoded'] = le.fit_transform(df['congestion'])

congestion_model, _, _ = train_model(df)
volume_model = train_volume_model(df)

# ---- USER INPUTS ----
day = st.selectbox(
    "Select Day",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

hour = st.slider("Select Hour", 0, 23, 9)
rain = st.selectbox("Is it raining?", ["No","Yes"])
location = st.selectbox(
    "Location Type",
    ["college","it_park","metro","highway"]
)
holiday = st.selectbox("Holiday / Event?", ["No","Yes"])

# ---- CONVERSIONS ----
day_map = {
    "Monday":0,"Tuesday":1,"Wednesday":2,
    "Thursday":3,"Friday":4,"Saturday":5,"Sunday":6
}

location_map = {
    "college":1,"it_park":2,"metro":3,"highway":4
}

day_of_week = day_map[day]
rain_flag = 1 if rain == "Yes" else 0
location_type = location_map[location]
holiday_flag = 1 if holiday == "Yes" else 0

# ---- PREDICTION ----
if st.button("Predict Congestion"):
    predicted_volume = predict_traffic_volume(
        volume_model, hour, day_of_week, rain_flag
    )

    prediction, reason = predict_and_explain(
        congestion_model, le,
        hour, day_of_week, rain_flag, predicted_volume
    )

    emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}

    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Traffic Volume:** {predicted_volume}")
    st.write(f"**Congestion Level:** {emoji[prediction]} {prediction}")
    st.write(f"**Reason:** {reason}")

    if prediction == "High":
        st.error("⚠️ Advisory: Avoid travel or choose alternate route")
    elif prediction == "Medium":
        st.warning("⏳ Advisory: Expect delays")
    else:
        st.success("✅ Advisory: Safe to travel")
