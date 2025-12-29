from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import add_time_features, add_rain_feature
from src.label_creation import create_congestion_labels
from src.train_model import train_model
from src.evaluate_model import evaluate
from sklearn.preprocessing import LabelEncoder
from src.predict_congestion import predict_and_explain
from src.traffic_volume_predictor import train_volume_model, predict_traffic_volume






df = load_and_clean_data(r"C:/Users/chira/Documents/python_programing/Traffic Congestion Prediction System/Traffic_Volume.csv")

df = add_time_features(df)
df = add_rain_feature(df)
# ------------------------------
# TRAIN TRAFFIC VOLUME MODEL
# ------------------------------
volume_model = train_volume_model(df)


df = create_congestion_labels(df)

le = LabelEncoder()
df['congestion_encoded'] = le.fit_transform(df['congestion'])

model, X_test, y_test = train_model(df)
evaluate(model, X_test, y_test)

from src.predict_congestion import predict_and_explain

from src.predict_congestion import predict_and_explain

print("\n====== Traffic Congestion Prediction System ======\n")

# ---- USER INPUTS ----
day_name = input("Enter day (Monday, Tuesday, ...): ").strip()

hour = int(input("Enter hour (0–23): "))

rain_input = input("Is it raining? (yes/no): ").strip().lower()
location = input(
    "Enter location type (college / it_park / metro / highway): "
).strip().lower()
holiday_input = input("Is today a holiday/event? (yes/no): ").strip().lower()
is_holiday = 1 if holiday_input == "yes" else 0

location_map = {
    "college": 1,
    "it_park": 2,
    "metro": 3,
    "highway": 4
}

location_type = location_map.get(location)

if location_type is None:
    print("Invalid location type")
    exit()

rain_flag = 1 if rain_input == "yes" else 0
# ------------------------------
# DAY NAME TO NUMBER (FIX)
# ------------------------------
day_map = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6
}

day_of_week = day_map.get(day_name.lower())

if day_of_week is None:
    print("Invalid day entered. Please enter a valid day.")
    exit()


# Predict traffic volume automatically
predicted_volume = predict_traffic_volume(
    volume_model,
    hour,
    day_of_week,
    rain_flag
)




# ---- PREDICTION ----
prediction, reason = predict_and_explain(
    model, le, hour, day_of_week, rain_flag, predicted_volume
)


emoji_map = {
    "Low": "🟢",
    "Medium": "🟡",
    "High": "🔴"
}

# ---- OUTPUT ----
print("\n--- Prediction Result ---")
print(f"Time: {day_name} {hour}:00")
print(f"Rain: {'Yes' if rain_flag == 1 else 'No'}")
print(f"Output: {emoji_map[prediction]} {prediction} Congestion")
print(f"Reason: {reason}")

print("\nSmart Traffic Congestion Prediction completed successfully")
