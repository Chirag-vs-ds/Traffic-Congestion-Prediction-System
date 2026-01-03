🚦 Traffic Congestion Prediction System


The Traffic Congestion Prediction System is a data science and machine learning project that predicts traffic congestion levels and traffic volume based on time-related and weather-based factors.
The system helps in traffic planning, congestion management, and decision-making by providing real-time predictions through a web interface.

🎯 Objectives

Predict traffic congestion level (Low / Medium / High)

Predict traffic volume automatically

Analyze how time, day, and weather conditions affect traffic

Provide a user-friendly Streamlit web application

Deploy the solution on the cloud for real-world accessibility

🧠 Problem Statement

Urban areas suffer from frequent traffic congestion, leading to:

Increased travel time

Fuel wastage

Environmental pollution

Poor emergency response times

This project aims to predict congestion in advance so that authorities and users can take preventive actions.

⚙️ Tech Stack

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Machine Learning: Classification & Regression models

Web Framework: Streamlit

Deployment: Streamlit Cloud

🏗️ Project Architecture
Traffic-Congestion-Prediction-System/
│
├── app.py                     # Streamlit application
├── data_preprocessing.py      # Data cleaning & preparation
├── feature_engineering.py     # Feature creation
├── label_creation.py          # Congestion label logic
├── train_model.py             # ML model training
├── predict_congestion.py      # Prediction & explanation
├── traffic_volume_predictor.py# Traffic volume prediction
├── Traffic_Volume.csv         # Dataset
├── requirements.txt           # Dependencies
└── README.md

🔍 How It Works

Load and clean traffic dataset

Engineer time and weather features

Create congestion labels

Train ML models

Take user input via Streamlit UI

Predict:

Traffic congestion level

Traffic volume

Display results with explanations

🧪 Sample Output
Time: Monday 9 AM
Rain: Yes
Predicted Traffic Volume: 820 vehicles
Congestion Level: High
Reason: Peak hour + rain + high traffic volume

🌍 Real-World Use Cases

Smart city traffic management

Navigation and route optimization

Emergency vehicle planning

Urban infrastructure planning

Traffic police decision support

🚀 Live Demo

👉 Streamlit App Link: https://traffic-congestion-prediction-system-w2pmrnazztxatxztzmz2sq.streamlit.app/

🧑‍💻 How to Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Future Enhancements

Live traffic data integration (Google Maps API)

Map-based visualization

Deep learning models (LSTM)

Mobile app integration

