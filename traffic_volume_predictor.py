import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def train_volume_model(df):
    """
    Trains a model to predict traffic volume
    """
    X = df[['hour', 'day_of_week', 'is_weekend', 'rain_flag']]
    y = df['traffic_volume']

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    model.fit(X, y)

    return model


def predict_traffic_volume(model, hour, day_of_week, rain_flag):
    """
    Predicts traffic volume for given conditions
    """
    input_df = pd.DataFrame([{
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': 1 if day_of_week >= 5 else 0,
        'rain_flag': rain_flag
    }])

    predicted_volume = int(model.predict(input_df)[0])
    return predicted_volume
