def add_time_features(df):
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def add_rain_feature(df):
    df['rain_flag'] = df['weather_main'].apply(
        lambda x: 1 if x in ['Rain','Snow','Drizzle'] else 0
    )
    return df
def add_location_feature(df):
    location_map = {
        "college": 1,
        "it_park": 2,
        "metro": 3,
        "highway": 4
    }
    df['location_type'] = df['location_type'].map(location_map)
    return df
