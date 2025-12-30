def create_congestion_labels(df):
    low = df['traffic_volume'].quantile(0.33)
    high = df['traffic_volume'].quantile(0.66)

    def label(v):
        if v <= low:
            return "Low"
        elif v <= high:
            return "Medium"
        else:
            return "High"

    df['congestion'] = df['traffic_volume'].apply(label)
    return df
