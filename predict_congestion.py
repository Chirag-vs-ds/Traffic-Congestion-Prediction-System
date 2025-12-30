def explain_prediction(hour, day_of_week, rain_flag, traffic_volume, prediction):
    reasons = []

    # Peak hour logic
    if hour in [8, 9, 10, 17, 18, 19]:
        reasons.append("peak hour")

    # Rain logic
    if rain_flag == 1:
        reasons.append("rain")

    # High volume logic
    if traffic_volume > 3000:
        reasons.append("high volume")

    if not reasons:
        reasons.append("normal traffic conditions")

    return " + ".join(reasons)


def predict_and_explain(model, le, hour, day_of_week, rain_flag, traffic_volume):
    import pandas as pd

    # prepare input
    input_df = pd.DataFrame([{
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "traffic_volume": traffic_volume,
        "rain_flag": rain_flag
    }])

    # prediction
    pred_encoded = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    # explanation
    reason = explain_prediction(
        hour, day_of_week, rain_flag, traffic_volume, pred_label
    )

    return pred_label, reason
