import numpy as np
from joblib import load

# Load trained model
model = load("models/wait_time_model.pkl")

def predict_wait_time(input_data: dict):
    """
    input_data example:
    {
        "queue_position_at_booking": 2,
        "Department": 4,
        "hour_of_day": 9,
        "day_of_week": 0,
        "avg_service_time_ward": 18.9,
        "priority_count_ahead": 1
    }
    """

    X = np.array([[
        input_data["queue_position_at_booking"],
        input_data["Department"],
        input_data["hour_of_day"],
        input_data["day_of_week"],
        input_data["avg_service_time_ward"],
        input_data["priority_count_ahead"]
    ]])

    prediction = model.predict(X)[0]
    return round(float(prediction), 2)


# Example test
if __name__ == "__main__":
    sample_input = {
        "queue_position_at_booking": 2,
        "Department": 4,
        "hour_of_day": 9,
        "day_of_week": 0,
        "avg_service_time_ward": 18.9,
        "priority_count_ahead": 1
    }

    print("Predicted wait time:", predict_wait_time(sample_input), "minutes")
    