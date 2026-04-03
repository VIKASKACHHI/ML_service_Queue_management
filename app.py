from fastapi import FastAPI
from joblib import load
import pandas as pd

app = FastAPI()

# Load model once
bundle = load("models/grokDataModel.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

@app.post("/predict")
def predict(data: dict):

    X_input = pd.DataFrame([data], columns=FEATURES)

    prediction = model.predict(X_input)[0]

    return {
        "predicted_wait_time_minutes": round(float(prediction), 2)
    }

@app.get("/")
def health_check():
    return {"status": "ML service running"}



import uvicorn
import os

if __name__ == "__main__":
    # Render provides a PORT environment variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
