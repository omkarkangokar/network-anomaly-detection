from fastapi import FastAPI
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Network Anomaly Detection API"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([data["features"]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return {"anomaly": int(prediction[0])}
