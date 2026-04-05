import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyModel:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self):
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")

    def load(self):
        self.model = joblib.load("model.pkl")
        self.scaler = joblib.load("scaler.pkl")
