# Network Traffic Anomaly Detection

This project detects anomalies in network traffic using machine learning.

## Features
- Isolation Forest-based anomaly detection
- Real-time prediction using FastAPI
- Scalable pipeline design
- Docker support

## Tech Stack
- Python, Scikit-learn
- FastAPI
- Docker

## Run Locally
```bash
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
