import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/sample_data.csv")

# Features
X = df[['duration', 'src_bytes', 'dst_bytes']]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(X_scaled)

# Line plot (normal data)
plt.figure()
plt.plot(df.index, df['src_bytes'])

# Highlight anomalies
anomalies = df[df['anomaly'] == -1]
plt.scatter(anomalies.index, anomalies['src_bytes'])

# Labels
plt.title("Network Traffic Anomaly Detection (Line Graph)")
plt.xlabel("Time Index")
plt.ylabel("Source Bytes")

# Save
plt.savefig("outputs/anomaly_plot.png")
plt.close()

print("Line graph saved successfully")