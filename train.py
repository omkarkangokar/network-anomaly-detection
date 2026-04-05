import pandas as pd
from app.model import AnomalyModel

df = pd.read_csv("data/sample_data.csv")
X = df.values

model = AnomalyModel()
model.train(X)
model.save()
