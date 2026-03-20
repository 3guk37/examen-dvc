import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").squeeze()

with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
