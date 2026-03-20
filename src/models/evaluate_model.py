import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred}).to_csv("data/processed/predictions.csv", index=False)

scores = {
    "MSE": mean_squared_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred)
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(scores)
