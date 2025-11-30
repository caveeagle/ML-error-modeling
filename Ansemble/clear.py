"""
Evaluate best_model R2 on FULL base_dataset
"""

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import r2_score

BASE_DATASET_PATH = "../data/base_dataset.csv"
BEST_MODEL_PATH = "../models/best_model.joblib"

df = pd.read_csv(BASE_DATASET_PATH, delimiter=",")

y = df["price"].values
X = df.drop(columns=["price"])

best_model = load(BEST_MODEL_PATH)

y_pred = best_model.predict(X)

r2 = r2_score(y, y_pred)

print(f"\nBest model R2 on FULL dataset: {r2:.4f}\n")
