"""
Evaluate ensemble R2 on the FULL base_dataset
"""

import numpy as np
import pandas as pd

from joblib import load
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

#############################################
# Paths
#############################################

BASE_DATASET_PATH = "../data/base_dataset.csv"

BEST_MODEL_PATH = "../models/best_model.joblib"
ERROR_MODEL_PATH = "../models/error_predict_model.joblib"

ALT_MODEL_PATH = "../models/alternative_model.keras"
ALT_NUM_SCALER_PATH = "../models/alt_num_scaler.joblib"
ALT_Y_SCALER_PATH = "../models/alt_y_scaler.joblib"
ALT_POSTAL_MAPPING_PATH = "../models/alt_postal_mapping.joblib"
ALT_OTHER_COLS_PATH = "../models/alt_other_cols.joblib"

#############################################
# Custom metric for loading FNN
#############################################

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

#############################################
# Load the FULL dataset
#############################################

df_full = pd.read_csv(BASE_DATASET_PATH, delimiter=",")

y = df_full["price"].values
X = df_full.drop(columns=["price"])

#############################################
# Load models
#############################################

best_model = load(BEST_MODEL_PATH)
error_model = load(ERROR_MODEL_PATH)

alternative_model = load_model(
    ALT_MODEL_PATH,
    custom_objects={"r2_metric": r2_metric}
)

alt_num_scaler = load(ALT_NUM_SCALER_PATH)
alt_y_scaler = load(ALT_Y_SCALER_PATH)
postal_to_index = load(ALT_POSTAL_MAPPING_PATH)
alt_other_cols = load(ALT_OTHER_COLS_PATH)

#############################################
# Base model predictions (full dataset)
#############################################

y_pred_best_full = best_model.predict(X)

#############################################
# Prepare data for error_model
#############################################

X_loss_full = X.copy()
X_loss_full["y_pred_oof"] = y_pred_best_full

predicted_loss_full = error_model.predict(X_loss_full)

#############################################
# Split full dataset by predicted difficulty
#############################################

threshold = np.quantile(predicted_loss_full, 0.7)

mask_difficult = predicted_loss_full >= threshold
mask_easy = ~mask_difficult

print(f"Total objects    : {len(y)}")
print(f"Easy objects     : {mask_easy.sum()}")
print(f"Difficult objects: {mask_difficult.sum()}")

#############################################
# Prepare inputs for alternative_model
#############################################

df_fnn = X.copy()

# Drop the same columns as in training
cols_to_drop = []
for col in ["cadastral_income", "loss", "y_pred_oof", "y_true"]:
    if col in df_fnn.columns:
        cols_to_drop.append(col)

if cols_to_drop:
    df_fnn = df_fnn.drop(columns=cols_to_drop)

# Postal
postal_codes = df_fnn["postal_code"].astype(int).values
postal_idx = np.array([postal_to_index.get(p, 0) for p in postal_codes], dtype="int32")

# Numeric
num_features = ["area", "build_year", "primary_energy_consumption"]
X_num_full = df_fnn[num_features].values

# Other features
X_other_full = df_fnn[alt_other_cols].values.astype("float32")

# Select only difficult objects
postal_diff = postal_idx[mask_difficult]
X_num_diff = X_num_full[mask_difficult]
X_other_diff = X_other_full[mask_difficult]

# Scale numeric using saved scaler
X_num_diff_scaled = alt_num_scaler.transform(X_num_diff)

#############################################
# Alternative model predictions
#############################################

y_pred_alt_scaled = alternative_model.predict(
    [postal_diff, X_num_diff_scaled, X_other_diff],
    verbose=0
)

y_pred_alt = alt_y_scaler.inverse_transform(y_pred_alt_scaled).ravel()

#############################################
# Build ensemble prediction for full dataset
#############################################

y_pred_ensemble = y_pred_best_full.copy()
y_pred_ensemble[mask_difficult] = y_pred_alt

#############################################
# Evaluate R? on FULL DATASET
#############################################

r2_base_full = r2_score(y, y_pred_best_full)
r2_ensemble_full = r2_score(y, y_pred_ensemble)

print("\n========== R2 on FULL DATASET ==========")
print(f"Base model R2      : {r2_base_full:.3f}")
print(f"Ensemble R2 (70/30): {r2_ensemble_full:.3f}")
print("========================================\n")
