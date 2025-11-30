"""
Ensemble of saved models:
- best_model         : GradientBoostingRegressor (base model, price)
- error_predict_model: XGBRegressor (predicts loss)
- alternative_model  : Keras FNN (better on difficult objects)

We evaluate ensemble R2 on the original task (price) on meta_test part
of base_dataset.csv.
"""

import numpy as np
import pandas as pd

from pathlib import Path
import sys
import os

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.models import load_model

#############################################
#           Paths and config                #
#############################################

BASE_DATASET_PATH = "../data/base_dataset.csv"

BEST_MODEL_PATH = "../models/best_model.joblib"
ERROR_MODEL_PATH = "../models/error_predict_model.joblib"

ALT_MODEL_PATH = "../models/alternative_model.keras"
ALT_NUM_SCALER_PATH = "../models/alt_num_scaler.joblib"
ALT_Y_SCALER_PATH = "../models/alt_y_scaler.joblib"
ALT_POSTAL_MAPPING_PATH = "../models/alt_postal_mapping.joblib"
ALT_OTHER_COLS_PATH = "../models/alt_other_cols.joblib"

# import RANDOM_STATE from config
config_dir = Path(__file__).resolve().parents[1] / "config"
sys.path.append(str(config_dir))
from settings import RANDOM_STATE

#############################################
#         Load base dataset                 #
#############################################

df_base = pd.read_csv(BASE_DATASET_PATH, delimiter=",")

# Target and features for original task
y = df_base["price"].values
X = df_base.drop(columns=["price"])

# Reproduce the same split as for best_model training
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

#############################################
#         Load saved models                 #
#############################################

# 1) Base model
best_model = load(BEST_MODEL_PATH)

# 2) Error-predict model (XGBRegressor)
error_model = load(ERROR_MODEL_PATH)

# 3) Alternative FNN model + its preprocessing
# r2_metric must be defined the same way as in training script
import tensorflow.keras.backend as K

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

alternative_model = load_model(
    ALT_MODEL_PATH,
    custom_objects={"r2_metric": r2_metric}
)

alt_num_scaler = load(ALT_NUM_SCALER_PATH)
alt_y_scaler = load(ALT_Y_SCALER_PATH)
postal_to_index = load(ALT_POSTAL_MAPPING_PATH)  # dict: postal_code -> index
alt_other_cols = load(ALT_OTHER_COLS_PATH)      # list of columns for "other_input"

#############################################
#   Step 1: base model predictions          #
#############################################

# Base model predictions on meta_test (original task)
y_pred_best_test = best_model.predict(X_meta_test)

#############################################
#   Step 2: predicted loss for meta_test    #
#############################################

# Error model was trained on loss_dataset with features:
#   all original features + column "y_pred_oof"
# For new data we use base model predictions instead of y_pred_oof.
X_loss_test = X_meta_test.copy()
X_loss_test["y_pred_oof"] = y_pred_best_test

# Predict loss for each row in meta_test
predicted_loss_test = error_model.predict(X_loss_test)

#############################################
#   Step 3: split into easy / difficult     #
#############################################

# 30% of objects with the highest predicted loss are "difficult"
threshold = np.quantile(predicted_loss_test, 0.70)  # 70th percentile
mask_difficult = predicted_loss_test >= threshold
mask_easy = ~mask_difficult

print(f"Meta_test size: {len(y_meta_test)}")
print(f"Easy objects    : {mask_easy.sum()}")
print(f"Difficult objects: {mask_difficult.sum()}")

#############################################
#   Step 4: alternative model inputs        #
#############################################

# Alternative model was trained on a dataset where:
# - df.drop(columns=['cadastral_income', 'loss', 'y_pred_oof'])
# - y_true was the target (standardized by alt_y_scaler)
# - Features are split into:
#     postal_code -> Embedding
#     num_features -> StandardScaler
#     other_cols -> fed as float32

# We need to build the same inputs for meta_test (only for difficult objects).

# Take test subset (features only)
df_test = X_meta_test.copy()

# Drop columns that were dropped in alternative model script
cols_to_drop = []
if "cadastral_income" in df_test.columns:
    cols_to_drop.append("cadastral_income")
# In base_dataset there is no 'loss' or 'y_pred_oof', but keep for safety:
for col in ["loss", "y_pred_oof"]:
    if col in df_test.columns:
        cols_to_drop.append(col)

if cols_to_drop:
    df_test = df_test.drop(columns=cols_to_drop)

# Postal code column
postal_codes_test = df_test["postal_code"].astype(int).values
# Map postal_code -> index (unknown -> 0)
postal_indexes_test = np.array(
    [postal_to_index.get(code, 0) for code in postal_codes_test],
    dtype="int32"
)

# Numeric features (same as in alternative model script)
num_features = ["area", "build_year", "primary_energy_consumption"]
X_num_test = df_test[num_features].values

# Other features in the same order as during training
X_other_test = df_test[alt_other_cols].values

# Use only "difficult" objects
postal_difficult = postal_indexes_test[mask_difficult]
X_num_difficult = X_num_test[mask_difficult]
X_other_difficult = X_other_test[mask_difficult]

# Scale numeric features with saved scaler
X_num_difficult_scaled = alt_num_scaler.transform(X_num_difficult)

# Convert other features to float32 (as in training)
X_other_difficult = X_other_difficult.astype("float32")

#############################################
#   Step 5: predictions from alternative    #
#############################################

# Predict scaled target
y_pred_alt_scaled = alternative_model.predict(
    [postal_difficult, X_num_difficult_scaled, X_other_difficult],
    verbose=0
)

# Inverse transform to original price scale
y_pred_alt = alt_y_scaler.inverse_transform(y_pred_alt_scaled).ravel()

#############################################
#   Step 6: build ensemble predictions      #
#############################################

# Start from base model predictions
y_pred_ensemble = y_pred_best_test.copy()

# Replace predictions on difficult objects with alternative model predictions
y_pred_ensemble[mask_difficult] = y_pred_alt

#############################################
#   Step 7: evaluate R2 on meta_test        #
#############################################

r2_base = r2_score(y_meta_test, y_pred_best_test)
r2_ensemble = r2_score(y_meta_test, y_pred_ensemble)

print(f"\nBase model R2 on meta_test      : {r2_base:.3f}")
print(f"Ensemble R2 on meta_test (70/30): {r2_ensemble:.3f}")

