"""
compare models (by hand)
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

### Models:
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import time
start_time = time.perf_counter()
#############################################

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

dataset_filename = '../data/loss_dataset.csv'

loss_dataset = pd.read_csv(dataset_filename, delimiter=',')

y_loss = loss_dataset["loss"]

X_loss = loss_dataset.drop(columns=["y_true", "loss"])

X_train, X_test, y_train, y_test = train_test_split(
    X_loss,
    y_loss,
    test_size=0.2,
    random_state=RANDOM_STATE
)

#############################################
#############################################

model_name = 'XGBoost'

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.05,
    subsample=0.7,
    reg_lambda=5,
    max_depth=7,
    gamma = 0.5,
    colsample_bytree=0.6,
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# XGBoost - R2: 0.336

#############################################

model.fit(X_train, y_train)

y_pred  = model.predict(X_test)

#############################################

best_r2  = r2_score(y_test, y_pred)

best_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

best_mae = mean_absolute_error(y_test, y_pred)

print(f"{model_name} - R2: {best_r2:.3f}")

#############################################

if(1):

    from joblib import dump
    
    dump(model, "../models/error_predict_model.joblib")


end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")


print('\nTask completed!\n')

