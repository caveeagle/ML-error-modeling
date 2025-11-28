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
    shuffle=True,
    random_state=RANDOM_STATE
)

#############################################
#############################################

MODEL_NUM = 1  # Choose the model

#############################################


if(MODEL_NUM==1):
    
    model_name = 'Gradient Boosting'
    
    best_model = GradientBoostingRegressor(
        n_estimators=600,        
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=50,
        subsample=0.9,
        random_state=RANDOM_STATE
    )

#############################################

best_model.fit(X_train, y_train)

best_pred  = best_model.predict(X_test)

#############################################

best_r2  = r2_score(y_test, best_pred)

best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))

best_mae = mean_absolute_error(y_test, best_pred)

print(f"{model_name} - R2: {best_r2:.4f}, RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}")

#############################################

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")


print('\nTask completed!\n')

