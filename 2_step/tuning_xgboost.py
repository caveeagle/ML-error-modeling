"""
XGBoost model
"""
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time
import logging

start_time = time.perf_counter()

#############################################

dataset_filename = '../data/loss_dataset.csv'

loss_dataset = pd.read_csv(dataset_filename, delimiter=',')

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler('./log_xgboost.txt')
logger.addHandler(file_handler)

logger.info('Begin to work...\n')

#############################################

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

y_loss = loss_dataset["loss"]
X_loss = loss_dataset.drop(columns=["y_true", "loss"])

#############################################
# 2. Train / test split
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X_loss,
    y_loss,
    test_size=0.2,
    shuffle=True,
    random_state=RANDOM_STATE
)

#############################################
# 3. Base model
#############################################

xgb_base = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1200,
    learning_rate=0.05,
    subsample=0.8,
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

#############################################
# 4. Hyperparameter search space
#############################################

param_dist = {
    "max_depth": [3, 4, 5, 6, 7],
    "min_child_weight": [1, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.5, 1, 2, 5],
    "reg_lambda": [0.5, 1, 2, 5],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1],
}

ITERATIONS = 4 # 1200

#############################################
# 5. RandomizedSearchCV (tuning on train)
#############################################

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=ITERATIONS,                         
    scoring="neg_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=RANDOM_STATE
)

random_search.fit(X_train, y_train)

logger.info(f"Best params: {random_search.best_params_}")
logger.info(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")

#############################################
# 6. Final model with best params
#############################################

best_params = random_search.best_params_

xgb_best = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    **best_params
)

xgb_best.fit(X_train, y_train)

y_test_pred = xgb_best.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae  = mean_absolute_error(y_test, y_test_pred)
test_r2   = r2_score(y_test, y_test_pred)

logger.info("\n=== TEST METRICS ===")
logger.info(f"RMSE: {test_rmse:.4f}")
logger.info(f"MAE : {test_mae:.4f}")
logger.info(f"R2  : {test_r2:.4f}")

end_time = time.perf_counter()
elapsed_time = end_time - start_time
logger.info(f"\nTimer: {elapsed_time:.0f} sec\n")


logger.info('\nTask completed!')

if(0):
    from service_functions import send_telegramm_message
    send_telegramm_message("Job completed")
