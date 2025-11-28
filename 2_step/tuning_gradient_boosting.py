"""
classical gradient boosting
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time
import logging

start_time = time.perf_counter()
#############################################

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler('./log_boosting.txt')
logger.addHandler(file_handler)

logger.info('Begin to work...\n')

#############################################
# 1. Load the dataset
#############################################

dataset_filename = '../data/loss_dataset.csv'

loss_dataset = pd.read_csv(dataset_filename, delimiter=',')

# Target
y_loss = loss_dataset["loss"]

# Features: exclude target and the data-leak column
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
# 3. Cross-validation scheme
#############################################

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

#############################################
# 4. Stage 1 GridSearch: max_depth 
#############################################

gb_stage1 = GradientBoostingRegressor(
    learning_rate=0.05,     # fixed at stage 1
    n_estimators=600,
    subsample=0.9,
    random_state=RANDOM_STATE
)

param_grid_stage1 = {
    "max_depth": [3, 5, 6]
}

grid_stage1 = GridSearchCV(
    estimator=gb_stage1,
    param_grid=param_grid_stage1,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_stage1.fit(X_train, y_train)

logger.info(f"Best parameters (tree structure): {grid_stage1.best_params_}")
logger.info(f"CV RMSE: {np.sqrt(-grid_stage1.best_score_):.4f}")


best_max_depth = grid_stage1.best_params_["max_depth"]

#############################################
# 5. Stage 2 GridSearch: learning_rate + n_estimators
#############################################

gb_stage2 = GradientBoostingRegressor(
    max_depth=best_max_depth,
    subsample=0.9,
    random_state=RANDOM_STATE
)

param_grid_stage2 = {
    "learning_rate": [0.03, 0.05, 0.08],
    "n_estimators": [600, 800, 1000, 1200, 1500]
}

grid_stage2 = GridSearchCV(
    estimator=gb_stage2,
    param_grid=param_grid_stage2,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid_stage2.fit(X_train, y_train)

logger.info(f"Best parameters (LR + estimators): {grid_stage2.best_params_}")
logger.info(f"CV RMSE: {np.sqrt(-grid_stage2.best_score_):.4f}")


#############################################
# 6. Final model and TEST evaluation
#############################################

final_params = {
    **grid_stage1.best_params_,
    **grid_stage2.best_params_
}

final_model = GradientBoostingRegressor(
    subsample=0.9,
    random_state=RANDOM_STATE,
    **final_params
)

# Train final model
final_model.fit(X_train, y_train)

# Predict on test set
y_test_pred = final_model.predict(X_test)

# Metrics
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
