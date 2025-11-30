"""
First step for Error modeling (see Roadmap)
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.base import clone

#############################################

dataset_filename = '../data/base_dataset.csv'

df = pd.read_csv(dataset_filename, delimiter=',')

#############################################

###        Import global variables        ###

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

print('Begin to work')

###          Split dataset                ###

X = df.drop("price", axis=1)

y = df["price"]

X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

#############################################

###  Best Model  ###

model = GradientBoostingRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    random_state=RANDOM_STATE)

# ----- 1. KFold on meta_train -----
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE  # same as train_test_split if you want consistency
)

# Array for OOF predictions (Out-Of-Fold predictions)
y_pred_oof = np.zeros(len(y_meta_train))

# ----- 2. Loop through the folds -----
i = 1
for train_idx, val_idx in kf.split(X_meta_train):
    
    X_fold_train = X_meta_train.iloc[train_idx]
    X_fold_val   = X_meta_train.iloc[val_idx]
    y_fold_train = y_meta_train.iloc[train_idx]
    y_fold_val   = y_meta_train.iloc[val_idx]

    # Clone the model so each fold is trained from scratch
    model_fold = clone(model)
    model_fold.fit(X_fold_train, y_fold_train)

    # Predict on the validation part of the fold
    y_pred_oof[val_idx] = model_fold.predict(X_fold_val)
    print(f'Fold {i} completed')
    i+=1
    
# ----- 3. Compute loss for each row of meta_train -----
# Using absolute error; It's AE, like MAE
loss = np.abs(y_meta_train.to_numpy() - y_pred_oof)

# ----- 4. Build loss_dataset -----
loss_dataset = X_meta_train.copy()
loss_dataset["y_true"]      = y_meta_train.values      # original price
loss_dataset["y_pred_oof"]  = y_pred_oof               # OOF predictions of the base model
loss_dataset["loss"]        = loss                     # error (target for the error model)

# Save dataset
if(0):
    filename_loss = '../data/loss_dataset.csv'
    loss_dataset.to_csv(filename_loss, sep=',', index=False)

if(1):
    
    # Train final model on full meta_train
    final_model = clone(model)
    final_model.fit(X_meta_train, y_meta_train)
    
    # Save it
    from joblib import dump
    dump(final_model, '../models/best_model.joblib')
    
    print("Final model saved!")
#############################################

print('\nTask completed!')

#############################################
