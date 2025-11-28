"""
Linear regression (LASSO model) for estimation
"""
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#############################################

dataset_filename = '../data/loss_dataset.csv'

loss_dataset = pd.read_csv(dataset_filename, delimiter=',')

#############################################

###        Import global variables        ###

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

# Separate features and target
X_loss = loss_dataset.drop(columns=["loss", "y_true", "y_pred_oof"])
y_loss = loss_dataset["loss"]

# Build a pipeline: scaling + Lasso
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.01))  # alpha controls strength of regularization
])

# Fit model
lasso_model.fit(X_loss, y_loss)

# Extract coefficients
coef = lasso_model.named_steps["lasso"].coef_
features = X_loss.columns

coeffs = pd.Series(coef, index=features).sort_values(ascending=False)
coeffs = coeffs.astype(int)

print(coeffs)


#############################################

print('\nTask completed!')

#############################################
