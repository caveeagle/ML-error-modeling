import pandas as pd
import numpy as np
import random

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

#############################################

dataset_filename = '../data/difficult_dataset.csv'

df = pd.read_csv(dataset_filename, delimiter=',')

#############################################

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

# Target variable
y = df["y_true"].values.reshape(-1, 1)  

# Remove unnecessary columns
df = df.drop(columns=['cadastral_income','loss','y_pred_oof'])  

# Embedding column
postal = df["postal_code"].astype(int).values

# Numeric features
num_features = ["area", "build_year", "primary_energy_consumption"]
X_num = df[num_features]

# Other remaining features
other_cols = [c for c in df.columns if c not in num_features + ["postal_code"]]
X_other = df[other_cols]

############################################

### ===== Train / Test Split (80% / 20%) ===== ###

X_train_num, X_test_num, \
X_train_postal, X_test_postal, \
X_train_other, X_test_other, \
y_train, y_test = train_test_split(
    X_num, postal, X_other, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

############################################

###  Scaling ###

num_scaler = StandardScaler()

X_train_num = num_scaler.fit_transform(X_train_num)
X_test_num = num_scaler.transform(X_test_num)

y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

#############################################

###  Encode postal_code to continuous range 0..n_categories-1
unique_postals = np.sort(np.unique(X_train_postal))  # Only from train !
postal_to_index = {v: i for i, v in enumerate(unique_postals)}

X_train_postal_codes = np.array([postal_to_index[x] for x in X_train_postal])
# Test set: unknown postal_codes mapped to -1
X_test_postal_codes = np.array([postal_to_index.get(x, -1) for x in X_test_postal])

n_categories = len(unique_postals)

### Convert other features to numpy ###
X_train_other = X_train_other.to_numpy()
X_test_other = X_test_other.to_numpy()

X_train_other = X_train_other.astype("float32")
X_test_other = X_test_other.astype("float32")

#############################################

### There is no R2 metric in TensorFlow! ###

import tensorflow.keras.backend as K

def r2_metric(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


#############################################
#############################################
#############################################

#############################################
#                                           #
#        Build Keras model                  #
#                                           #
#############################################

emb_dim = 16  # size of embedding

postal_input = Input(shape=(1, ), name="postal_input")
# Add +1 to input_dim for unknown category (-1 mapped to 0)
postal_emb = Embedding(input_dim=n_categories + 1,
                       output_dim=emb_dim)(postal_input)
postal_emb = Flatten()(postal_emb)

num_input = Input(shape=(X_train_num.shape[1], ), name="num_input")
other_input = Input(shape=(X_train_other.shape[1], ), name="other_input")

#############################################

LEARNING_RATE = 0.001

BATCH_SIZE = 64

EPOCHS_NUM = 250

input_layer = Concatenate()([postal_emb, num_input, other_input])

x = Dense(128, activation="relu")(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.15)(x)

x = Dense(64, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.10)(x)

x = Dense(32, activation="relu")(x)

output = Dense(1)(x)


model = Model(inputs=[postal_input, num_input, other_input], outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=[r2_metric, "mae"]
)

#############################################
#                                           #
#        Train the model                    #
#                                           #
#############################################

callbacks = [
    EarlyStopping(
        patience=25,
        restore_best_weights=True,
        monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )
]

history = model.fit(
    [X_train_postal_codes, X_train_num, X_train_other],
    y_train,
    epochs=EPOCHS_NUM,
    batch_size=BATCH_SIZE,
    validation_split=0.1,   # <---- added validation inside keras
    callbacks=callbacks,
    verbose=1
)

#############################################
#                                           #
#        Evaluate the model                 #
#                                           #
#############################################

y_pred_scaled = model.predict([X_test_postal_codes, X_test_num, X_test_other])

y_pred = y_scaler.inverse_transform(y_pred_scaled)

y_test = y_scaler.inverse_transform(y_test)

r2 = r2_score(y_test, y_pred)

print('\n ************************** \n\n')
print(f"R2 on test set:{r2:.3f}") 
print('\n ************************** \n\n')

#############################################
#        Best epoch info and loss gap
#############################################

# best epoch (0-based index > +1 for human-readable)
best_epoch = np.argmin(history.history["val_loss"]) + 1

train_loss_best = history.history["loss"][best_epoch - 1]
val_loss_best = history.history["val_loss"][best_epoch - 1]

gap = val_loss_best - train_loss_best

print("\n===== Best Epoch Summary =====")
print(f"Best epoch: {best_epoch}")
print(f"Train loss at best epoch: {train_loss_best:.4f}")
print(f"Val loss at best epoch:   {val_loss_best:.4f}")
print(f"Gap (val - train):        {gap:.64f}")
print("================================\n")


#############################################
#############################################
#############################################

print('\nTask completed!')

#############################################
