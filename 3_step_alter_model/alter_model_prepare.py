"""
Prepare datase
"""
import pandas as pd
import numpy as np
import random

import os

#############################################

dataset_filename = '../data/loss_dataset.csv'

df = pd.read_csv(dataset_filename, delimiter=',')

##############################################

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

##############################################

threshold = np.percentile(df["loss"], 70)   # top 30%

difficult_mask = df["loss"] >= threshold

df_difficult = df.loc[difficult_mask]

print(f"Total rows: {len(df)}")
print(f"Difficult rows selected: {len(df_difficult)}")

##################################################

if(1):
    filename_loss = '../data/difficult_dataset.csv'
    df_difficult.to_csv(filename_loss, sep=',', index=False)

#############################################

print('\nTask completed!')

#############################################
