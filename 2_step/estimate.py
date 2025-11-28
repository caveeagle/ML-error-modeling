"""
Dataset estimation
""" 

import pandas as pd

#############################################

dataset_filename = '../data/loss_dataset.csv'

df = pd.read_csv(dataset_filename, delimiter=',')

#############################################

###        Import global variables        ###

from pathlib import Path
import sys

config_dir = Path(__file__).resolve().parents[1] / 'config'
sys.path.append(str(config_dir))

from settings import RANDOM_STATE

#############################################

print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")

col = 'loss'

mean_val = df[col].mean()
median_val = df[col].median()
max_val = df[col].max()
min_val = df[col].min()

print(f"{col}: mean = {mean_val:.2f}, median = {median_val:.2f}, max = {max_val}, min = {min_val}")

'''
rows: 16207, columns: 51

loss: 
mean = 56288, 
median = 32032, 
max = 1332406, 
min = 0.32
''' 
#############################################

if(1): 
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col]) 
    
    plt.title(f"Distribution of column {col}")
    plt.xlim(0, 400_000)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    if(1):
        plt.savefig("loss_distribution.png", dpi=150)
    plt.show() 

#############################################

print('\nTask completed!')

#############################################
