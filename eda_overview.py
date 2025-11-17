import pandas as pd
import numpy as np
import os

# Load dataset
df = pd.read_csv("data/ckd_data.csv")

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(list(df.columns))

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 10 ROWS =====")
print(df.head(10))

print("\n===== MISSING VALUE COUNTS =====")
print(df.isnull().sum())

print("\n===== UNIQUE VALUE COUNTS (For each column) =====")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique → {df[col].unique()[:10]}")

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe().T)
