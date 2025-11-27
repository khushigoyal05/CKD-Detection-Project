import pandas as pd

df = pd.read_csv("data/ckd_data.csv")   # not scaled one
print(df.describe())
print(df.isna().sum())
