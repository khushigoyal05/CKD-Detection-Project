import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib


# 1. Load dataset
df = pd.read_csv("data/ckd_data.csv")

# ---- STEP 3.1: CLEAN WRONG CATEGORICAL VALUES ----

# Fix 'class' column (replace 'no' → 'notckd')
df["class"] = df["class"].replace("no", "notckd")

# Fix incorrect entries in 'pe', 'appet'
df["pe"] = df["pe"].replace("good", np.nan)
df["appet"] = df["appet"].replace("no", np.nan)

# ---- STEP 3.2: CONVERT NUMERIC COLUMNS FROM STRING TO FLOAT ----

numeric_cols = ["pcv", "rbcc", "wbcc"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---- STEP 3.3: HANDLE MISSING VALUES ----

# Separate numeric & categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Numeric imputation → median (robust against outliers)
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical imputation → mode
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# ---- STEP 3.4: ENCODE CATEGORICAL VARIABLES ----

label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ---- STEP 3.5: SCALE NUMERICAL FEATURES ----

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ---- STEP 3.6: SAVE CLEAN DATASET ----

# ---- SAVE CLEAN DATASET ----
# ---- STEP 3.6: SAVE CLEAN DATASET ----
df.to_csv("data/ckd_cleaned.csv", index=False)

# ---- SAVE ENCODERS + SCALER FOR STREAMLIT ----
import joblib

joblib.dump({
    "label_encoders": label_encoders,
    "scaler": scaler,
    "numeric_cols": num_cols,
    "categorical_cols": cat_cols
}, "models/preprocessing_objects.pkl")

print("✔ Preprocessing complete!")
print("✔ Cleaned dataset saved as data/ckd_cleaned.csv")
print("✔ Preprocessing objects saved to models/preprocessing_objects.pkl")
print("✔ Final shape:", df.shape)
