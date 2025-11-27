import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

print("🔄 Loading raw dataset...")
df = pd.read_csv("data/ckd_data.csv")

# ----------------------------------------------------
# FIX WRONG CATEGORICAL VALUES
# ----------------------------------------------------
df["class"] = df["class"].replace("no", "notckd")
df["pe"] = df["pe"].replace("good", np.nan)
df["appet"] = df["appet"].replace("no", np.nan)

# ----------------------------------------------------
# FORCE PROPER NUMERIC CONVERSION
# ----------------------------------------------------
numeric_cols = [
    "age","bp","sg","al","su","bgr","bu","sc","sod",
    "pot","hemo","pcv","wbcc","rbcc"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------------------------------
# IDENTIFY REMAINING CATEGORICAL COLS
# ----------------------------------------------------
cat_cols = [c for c in df.columns if c not in numeric_cols and c != "class"]

# ----------------------------------------------------
# IMPUTE NUMERIC (median)
# ----------------------------------------------------
num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# ----------------------------------------------------
# IMPUTE CATEGORICAL (mode)
# ----------------------------------------------------
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# ----------------------------------------------------
# LABEL ENCODE CATEGORICAL COLUMNS
# ----------------------------------------------------
label_encoders = {}

for col in cat_cols + ["class"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ----------------------------------------------------
# SCALE NUMERIC FEATURES
# ----------------------------------------------------
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ----------------------------------------------------
# SAVE CLEAN DATASET
# ----------------------------------------------------
df.to_csv("data/ckd_cleaned.csv", index=False)

joblib.dump({
    "label_encoders": label_encoders,
    "scaler": scaler,
    "numeric_cols": numeric_cols,
    "categorical_cols": cat_cols
}, "models/preprocessing_objects.pkl")

print("✨ Preprocessing complete!")
print("✔ Cleaned dataset saved to data/ckd_cleaned.csv")
print("✔ Preprocessing objects saved to models/preprocessing_objects.pkl")
print("✔ Final shape:", df.shape)
print("✔ Missing values (should be all 0):")
print(df.isna().sum())
