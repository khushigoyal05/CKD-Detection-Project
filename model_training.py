import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

print("\n📂 Loading dataset...")
df = pd.read_csv("data/ckd_data.csv")

# Rename columns to match the names used throughout the code
df = df.rename(columns={'wc': 'wbcc', 'rc': 'rbcc'})

# ------------------------- CLEANING -------------------------

df["class"] = df["class"].replace({"no": "notckd"})
df = df[df["class"].isin(["ckd", "notckd"])]   # remove 1 outlier sample

# DEFINE COLUMNS
# Note: 'bp_diff' is a new feature we will engineer below, so we add it to num_cols
num_cols = ["age","bp","bp_diff","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
cat_cols = ["sg","al","su","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

# Convert numerical columns safely
# We exclude 'bp_diff' from this loop because it doesn't exist in the CSV yet
raw_num_cols = [c for c in num_cols if c != "bp_diff"]
for col in raw_num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Standardize categorical value strings
for col in ["al", "su"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int).astype(str)
    df[col] = df[col].replace('-1', np.nan)

# For sg, convert to string cleaning trailing zeros
df['sg'] = pd.to_numeric(df['sg'], errors='coerce').astype(str).str.replace(r'\.0$', '', regex=True)
df['sg'] = df['sg'].replace('nan', np.nan)


# STORE IMPUTATION VALUES
num_medians = df[raw_num_cols].median().to_dict()

cat_modes = {}
for col in cat_cols:
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.replace('\t', '', regex=False)
        cat_modes[col] = df[col].mode()[0]
    else:
        cat_modes[col] = df[col].mode()[0]

# IMPUTE IN TRAINING DATA
df[raw_num_cols] = df[raw_num_cols].fillna(num_medians)

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.replace('\t', '', regex=False)
    df[col] = df[col].replace(['nan', 'unknown', 'None'], cat_modes.get(col, 'Mode_Error'))

# ================= FEATURE ENGINEERING ================= #
# Create bp_diff: Distance from ideal diastolic BP (80)
# This forces the model to treat Low BP (e.g. 50) as a deviation, similar to High BP
df['bp_diff'] = np.abs(df['bp'] - 80)
# =======================================================

df["class"] = df["class"].map({"ckd":1,"notckd":0})

# ------------------------- PREPROCESSING -------------------------

# SCALING NUMERICAL FEATURES
print("\n🔍 Fitting Scaler...")
scaler = StandardScaler()
# Now we scale all numerical columns INCLUDING the new bp_diff
df[num_cols] = scaler.fit_transform(df[num_cols])

# ENCODING CATEGORICAL FEATURES 
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[cat_cols]).toarray()

X = np.concatenate([df[num_cols], X_encoded], axis=1)
y = df["class"]

# ------------------------- MODEL: RANDOM FOREST + CLASS WEIGHTS -------------------------

base_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight='balanced', # Handle imbalance
    random_state=42,
    n_jobs=-1
)

# Calibrate probabilities using Sigmoid
model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")

print("\n🚀 Training Calibrated Random Forest (Balanced + Feature Eng) using 5-fold CV...")
model.fit(X, y)

# CV Probabilities
cv_pred = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:,1]
cv_class = (cv_pred > 0.5).astype(int)

print("\n===== PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y, cv_class),4))
print("AUC:", round(roc_auc_score(y, cv_pred),4))
print("\nConfusion Matrix:\n", confusion_matrix(y, cv_class))
print("\nClassification Report:\n", classification_report(y, cv_class))

# ------------------------- SAVE MODEL + ENCODER -------------------------

joblib.dump(model, "models/best_model.pkl")
joblib.dump({
    "onehot": encoder,
    "scaler": scaler,
    "num_cols":num_cols,
    "cat_cols":cat_cols,
    "num_medians": num_medians,
    "cat_modes": cat_modes
}, "models/preprocessing.pkl")

print("\n🎉 Model saved! (Added 'BP Deviation' Feature)\n")
print("➡ Now run:  streamlit run app.py\n")