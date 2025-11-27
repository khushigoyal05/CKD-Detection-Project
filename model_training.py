import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

print("\n📂 Loading dataset...")
df = pd.read_csv("data/ckd_data.csv")

# ------------------------- CLEANING -------------------------

df["class"] = df["class"].replace({"no": "notckd"})
df = df[df["class"].isin(["ckd", "notckd"])]   # remove 1 outlier sample

num_cols = ["age","bp","sg","al","su","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
cat_cols = ["rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

# Convert numbers safely
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna("unknown")

df["class"] = df["class"].map({"ckd":1,"notckd":0})

# ------------------------- ENCODE NOW (before SMOTE) -------------------------

encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(df[cat_cols]).toarray()

X = np.concatenate([df[num_cols], X_encoded], axis=1)  # numerical + encoded categorical
y = df["class"]

# ------------------------- APPLY SMOTE -------------------------

print("\n⚖ Balancing dataset with SMOTE...")
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# ------------------------- MODEL -------------------------

base_model = LogisticRegression(max_iter=300, C=0.6)  # reduce overconfident fit
model = CalibratedClassifierCV(base_model, cv=5, method="isotonic")

print("\n🚀 Training CKD Model using 5-fold CV...")
model.fit(X_resampled, y_resampled)

# CV Probabilities for realistic evaluation
cv_pred = cross_val_predict(model, X_resampled, y_resampled, cv=5, method="predict_proba")[:,1]
cv_class = (cv_pred > 0.5).astype(int)

print("\n===== PERFORMANCE =====")
print("Accuracy:", round(accuracy_score(y_resampled, cv_class),4))
print("AUC:", round(roc_auc_score(y_resampled, cv_pred),4))
print("\nConfusion Matrix:\n", confusion_matrix(y_resampled, cv_class))
print("\nClassification Report:\n", classification_report(y_resampled, cv_class))

# ------------------------- SAVE MODEL + ENCODER -------------------------

joblib.dump(model, "models/best_model.pkl")
joblib.dump({"onehot": encoder, "num_cols":num_cols, "cat_cols":cat_cols},
            "models/preprocessing.pkl")

print("\n🎉 Model + preprocessors saved successfully!\n")
print("➡ Now run:  streamlit run app.py\n")
