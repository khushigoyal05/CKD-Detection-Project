import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= Load Model + Preprocessing ================= #
try:
    model = joblib.load("models/best_model.pkl")
    pre = joblib.load("models/preprocessing.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Columns must match training exactly (including new bp_diff)
num_cols = ["age","bp","bp_diff","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
cat_cols = ["sg","al","su","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]

num_medians = pre["num_medians"]
cat_modes = pre["cat_modes"]
encoder = pre["onehot"]
scaler = pre["scaler"]

# ================= HEALTHY DEFAULTS (For Empty Optional Fields) ================= #
HEALTHY_DEFAULTS = {
    "sod": 140.0,   "pot": 4.5,     "pcv": 45.0,
    "wbcc": 8000.0, "rbcc": 5.0,
    "rbc": "normal", "pc": "normal", "pcc": "notpresent", "ba": "notpresent",
    "cad": "no", "appet": "good", "pe": "no", "ane": "no"
}

# ================ UI ================ #
st.set_page_config(page_title="CKD Prediction", page_icon="🩺")
st.title("🩺 CKD Risk Predictor")

st.markdown("Enter the clinical values below. **Required fields are mandatory.**")
st.write("---")

# ================= REQUIRED INPUT ================= #
st.subheader("🔥 Required Clinical Indicators")

c1,c2,c3 = st.columns(3)
with c1:
    sc = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, step=0.1, value=1.0)
    bu = st.number_input("Blood Urea (mg/dL)", 5.0, 300.0, step=1.0, value=30.0)
with c2:
    al_val = st.selectbox("Urine Albumin (0-5)", [0,1,2,3,4,5], index=0)
    su_val = st.selectbox("Urine Sugar (0-5)", [0,1,2,3,4,5], index=0)
with c3:
    sg_val = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025], index=2)
    bgr = st.number_input("Random Glucose (mg/dL)", 50, 500, step=1, value=90)

b1,b2,b3 = st.columns(3)
with b1: bp = st.number_input("Blood Pressure (mmHg)", 50, 200, value=80)
with b2: hemo = st.number_input("Hemoglobin (g/dL)", 3.0, 20.0, step=0.1, value=14.0)
with b3: age = st.number_input("Age (Years)", 1, 120, value=35)

hy1,hy2 = st.columns(2)
with hy1: htn = st.selectbox("Hypertension", ["yes","no"], index=1)
with hy2: dm = st.selectbox("Diabetes Mellitus", ["yes","no"], index=1)

# Convert strings immediately
sg = str(sg_val)
al = str(al_val)
su = str(su_val)

st.write("---")

# ================= OPTIONAL INPUT ================= #
st.subheader("🔹 Optional Fields")
st.info("If left unchecked, healthy default values will be assumed.")

use_optional = st.checkbox("Enter Optional Values manually?")

sod=pot=pcv=wbcc=rbcc= None
rbc=pc=pcc=ba=cad=appet=pe=ane= None

if use_optional:
    o1,o2,o3 = st.columns(3)
    with o1:
        sod = st.number_input("Sodium (mEq/L)", 0.0, 200.0, step=0.1)
        pot = st.number_input("Potassium (mEq/L)", 0.0, 20.0, step=0.1)
        rbc = st.selectbox("Red Blood Cells", ["normal","abnormal"])
    with o2:
        pc = st.selectbox("Pus Cells", ["normal","abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["present","notpresent"])
        ba = st.selectbox("Bacteria", ["present","notpresent"])
    with o3:
        pcv = st.number_input("Packed Cell Volume", 0.0, 60.0, step=0.1)
        wbcc = st.number_input("WBC Count", 0.0, 30000.0, step=100.0)
        rbcc = st.number_input("RBC Count", 0.0, 10.0, step=0.1)

    cad = st.selectbox("Coronary Artery Disease", ["yes","no"])
    appet = st.selectbox("Appetite", ["good","poor"])
    pe = st.selectbox("Pedal Edema", ["yes","no"])
    ane = st.selectbox("Anemia", ["yes","no"])

# ================= PREDICTION LOGIC ================= #
if st.button("Predict Risk"):
    
    # CALCULATE FEATURE ENGINEERING (BP Deviation)
    bp_diff = abs(bp - 80)

    # Build Row Data
    row = {
        "age":age, "bp":bp, "bp_diff":bp_diff, "bgr":bgr, "bu":bu, "sc":sc, "hemo":hemo,
        "sg":sg, "al":al, "su":su, "htn":htn, "dm":dm,
        "sod":sod, "pot":pot, "pcv":pcv, "wbcc":wbcc, "rbcc":rbcc,
        "rbc":rbc, "pc":pc, "pcc":pcc, "ba":ba, 
        "cad":cad, "appet":appet, "pe":pe, "ane":ane
    }
    
    df = pd.DataFrame([row])
    df = df[num_cols + cat_cols] # Enforce order

    # 1. IMPUTE NUMERICAL COLUMNS
    # bp_diff is calculated from input, so it won't be NaN
    for col in num_cols:
        if df[col].isna().any():
            if col in HEALTHY_DEFAULTS:
                df[col] = df[col].fillna(HEALTHY_DEFAULTS[col])
            else:
                df[col] = df[col].fillna(num_medians.get(col, 0))

    # 2. IMPUTE CATEGORICAL COLUMNS
    for col in cat_cols:
        df[col] = df[col].astype(object).astype(str).str.strip()
        is_missing = df[col].isin(['None', 'nan', 'unknown', 'NaN'])
        if is_missing.any():
            if col in HEALTHY_DEFAULTS:
                df.loc[is_missing, col] = HEALTHY_DEFAULTS[col]
            else:
                df.loc[is_missing, col] = str(cat_modes[col])

    # 3. SCALE & ENCODE
    try:
        X_scaled_num = scaler.transform(df[num_cols])
        X_encoded_cat = encoder.transform(df[cat_cols]).toarray()
    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.stop()

    # 4. PREDICT
    X_final = np.concatenate([X_scaled_num, X_encoded_cat], axis=1)
    prediction_prob = model.predict_proba(X_final)[0][1] * 100
    
    # ================= DISPLAY RESULTS ================= #
    st.markdown(f"### 🔍 Prediction Result")
    
    if prediction_prob < 40:
        st.success(f"🟢 **Low Risk ({prediction_prob:.2f}%)**\n\nPatient appears healthy based on inputs.")
    elif prediction_prob < 70:
        st.warning(f"🟡 **Moderate Risk ({prediction_prob:.2f}%)**\n\nConsider further clinical evaluation.")
    else:
        st.error(f"🔴 **High Risk ({prediction_prob:.2f}%)**\n\nMedical attention recommended.")

    st.caption("Note: 'BP Deviation' logic ensures extremely low BP is treated similarly to high BP.")