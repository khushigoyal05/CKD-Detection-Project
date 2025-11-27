import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================= Load Model + Preprocessing ================= #
model = joblib.load("models/best_model.pkl")
pre = joblib.load("models/preprocessing.pkl")

num_cols = pre["num_cols"]
cat_cols = pre["cat_cols"]
num_medians = pre["num_medians"]
cat_modes = pre["cat_modes"]
encoder = pre["onehot"]

# ================ UI ================ #
st.set_page_config(page_title="CKD Prediction", page_icon="🩺")
st.title("🩺 CKD Risk Predictor — Clinically Tuned Model")

st.markdown("""
Enter the required values below.  
Optional inputs improve accuracy — leave blank if unknown.
""")
st.write("---")

# ================= REQUIRED INPUT (High impact) ================= #
st.subheader("🔥 Required Clinical Indicators (Strong CKD Features)")

c1,c2,c3 = st.columns(3)
with c1:
    sc = st.number_input("Serum Creatinine (mg/dL)",0.1,15.0,step=0.1)
    bu = st.number_input("Blood Urea (mg/dL)",5.0,300.0,step=1.0)
with c2:
    al = st.selectbox("Urine Albumin (0-5)",[0,1,2,3,4,5])
    su = st.selectbox("Urine Sugar (0-5)",[0,1,2,3,4,5])
with c3:
    sg = st.selectbox("Urine Specific Gravity",[1.005,1.010,1.015,1.020,1.025])

b1,b2,b3 = st.columns(3)
with b1: bp = st.number_input("Blood Pressure (mmHg)",60,220)
with b2: hemo = st.number_input("Hemoglobin g/dL",3.0,20.0,step=0.1)
with b3: age = st.number_input("Age",1,120)

hy1,hy2 = st.columns(2)
with hy1: htn = st.selectbox("Hypertension",["yes","no"])
with hy2: dm = st.selectbox("Diabetes",["yes","no"])

st.write("---")

# ================= OPTIONAL INPUT ================= #
st.subheader("🔹 Optional — enhances accuracy")

if st.checkbox("Show optional fields"):

    o1,o2,o3 = st.columns(3)

    with o1:
        sod = st.number_input("Sodium (mEq/L)",min_value=0.0,step=0.1)
        pot = st.number_input("Potassium",min_value=0.0,step=0.1)
        rbc = st.selectbox("RBC",["normal","abnormal","unknown"])

    with o2:
        pc = st.selectbox("Pus Cells",["normal","abnormal","unknown"])
        pcc = st.selectbox("Pus Clumps",["present","notpresent","unknown"])
        ba = st.selectbox("Bacteria",["present","notpresent","unknown"])

    with o3:
        pcv = st.number_input("PCV",min_value=0.0,step=0.5)
        wbcc = st.number_input("WBC Count",min_value=0.0,step=10.0)
        rbcc = st.number_input("RBC Count",min_value=0.0,step=0.1)

    cad = st.selectbox("Coronary Artery Disease",["yes","no","unknown"])
    appet = st.selectbox("Appetite",["good","poor","unknown"])
    pe = st.selectbox("Pedal Edema",["yes","no","unknown"])
    ane = st.selectbox("Anemia",["yes","no","unknown"])

else:
    # missing = treat as unknown automatically
    sod=pot=pcv=wbcc=rbcc=np.nan
    rbc=pc=pcc=ba=cad=appet=pe=ane="unknown"

# ================= BUILD INPUT DATAFRAME ================= #
row = {
"age":age,"bp":bp,"sg":sg,"al":al,"su":su,"rbc":rbc,"pc":pc,"pcc":pcc,"ba":ba,
"bgr":np.nan,"bu":bu,"sc":sc,"sod":sod,"pot":pot,"hemo":hemo,"pcv":pcv,"wbcc":wbcc,
"rbcc":rbcc,"htn":htn,"dm":dm,"cad":cad,"appet":appet,"pe":pe,"ane":ane
}
df = pd.DataFrame([row])

# ================= HANDLE MISSING VALUES (FIX) ================= #
for col in num_cols:
    df[col] = df[col].fillna(num_medians[col])

for col in cat_cols:
    df[col] = df[col].replace("unknown",cat_modes[col])

encoded = encoder.transform(df[cat_cols])
X = np.concatenate([df[num_cols],encoded.toarray()],axis=1)

# ================= PREDICT ================= #
if st.button("Predict CKD Probability"):
    prob = model.predict_proba(X)[0][1] * 100
    prob = round(prob, 2)

    if prob < 30:
        st.success(f"🟢 CKD Risk: {prob}% — Low")
    elif prob < 60:
        st.warning(f"🟡 CKD Risk: {prob}% — Moderate")
    else:
        st.error(f"🔴 CKD Risk: {prob}% — HIGH (medical review suggested)")

    st.caption("⚠ AI estimation — not a clinical diagnosis.")
