import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("models/best_model.pkl")

# Load preprocessing objects
preprocess = joblib.load("models/preprocessing_objects.pkl")
label_encoders = preprocess["label_encoders"]
scaler = preprocess["scaler"]
num_cols = preprocess["numeric_cols"]
cat_cols = preprocess["categorical_cols"]

st.set_page_config(page_title="CKD Prediction App", layout="wide")

st.title("🧪 Early Detection of Chronic Kidney Disease (CKD)")
st.write("""
This tool predicts the risk of **Chronic Kidney Disease** using a Machine Learning model.
Please enter the patient’s medical parameters below.
""")

# ----------------------
# INPUT UI
# ----------------------
st.header("Enter Patient Medical Parameters")

# Categorical dropdown options (original string values)
category_options = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"]
}

user_data = {}

# Create two sections: numeric inputs & categorical inputs
col_num, col_cat = st.columns(2)

# --------- NUMERIC INPUTS -----------
with col_num:
    st.subheader("🔢 Numeric Inputs")
    for col in num_cols:
        user_data[col] = st.number_input(f"{col}", value=0.0)

# --------- CATEGORICAL INPUTS -------
with col_cat:
    st.subheader("🔤 Categorical Inputs")
    for col in cat_cols:
        if col != "class":   # class is target
            options = category_options.get(col, [])
            if options:
                user_data[col] = st.selectbox(f"{col}", options)
            else:
                # Other unknown categorical columns
                user_data[col] = st.text_input(f"{col}")

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# ----------------------
# PREPROCESSING
# ----------------------
def preprocess_input(df):
    df_processed = df.copy()

    # Encode categorical
    for col in cat_cols:
        if col in df_processed.columns:
            le = label_encoders[col]
            df_processed[col] = le.transform(df_processed[col])

    # Scale numeric
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])

    return df_processed

# ----------------------
# PREDICT BUTTON
# ----------------------
st.write("---")
if st.button("🔍 Predict CKD"):

    processed_input = preprocess_input(input_df)

    pred = model.predict(processed_input)[0]
    prob = model.predict_proba(processed_input)[0][1]

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠ High Risk of CKD\n**Probability: {prob:.2f}**")
    else:
        st.success(f"✔ Low Risk / Normal\n**Probability: {prob:.2f}**")

st.write("---")
st.caption("Developed by Khushi Goyal · Machine Learning Project · CKD Early Detection")
