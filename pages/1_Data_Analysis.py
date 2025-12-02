import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Note: Removed 'import seaborn as sns' to avoid ModuleNotFoundError

st.set_page_config(page_title="Data Analysis", page_icon="📊")
st.title("📊 Exploratory Data Analysis (EDA)")

# Load Raw Data
try:
    df = pd.read_csv("data/ckd_data.csv")
    
    # Rename columns to match model training
    df = df.rename(columns={'wc': 'wbcc', 'rc': 'rbcc'})
    
    # Clean Numerical Columns
    num_cols = ["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wbcc","rbcc"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Clean Categorical Columns (just for display purposes)
    cat_cols = ["sg","al","su","rbc","pc","pcc","ba","htn","dm","cad","appet","pe","ane"]
    for col in cat_cols:
        df[col] = df[col].astype(str)

except FileNotFoundError:
    st.error("❌ Data file not found. Please ensure 'data/ckd_data.csv' exists.")
    st.stop()

# --- PREVIEW ---
st.subheader("Dataset Preview")
st.write(f"**Dimensions:** {df.shape[0]} rows, {df.shape[1]} columns")
st.dataframe(df.head())

# --- STATISTICS ---
st.subheader("Statistical Summary")
st.write(df[num_cols].describe())

# --- MISSING VALUES ---
st.subheader("Missing Value Analysis")
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    st.bar_chart(missing)
else:
    st.success("No missing values found (after cleaning).")

# --- CORRELATION ---
st.subheader("Feature Correlation Heatmap")
# Using Streamlit's native dataframe styling instead of Seaborn
numeric_df = df[num_cols]
corr = numeric_df.corr()
st.dataframe(corr.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

# --- DISTRIBUTIONS ---
st.subheader("Target Distribution (CKD vs Not CKD)")
class_counts = df['class'].value_counts()
st.bar_chart(class_counts)

st.subheader("Distribution of Key Features")
tab1, tab2, tab3 = st.tabs(["Hemoglobin", "Serum Creatinine", "Blood Pressure"])

with tab1:
    fig, ax = plt.subplots()
    # Replaced sns.histplot with standard plt.hist
    ax.hist(df["hemo"].dropna(), bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Hemoglobin Distribution")
    ax.set_xlabel("Hemoglobin")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    ax.hist(df["sc"].dropna(), bins=20, color="salmon", edgecolor="black")
    ax.set_title("Serum Creatinine Distribution")
    ax.set_xlabel("Serum Creatinine")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots()
    ax.hist(df["bp"].dropna(), bins=20, color="lightgreen", edgecolor="black")
    ax.set_title("Blood Pressure Distribution")
    ax.set_xlabel("Blood Pressure")
    ax.set_ylabel("Count")
    st.pyplot(fig)