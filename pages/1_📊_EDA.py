import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis (EDA)")

df = pd.read_csv("data/ckd_cleaned.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Shape of Dataset")
st.write(df.shape)

st.subheader("Missing Value Count")
missing = df.isnull().sum()
st.write(missing)

st.subheader("Correlation Heatmap")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
st.pyplot(plt)

st.subheader("Distribution of Hemoglobin")
plt.hist(df["hemo"], bins=20)
plt.xlabel("Hemoglobin")
plt.ylabel("Count")
st.pyplot(plt)
