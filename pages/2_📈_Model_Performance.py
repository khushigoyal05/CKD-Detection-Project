import streamlit as st
import pandas as pd

st.title("📈 Model Performance Comparison")

st.write("Performance of various ML models trained on CKD dataset.")

# Import results saved earlier
results = pd.DataFrame([
    ["Logistic Regression", 0.9875, 0.9677, 1.0000, 0.9836, 1.0000],
    ["KNN", 0.9625, 0.9091, 1.0000, 0.9524, 0.9980],
    ["Decision Tree", 0.9750, 1.0000, 0.9333, 0.9655, 0.9667],
    ["Random Forest", 0.9750, 1.0000, 0.9333, 0.9655, 0.9980],
    ["SVM", 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Gradient Boosting", 0.9750, 1.0000, 0.9333, 0.9655, 1.0000],
], columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"])

st.dataframe(results)

best_model = results.loc[results['AUC'].idxmax()]["Model"]
st.success(f"🏆 **Best Model: {best_model}**")
