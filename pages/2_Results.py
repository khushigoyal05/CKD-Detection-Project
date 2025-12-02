import streamlit as st
import pandas as pd

st.set_page_config(page_title="Model Results", page_icon="📈")
st.title("📈 Model Performance")

# st.write("""
# We trained multiple models to find the best predictor for Chronic Kidney Disease.
# The **Random Forest Classifier** was selected for the final application due to its stability, high accuracy, and robustness against overfitting.
# """)

# # Latest Results from your recent training run
# # Accuracy: 0.9875, AUC: 0.9995
# results = pd.DataFrame([
#     ["Random Forest (Calibrated)", 0.9875, 0.9900, 0.9900, 0.9900, 0.9995],
#     ["Logistic Regression", 0.9650, 0.9600, 0.9700, 0.9650, 0.9800],
#     ["Decision Tree", 0.9750, 0.9500, 0.9600, 0.9550, 0.9750],
#     ["SVM", 0.9600, 0.9400, 0.9500, 0.9450, 0.9700],
# ], columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"])

# # Highlight the best model
# st.dataframe(results.style.highlight_max(axis=0, color='lightgreen'))

st.subheader("🏆 Selected Model: Random Forest")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Accuracy", value="98.75%")
with col2:
    st.metric(label="AUC Score", value="0.9995")
with col3:
    st.metric(label="F1 Score", value="0.99")

st.markdown("""
### Why Random Forest?
1. **Handling Non-Linearity:** Medical data often has complex, non-linear relationships (e.g., risk doesn't increase in a straight line with age).
2. **Robustness:** It is less likely to overfit than a single Decision Tree.
3. **Imbalance Handling:** We used **Class Weights** to ensure the model learns to identify Healthy patients just as well as CKD patients.
""")