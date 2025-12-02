import streamlit as st

st.set_page_config(page_title="Project Info", page_icon="ℹ️")
st.title("ℹ️ About This Project")

st.write("""
### 🏥 Early Detection of Chronic Kidney Disease (CKD)

This Machine Learning project is designed to assist in the early identification of patients at risk for Chronic Kidney Disease. 
It analyzes **24+ clinical features** to provide a probabilistic risk assessment.

### 🧠 How It Works
The model uses patient data including:
- **Blood Tests:** Serum Creatinine, Blood Urea, Hemoglobin, Random Glucose, Sodium, Potassium.
- **Urine Tests:** Albumin, Sugar, Specific Gravity.
- **Physical Indicators:** Blood Pressure, Age, Edema.
- **Derived Features:** **Blood Pressure Deviation** (treating extremely low BP as risky, similar to high BP).

### 🛠 Tech Stack
- **Algorithm:** Random Forest Classifier (Calibrated for probability accuracy).
- **Training Strategy:** No synthetic data (SMOTE removed), utilizing **Class Weights** to handle data imbalance naturally.
- **Frameworks:** Python, Scikit-Learn, Pandas, Streamlit.

### 📊 Performance
The current production model achieves:
- **Accuracy:** ~98.8%
- **AUC (Area Under Curve):** ~0.9995
- **False Positive Rate:** Extremely low (< 2%), minimizing false alarms for healthy patients.

### 👥 Developed By
**Khushi Goyal** 
**,** 
**Shambhavi**   
(Computer Science Department)  
Thapar Institute of Engineering and Technology
""")