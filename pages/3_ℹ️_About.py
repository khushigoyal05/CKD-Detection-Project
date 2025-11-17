import streamlit as st

st.title("ℹ️ About This Project")

st.write("""
### Early Detection of Chronic Kidney Disease (CKD)

This machine learning project predicts whether a patient is at risk of CKD 
based on 25 clinical features such as:

- Age  
- Blood Pressure  
- Specific Gravity  
- Albumin  
- Hemoglobin  
- Serum Creatinine  
- Diabetes, Hypertension  
- RBC Count, WBC Count  
- Appetite, Edema  
- and more…

### Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-Learn  
- Streamlit  
- Logistic Regression (Best Model)

### Outcome
The model achieves:

- **Accuracy:** 98.75%  
- **AUC:** 1.00  
- **Sensitivity:** 1.00  

This means it identifies CKD patients with extremely high reliability.

### Developed By
**Khushi Goyal**  
Computer Science Engineering  
Thapar Institute of Engineering and Technology  
""")
