## 🚑 Early Detection of Chronic Kidney Disease (CKD)

A **Machine Learning–powered web application** that predicts the likelihood of **Chronic Kidney Disease (CKD)** based on patient medical parameters.

The project focuses on clinical accuracy and includes:
* ✔ Robust Data Preprocessing & Feature Engineering
* ✔ Exploratory Data Analysis (EDA)
* ✔ **Calibrated Random Forest Model** for realistic risk probability
* ✔ Smart Imputation for missing values
* ✔ An interactive **Streamlit UI** with detailed risk categorization

---

## 🌐 Live Demo (Optional)

Add link here after deployment:
**👉 *Coming Soon***

---

## 📊 Features

### 🔹 1. **Clinically Tuned Web Interface**
* **Smart Inputs:** Distinguishes between "Required" (high impact) and "Optional" clinical features.
* **Healthy Defaults:** Automatically handles missing optional data by imputing "healthy" values (instead of dataset averages) to prevent false alarms for healthy users.
* **Real-time Prediction:** Instant probabilistic risk assessment (Low/Moderate/High).

### 🔹 2. **Data Analysis Dashboard**
* Dataset overview and statistical summaries.
* Distribution plots for key biomarkers (Hemoglobin, Creatinine, BP).
* Correlation heatmaps to understand feature relationships.

### 🔹 3. **High-Performance Model**
* **Selected Model:** **Random Forest Classifier** (Calibrated).
* **Why Random Forest?** Selected over linear models (like Logistic Regression) for its ability to capture complex, non-linear relationships in medical data and its stability against overfitting.
* **Optimization:** Uses **Class Weights** to handle dataset imbalance and **Sigmoid Calibration** to provide accurate percentage probabilities.

**Performance Metrics:**
* **Accuracy:** ~98.75%
* **AUC Score:** ~0.9995
* **False Positive Rate:** Extremely low (< 2%)

---

## 📁 Project Structure

```

CKD\_Detection\_Project/
│── app.py                 \# Main Streamlit Application
│── model\_training.py      \# Training script (RF + Calibration)
│── preprocess\_data.py     \# Initial cleaning scripts
│── requirements.txt       \# Python Dependencies
│── README.md              \# Project Documentation
│── data/
│     └── ckd\_data.csv     \# Raw dataset
│── models/
│     ├── best\_model.pkl   \# Trained Model
│     ├── preprocessing.pkl\# Scalers, Encoders, & Imputation logic
│     └── metrics.json     \# Saved metrics from training
│── pages/
│     ├── 1\_Data\_Analysis.py  \# EDA Dashboard
│     ├── 2\_Results.py        \# Model Performance Metrics
│     └── 3\_Project\_Info.py   \# About Page

````

---

## 🚀 How to Run Locally

### 1️⃣ Create virtual environment

```bash
python -m venv venv
````

### 2️⃣ Activate it

  * **Windows:**
    ```cmd
    venv\Scripts\activate
    ```
  * **Mac/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (Optional)

*The repository comes with a pre-trained model, but if you want to retrain:*

```bash
python model_training.py
```

### 5️⃣ Run Streamlit app

```bash
streamlit run app.py
```

-----

## 🧠 Machine Learning Pipeline

1.  **Preprocessing & Cleaning**

      * **Renaming:** Standardization of column names (e.g., `wc` → `wbcc`).
      * **Type Conversion:** Explicit string conversion for categorical features.
      * **Imputation:** Median filling for numerical gaps; Mode filling for categorical gaps.

2.  **Feature Engineering**

      * **BP Deviation (`bp_diff`):** Calculates the absolute difference from a normal blood pressure of 80 mmHg. This ensures the model treats Hypotension (Low BP) as a risk factor, similar to Hypertension.

3.  **Model Training**

      * **Algorithm:** Random Forest Classifier (`n_estimators=300`, `max_depth=8`).
      * **Balancing:** Uses `class_weight='balanced'` to strictly handle the imbalance between CKD and Non-CKD samples without synthetic data (SMOTE removed).
      * **Calibration:** Wrapped in `CalibratedClassifierCV` (Sigmoid) to smooth probability outputs.

4.  **Inference (App Logic)**

      * Accepts user inputs.
      * Imputes missing *optional* fields with a **Healthy Profile** default.
      * Scales and Encodes data using the saved `preprocessing.pkl`.
      * Returns a risk percentage and category (Green/Yellow/Red).

-----

## 📦 Tech Stack

  * **Python**
  * **Pandas, NumPy** (Data Manipulation)
  * **Scikit-learn** (ML & Preprocessing)
  * **Streamlit** (Frontend)
  * **Joblib** (Model Serialization)
  * **Matplotlib** (Visualization)

-----

## 📜 Dataset

The model is trained on the **Chronic Kidney Disease dataset (UCI Repository)** containing **400 samples** and **25 medical attributes**, including age, blood pressure, specific gravity, albumin, sugar, red blood cells, pus cell, pus cell clumps, bacteria, blood glucose random, blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count, hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, and anemia.

-----

## 🙋‍♀️ Author

**Khushi Goyal**
GitHub: [@khushigoyal05](https://github.com/khushigoyal05)

**Shambhavi**
GitHub: [@shambhavi-coder](https://github.com/shambhavi-coder)

```
