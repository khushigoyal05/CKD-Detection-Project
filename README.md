🚑 Early Detection of Chronic Kidney Disease (CKD)

A Machine Learning–powered web application that predicts the likelihood of Chronic Kidney Disease (CKD) based on patient medical parameters.
The project includes:

✔ Data preprocessing

✔ Exploratory Data Analysis (EDA)

✔ Multiple ML models comparison

✔ Best model selection

✔ An interactive Streamlit UI

✔ Multi-page web app (Home, EDA, Model Performance, About)

🌐 Live Demo (Optional)

Add link here after deployment:
👉 Coming Soon

📊 Features
🔹 1. User-friendly Web Interface

Numeric & categorical medical inputs

Automatic preprocessing

Real-time CKD prediction

🔹 2. EDA Dashboard

Dataset preview

Missing value visualization

Normalized numerical data

Summary statistics

🔹 3. Model Performance Comparison

Logistic Regression

KNN

Decision Tree

Random Forest

SVM

Gradient Boosting

SVM achieved 100% accuracy, but Logistic Regression selected as best generalizable model.

📁 Project Structure
CKD_Detection_Project/
│── app.py
│── preprocess_data.py
│── eda_overview.py
│── model_training.py
│── convert_arff_to_csv.py
│── requirements.txt
│── README.md
│── data/
│── models/
│── pages/
│     ├── EDA.py
│     ├── Model_Performance.py
│     └── About.py

🚀 How to Run Locally
1️⃣ Create virtual environment
python -m venv venv

2️⃣ Activate it
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit app
streamlit run app.py

🧠 Machine Learning Pipeline

Preprocessing

Missing value imputation

Encoding categorical columns

Normalization

Saving preprocessing objects (scaler, encoder)

Model Training

Trains 6 models

Calculates accuracy, precision, recall, F1, AUC

Saves best model → models/best_model.pkl

Inference

User inputs → preprocessing → model predicts CKD / Not CKD

📦 Tech Stack

Python

Pandas, NumPy

Scikit-learn

Streamlit

Pickle

Matplotlib / Seaborn

📜 Dataset

The model is trained on the Chronic Kidney Disease dataset (UCI Repository) with 400 samples & 25 medical attributes.

🙋‍♀️ Author

Khushi Goyal
GitHub: @khushigoyal05

Shambhavi
GitHub: @shambhavi-coder