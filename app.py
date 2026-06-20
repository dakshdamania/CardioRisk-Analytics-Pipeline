# app.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# --- SELF-HEALING BLOCK FOR CLOUD DEPLOYMENT ---
# If the model weights aren't found, this block runs the training script instantly on the cloud server
if not os.path.exists('xgb_cardio_model.pkl') or not os.path.exists('data_scaler.pkl'):
    st.info("⚡ Initializing Cloud Machine Learning Pipeline... Please wait a few seconds.")
    
    # 1. Fetch data from secure cloud backup link
    dataset_url = "https://githubusercontent.com"
    df_raw = pd.read_csv(dataset_url)
    
    # 2. Build local SQLite table on the cloud container
    conn = sqlite3.connect('hospital_records.db')
    df_raw.to_sql('patient_records', conn, if_exists='replace', index=False)
    
    # 3. Pull required features out via SQL query
    query = "SELECT age, sex, cp, trestbps, chol, thalach, oldpeak, target FROM patient_records"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 4. Train the model weights instantly
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y)
    
    # 5. Save binaries to local cloud disk
    joblib.dump(model, 'xgb_cardio_model.pkl')
    joblib.dump(scaler, 'data_scaler.pkl')
    st.success("🎉 Cloud Pipeline Successfully Generated!")

# --- END OF SELF-HEALING BLOCK ---

# 1. Map production configuration assets
model = joblib.load('xgb_cardio_model.pkl')
scaler = joblib.load('data_scaler.pkl')

# 2. Layout configuration controls
st.set_page_config(page_title="CardioRisk Analytics", layout="centered")
st.title("🏥 Clinical Cardiovascular AI Diagnostic Portal")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 1, 100, 45)
    sex = st.selectbox("Biological Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                      format_func=lambda x: f"Type {x}: " + ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)

with col2:
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.slider("ST Depression (ECG Metric)", 0.0, 6.2, 1.0, step=0.1)

# 3. Vectorize user interface run-time data payloads
input_data = pd.DataFrame([{
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 
    'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak
}])

# 4. Process real-time prediction workflows upon interaction triggers
if st.button("Calculate Patient Risk Probability", type="primary"):
    scaled_input = scaler.transform(input_data)
    probability = model.predict_proba(scaled_input)[0][1]
    prediction = model.predict(scaled_input)[0]
    
    st.markdown("### **Diagnostic Evaluation:**")
    if prediction == 1:
        st.error(f"⚠️ **High Cardiovascular Risk Confirmed.** Risk Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ **Low Clinical Risk Profile.** Risk Probability: {probability*100:.2f}%")
