# train_model.py
import sqlite3
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# 1. Pull data directly via structured SQL query parameters
conn = sqlite3.connect('hospital_records.db')
query = "SELECT age, sex, cp, trestbps, chol, thalach, oldpeak, target FROM patient_records"
df = pd.read_sql_query(query, conn)
conn.close()

# 2. Segment feature matrices from prediction arrays
X = df.drop(columns=['target'])
y = df['target']

# 3. Stratified Splitting to balance health metric profiles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Standard Scaler transformations for numeric normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Fit an XGBoost Production Classifier
model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Output clinical metrics for your resume profile
y_pred = model.predict(X_test_scaled)
print("\n=== PIPELINE DIAGNOSTIC REPORT ===")
print(classification_report(y_test, y_pred))
print(f"Pipeline ROC-AUC Performance Target: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}\n")

# 7. Serialize and output pipeline configurations
joblib.dump(model, 'xgb_cardio_model.pkl')
joblib.dump(scaler, 'data_scaler.pkl')
print("Trained pipeline weights saved to production binaries successfully!")
