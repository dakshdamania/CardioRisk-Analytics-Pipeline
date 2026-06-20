# 🏥 CardioRisk-Analytics-Pipeline

An enterprise-grade, end-to-end clinical data engineering and machine learning system. This project demonstrates how to structure messy clinical data into a relational SQL database, extract features via optimized query scripts, train a high-performance tree-based classifier, and deploy a functional web interface for healthcare professionals.

## 🚀 Key Highlights & Impact
* **Relational Storage:** Migrated raw healthcare records into a structured SQLite relational database framework.
* **Tuned for Clinical Safety:** Fine-tuned an XGBoost classifier prioritizing **Recall (95%)** over basic metrics to actively minimize dangerous medical false-negatives.
* **Production Grade Evaluation:** Achieved a **95% general classification accuracy** and a **98.95% ROC-AUC evaluation target**.
* **Interactive UI Assistant:** Deployed a responsive frontend server using Streamlit to simulate real-time patient risk evaluations in a clinic.

## 🛠️ Tech Stack & Architecture
* **Data Engineering:** SQL (SQLite), Python, Pandas, SQLAlchemy
* **Predictive Layer:** Scikit-Learn, XGBoost Classifier, Joblib
* **Deployment & Frontend:** Streamlit Framework, Git Workflow

## 📊 Pipeline Workflow
1. **`database_setup.py`**: Initializes a secure relational table database schema and streams data profiles inside local storage.
2. **`train_model.py`**: Executes advanced feature scaling, splits cohorts stratifically, fits the XGBoost network weights, and exports binary configurations.
3. **`app.py`**: Loads the trained model matrix and sets up the interactive doctor diagnostic dashboard environment.

## 📈 Final Model Validation Metrics
```text
=== PIPELINE DIAGNOSTIC REPORT ===
              precision    recall  f1-score   support
           0       0.95      0.94      0.94       100
           1       0.94      0.95      0.95       105

    Accuracy: 95%
    Pipeline ROC-AUC Performance Target: 0.9895
```

## ⚙️ Local Installation & Setup
```bash
# 1. Clone the repository
git clone https://github.com
cd CardioRisk-Analytics-Pipeline

# 2. Install library configurations
pip install -r requirements.txt

# 3. Process database extraction & model training
python database_setup.py
python train_model.py

# 4. Launch the local portal instance
streamlit run app.py
```
