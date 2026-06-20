# database_setup.py
import sqlite3
import pandas as pd

# 1. Connect to local relational SQLite architecture
conn = sqlite3.connect('hospital_records.db')
cursor = conn.cursor()

# 2. Build patient clinical schemas
cursor.execute('''
CREATE TABLE IF NOT EXISTS patient_records (
    age INTEGER, sex INTEGER, cp INTEGER, trestbps INTEGER, 
    chol INTEGER, fbs INTEGER, restecg INTEGER, thalach INTEGER, 
    exang INTEGER, oldpeak REAL, slope INTEGER, ca INTEGER, 
    thal INTEGER, target INTEGER
)
''')
conn.commit()

# 3. Stream Kaggle parameters cleanly into SQL tables
df = pd.read_csv('heart.csv')
df.to_sql('patient_records', conn, if_exists='replace', index=False)

print(f"Securely populated {len(df)} patient profiles inside the local SQL engine!")
conn.close()
