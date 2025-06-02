import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load custom dataset
df = pd.read_csv('data/custom_diabetes_data.csv')

# Features and label
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, 'models/diabetes_model.pkl')
print("✅ Custom model trained and saved!")
