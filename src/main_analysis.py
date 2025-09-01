# In src/main_analysis.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def load_epa_data():
    data_files = glob.glob("data/epa_data_processed_*.csv")
    if not data_files:
        print("No EPA data found. Run download_epa_data.py first.")
        return None
    df = pd.read_csv(max(data_files))
    return df

def generate_microplastic_labels(df):
    # Original label logic
    features = {}
    features['turbidity_risk'] = df['turbidity_ntu'] / df['turbidity_ntu'].max()
    features['oxygen_risk'] = 1 / (df['dissolved_o2_mg_l'] + 1)
    features['temp_risk'] = np.abs(df['temp_c'] - 15) / 20
    features['ph_risk'] = np.abs(df['ph'] - 7) / 3
    features['flow_risk'] = np.where(df['flow_rate_cfs'] < 1, 0.5, 0.1)

    risk_score = (
        0.35 * features['turbidity_risk'] +
        0.25 * features['oxygen_risk'] +
        0.20 * features['temp_risk'] +
        0.15 * features['ph_risk'] +
        0.05 * features['flow_risk']
    )
    risk_score += np.random.normal(0, 0.1, len(df))
    risk_score = np.clip(risk_score, 0, 1)
    threshold = np.percentile(risk_score, 70)
    labels = (risk_score > threshold).astype(int)
    return labels, risk_score

def prepare_features(df):
    feature_names = ['temp_c', 'ph', 'turbidity_ntu', 'dissolved_o2_mg_l', 'flow_rate_cfs']
    X = df[feature_names].fillna(df[feature_names].median())
    return X, feature_names

def train_sensor_model(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    return model

def main():
    df = load_epa_data()
    if df is None: return
    y, _ = generate_microplastic_labels(df)
    X, features = prepare_features(df)
    model = train_sensor_model(X, y, features)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, "models/sensor_model.joblib")
    print("Model saved to models/sensor_model.joblib")

if __name__ == "__main__":
    main()
