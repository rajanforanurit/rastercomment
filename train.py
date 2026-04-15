# comment-service/train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
from utils import generate_synthetic_data, engineer_features

def train_model():
    print("Generating synthetic training data...")
    df = generate_synthetic_data(15000)
    
    print("Engineering features...")
    X_list = []
    y = df['severity']
    
    for _, row in df.iterrows():
        feats = engineer_features(row, {}, {}, None)  # dicts not needed for training
        X_list.append(feats)
    
    X = pd.DataFrame(X_list)
    
    # Encode categoricals
    categorical_cols = ['category_type']
    encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/comment_model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump({
        'feature_names': list(X.columns),
        'categorical_cols': categorical_cols,
        'training_date': pd.Timestamp.now().isoformat()
    }, 'models/metadata.pkl')
    
    print("Model saved to models/ directory.")

if __name__ == "__main__":
    train_model()
