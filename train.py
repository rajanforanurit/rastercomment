from utils import load_dictionaries, generate_synthetic_data, engineer_features
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    print("=== Starting Training with REBA Dictionaries ===")
    measure_dict, attr_dict, rich_mode = load_dictionaries(silent=False)
    
    print("Generating synthetic training data...")
    df = generate_synthetic_data(15000, measure_dict, rich_mode)
    
    print("Engineering features...")
    X_list = []
    y = df['severity']
    
    for _, row in df.iterrows():
        feats = engineer_features(row, measure_dict, attr_dict, rich_mode)
        X_list.append(feats)
    
    X = pd.DataFrame(X_list)
    
    # Encode categorical features
    categorical_cols = ['category_type']
    encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=2, 
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained successfully! Test Accuracy: {accuracy:.4f}")
    print(f"   Rich Mode during training: {'Yes' if rich_mode else 'No'}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/comment_model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump({
        'feature_names': list(X.columns),
        'categorical_cols': categorical_cols,
        'rich_mode_used': rich_mode,
        'training_date': datetime.now().isoformat()
    }, 'models/metadata.pkl')
    
    print("✅ Model and metadata saved to /models")

if __name__ == "__main__":
    train_model()
