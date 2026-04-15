# comment-service/app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import hashlib
import json
import os
from datetime import datetime

from utils import load_dictionaries, engineer_features

app = Flask(__name__)

# Global variables
model = None
encoders = None
metadata = None
measure_dict = None
attr_dict = None
rich_mode = False

def load_model_and_dictionaries():
    global model, encoders, metadata, measure_dict, attr_dict, rich_mode
    measure_dict, attr_dict, rich_mode = load_dictionaries(silent=False)
    
    if os.path.exists('models/comment_model.pkl'):
        model = joblib.load('models/comment_model.pkl')
        encoders = joblib.load('models/encoders.pkl') if os.path.exists('models/encoders.pkl') else {}
        metadata = joblib.load('models/metadata.pkl') if os.path.exists('models/metadata.pkl') else {}
        print(f"✅ Model loaded | Rich Mode: {'Yes' if rich_mode else 'No'}")
    else:
        print("⚠️ Model not found. Run 'python train.py' locally.")

load_model_and_dictionaries()

def hash_row(row_dict):
    row_str = json.dumps(sorted(row_dict.items()), sort_keys=True)
    return hashlib.md5(row_str.encode()).hexdigest()

@app.route('/generate-comment', methods=['POST'])
def generate_comment():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Support both single object and array
    inputs = data if isinstance(data, list) else [data]
    results = []

    for row in inputs:
        try:
            # Engineer features safely
            features = engineer_features(row, measure_dict, attr_dict, rich_mode)

            if model is not None and encoders:
                # Convert to DataFrame and ensure correct types
                X = pd.DataFrame([features])
                
                # Safe encoding for categorical columns
                for col in encoders.get('categorical_cols', []):
                    if col in X.columns:
                        le = encoders[col]
                        val = str(X[col].iloc[0])
                        if val in le.classes_:
                            X[col] = le.transform([val])[0]
                        else:
                            X[col] = -1  # unknown category

                # Ensure all columns are numeric
                X = X.astype(float)

                severity_pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
            else:
                severity_pred = "MEDIUM"
                confidence = 0.70

            comment, explanation, alternatives = generate_comment_text(row, severity_pred, features)

            result = {
                "comment": comment,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "severity": severity_pred,
                "alternatives": alternatives,
                "rich_mode": rich_mode,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing row: {e}")   # Log for debugging
            results.append({
                "comment": "Unable to generate detailed comment at this time.",
                "confidence": 0.6,
                "explanation": f"Processing error: {str(e)[:100]}",
                "severity": "MEDIUM",
                "alternatives": ["Please check the input data format"],
                "rich_mode": rich_mode,
                "timestamp": datetime.now().isoformat()
            })

    return jsonify(results[0] if len(results) == 1 else results)


def generate_comment_text(row, severity, features):
    amount = float(row.get('amount', 0))
    gl = str(row.get('gl_account', 'Unknown Account'))
    
    templates = {
        "CRITICAL": [
            f"**CRITICAL**: Significant loss of ${abs(amount):,.0f} detected in {gl}. Immediate review required.",
            f"ALERT: Major negative impact found in account {gl}."
        ],
        "HIGH": [
            f"High priority variance of ${amount:,.0f} observed in {gl}.",
            f"Notable financial movement detected in {gl} — recommend review."
        ],
        "MEDIUM": [
            f"Moderate activity recorded in {gl} (${amount:,.0f}).",
            f"Stable performance noted for account {gl}."
        ],
        "LOW": [
            f"Low value transaction recorded in {gl}."
        ]
    }
    
    comment = np.random.choice(templates.get(severity, templates["MEDIUM"]))
    explanation = f"Amount: ${amount:,.0f} | Category: {features.get('category_type', 'other')} | Rich Mode: {rich_mode}"
    
    return comment, explanation, ["Review recent trends", "Cross-check with related accounts"]


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "rich_mode": rich_mode
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
