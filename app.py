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
        print("⚠️ Model not found.")

load_model_and_dictionaries()

def hash_row(row_dict):
    row_str = json.dumps(sorted(row_dict.items()), sort_keys=True)
    return hashlib.md5(row_str.encode()).hexdigest()

@app.route('/generate-comment', methods=['POST'])
def generate_comment():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    inputs = data if isinstance(data, list) else [data]
    results = []

    for row in inputs:
        try:
            features = engineer_features(row, measure_dict, attr_dict, rich_mode)

            if model is not None and encoders:
                X = pd.DataFrame([features])
                
                categorical_cols = encoders.get('categorical_cols', [])
                for col in categorical_cols:
                    if col in X.columns:
                        le = encoders[col]
                        val = str(X[col].iloc[0])
                        if val in le.classes_:
                            X[col] = le.transform([val])[0]
                        else:
                            X[col] = -1
                
                for col in X.columns:
                    if col not in categorical_cols:
                        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
                severity_pred = model.predict(X)[0]
                confidence = float(np.max(model.predict_proba(X)[0]))
            else:
                severity_pred = "MEDIUM"
                confidence = 0.75

            comment, explanation, alternatives = generate_sophisticated_comment(row, severity_pred, features)

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
            print(f"ERROR: {str(e)}")
            results.append({
                "comment": "Unable to generate detailed comment at this time.",
                "confidence": 0.6,
                "explanation": f"Processing error: {str(e)[:100]}",
                "severity": "MEDIUM",
                "alternatives": ["Please ensure 'amount' is numeric"],
                "rich_mode": rich_mode,
                "timestamp": datetime.now().isoformat()
            })

    return jsonify(results[0] if len(results) == 1 else results)


def generate_sophisticated_comment(row, severity, features):
    try:
        amount = float(row.get('amount', 0))
    except:
        amount = 0.0
    
    gl_account = str(row.get('gl_account', 'Unknown Account')).strip()
    
    # Get rich description from dictionary if available
    measure_info = measure_dict.get(gl_account, {}) if measure_dict else {}
    description = measure_info.get('description', '') or gl_account
    table_name = measure_info.get('table', 'Unknown')
    
    # Smart comment generation
    if severity == "CRITICAL":
        if amount < 0:
            comment = f"**CRITICAL ALERT**: Significant loss of ${abs(amount):,.0f} detected in **{gl_account}** ({description}). Immediate investigation required."
        else:
            comment = f"**CRITICAL**: Unusual high positive variance of ${amount:,.0f} in **{gl_account}**. Verify data integrity."
    elif severity == "HIGH":
        direction = "loss" if amount < 0 else "gain"
        comment = f"**High Priority**: Notable {direction} of ${abs(amount):,.0f} observed in **{gl_account}** ({description}). Recommend detailed review."
    else:
        comment = f"Moderate activity of ${amount:,.0f} recorded in **{gl_account}** ({description})."
    
    explanation = f"Amount: ${amount:,.0f} | Category: {features.get('category_type', 'other').capitalize()} | Table: {table_name}"
    
    alternatives = [
        "Review this metric over the last 3 periods for trends",
        "Cross-reference with related measures (e.g., Gross Rent vs Effective Rent)",
        "Check for recent concessions, discounts, or adjustments"
    ]
    
    if "Rent" in gl_account:
        alternatives.append("Validate lease terms and concession amortization")
    elif "Concession" in gl_account:
        alternatives.append("Review impact on Effective Rent calculations")
    
    return comment, explanation, alternatives


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "rich_mode": rich_mode
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
