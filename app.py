from flask import Flask, request, jsonify, Response, stream_with_context
import pandas as pd
import numpy as np
import joblib
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache

# Import only existing functions from utils
from utils import (
    load_dictionaries, 
    generate_synthetic_data, 
    engineer_features,
    get_category_from_dict,      # ← Updated
    get_simple_category_fallback # ← Added
)

from cache import comment_cache

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
    
    # Load dictionaries first
    measure_dict, attr_dict, rich_mode = load_dictionaries(silent=False)
    
    # Load trained model
    if os.path.exists('models/comment_model.pkl'):
        model = joblib.load('models/comment_model.pkl')
        encoders = joblib.load('models/encoders.pkl') if os.path.exists('models/encoders.pkl') else {}
        metadata = joblib.load('models/metadata.pkl') if os.path.exists('models/metadata.pkl') else {}
        print(f"✅ Model loaded successfully. Rich Mode: {'Yes' if rich_mode else 'No'}")
    else:
        print("⚠️ Model not found. Please run 'python train.py' locally first.")

load_model_and_dictionaries()

def hash_row(row_dict):
    row_str = json.dumps(sorted(row_dict.items()), sort_keys=True)
    return hashlib.md5(row_str.encode()).hexdigest()

@app.route('/generate-comment', methods=['POST'])
def generate_comment():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    is_batch = isinstance(data, list)
    inputs = data if is_batch else [data]
    
    results = []
    for row in inputs:
        cache_key = hash_row(row)
        cached = comment_cache.get(cache_key)
        if cached:
            results.append(cached)
            continue
        
        try:
            features = engineer_features(row, measure_dict, attr_dict, rich_mode)
            
            if model and encoders:
                X = pd.DataFrame([features])
                for col in encoders.get('categorical_cols', []):
                    if col in X.columns and col in encoders:
                        le = encoders[col]
                        val = str(X[col].iloc[0])
                        X[col] = le.transform([val])[0] if val in le.classes_ else -1
                
                severity_pred = model.predict(X)[0]
                confidence = float(np.max(model.predict_proba(X)[0])) if hasattr(model, 'predict_proba') else 0.82
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
            
            comment_cache.set(cache_key, result)
            results.append(result)
            
        except Exception as e:
            results.append({
                "comment": "Unable to generate detailed comment at this time.",
                "confidence": 0.6,
                "explanation": f"Processing error: {str(e)[:80]}",
                "severity": "MEDIUM",
                "alternatives": ["Please check the input data"],
                "timestamp": datetime.now().isoformat()
            })
    
    return jsonify(results[0] if not is_batch else results)

def generate_comment_text(row, severity, features):
    amount = float(row.get('amount', 0))
    gl = row.get('gl_account', 'Unknown Account')
    
    templates = {
        "CRITICAL": [
            f"**CRITICAL**: Significant loss of ${abs(amount):,.0f} detected in {gl}. Immediate review required.",
            f"ALERT: Major negative impact found in account {gl}. Action needed urgently."
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
            f"Low value transaction recorded in {gl}.",
            f"Minor activity in {gl}."
        ]
    }
    
    comment = np.random.choice(templates.get(severity, templates["MEDIUM"]))
    explanation = f"Amount: ${amount:,.0f} | Category: {features.get('category_type')} | Severity based on deviation analysis."
    alternatives = ["Review trend over last 3 periods", "Cross-check with related accounts"]
    
    return comment, explanation, alternatives

# Keep other routes (health, feedback, streaming) same as before
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "rich_mode": rich_mode
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
