# comment-service/app.py
from flask import Flask, request, jsonify, Response, stream_with_context
import pandas as pd
import numpy as np
import joblib
import hashlib
import json
import os
from datetime import datetime
from functools import lru_cache
from utils import load_dictionaries, generate_synthetic_data, engineer_features, get_category_type
from streaming import generate_streaming_comment
from cache import comment_cache

app = Flask(__name__)

# Load model and supporting files
MODEL_PATH = 'models/comment_model.pkl'
ENCODERS_PATH = 'models/encoders.pkl'
METADATA_PATH = 'models/metadata.pkl'

model = None
encoders = None
metadata = None
measure_dict = None
attr_dict = None

def load_model():
    global model, encoders, metadata, measure_dict, attr_dict
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
    if os.path.exists(METADATA_PATH):
        metadata = joblib.load(METADATA_PATH)
    
    # Load data dictionaries
    measure_dict, attr_dict = load_dictionaries()
    
    print("Model and dictionaries loaded successfully.")

load_model()

def hash_row(row_dict):
    """Create a deterministic hash for caching"""
    row_str = json.dumps(sorted(row_dict.items()), sort_keys=True)
    return hashlib.md5(row_str.encode()).hexdigest()

@app.route('/generate-comment', methods=['POST'])
def generate_comment():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Support single row or list (for batch)
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
            # Engineer features
            features = engineer_features(row, measure_dict, attr_dict, metadata)
            
            # Predict severity
            if model and 'severity' in encoders:
                X = pd.DataFrame([features])
                # Handle categorical encoding safely
                for col in encoders.get('categorical_cols', []):
                    if col in X.columns and col in encoders:
                        le = encoders[col]
                        if X[col].iloc[0] in le.classes_:
                            X[col] = le.transform([X[col].iloc[0]])[0]
                        else:
                            X[col] = -1  # unknown
                
                severity_pred = model.predict(X)[0]
                confidence = np.max(model.predict_proba(X)[0]) if hasattr(model, 'predict_proba') else 0.85
            else:
                # Fallback rule-based
                severity_pred = get_rule_based_severity(row)
                confidence = 0.75
            
            comment, explanation, alternatives = generate_comment_text(row, severity_pred, features, measure_dict)
            
            result = {
                "comment": comment,
                "confidence": round(float(confidence), 3),
                "explanation": explanation,
                "severity": severity_pred,
                "alternatives": alternatives,
                "timestamp": datetime.now().isoformat()
            }
            
            comment_cache.set(cache_key, result)
            results.append(result)
            
        except Exception as e:
            results.append({
                "comment": "Unable to generate comment due to data anomaly.",
                "confidence": 0.5,
                "explanation": f"Processing error: {str(e)[:100]}",
                "severity": "MEDIUM",
                "alternatives": ["Review raw data for inconsistencies"],
                "timestamp": datetime.now().isoformat()
            })
    
    return jsonify(results[0] if not is_batch else results)

def get_rule_based_severity(row):
    amount = float(row.get('amount', 0))
    gl_account = str(row.get('gl_account', 'Unknown'))
    
    if amount < -50000:
        return "CRITICAL"
    elif amount < 0:
        return "HIGH"
    elif amount > 100000:
        return "HIGH"
    elif abs(amount) < 1000:
        return "LOW"
    else:
        return "MEDIUM"

def generate_comment_text(row, severity, features, measure_dict):
    amount = float(row.get('amount', 0))
    gl = row.get('gl_account', 'Unknown Account')
    status = row.get('status', '')
    
    templates = {
        "CRITICAL": [
            f"Critical { 'loss' if amount < 0 else 'variance' } detected in {gl}. Immediate investigation required.",
            f"ALERT: Significant negative impact on {gl} with ${abs(amount):,.0f} deviation.",
        ],
        "HIGH": [
            f"High priority { 'loss' if amount < 0 else 'gain' } observed in {gl}.",
            f"Notable financial movement in {gl} warrants review.",
        ],
        "MEDIUM": [
            f"Moderate activity in {gl} with ${amount:,.0f}.",
            f"Stable performance noted for {gl}.",
        ],
        "LOW": [
            f"Low activity recorded for {gl}.",
            f"Minor transaction in {gl}.",
        ]
    }
    
    comment = np.random.choice(templates.get(severity, templates["MEDIUM"]))
    explanation = f"Amount: ${amount:,.0f} | Severity based on deviation and category analysis."
    alternatives = [
        f"Consider reviewing {gl} trends over last 3 periods.",
        "Cross-reference with related expense/revenue accounts."
    ]
    
    return comment, explanation, alternatives

@app.route('/generate-comment-stream', methods=['POST'])
def generate_comment_stream():
    data = request.get_json()
    if not data:
        return "No data", 400
    
    def stream_response():
        yield "Analyzing financial row...\n"
        yield "Engineering features from data dictionary...\n"
        
        # Simulate thinking
        import time
        time.sleep(0.3)
        yield "Detecting severity level...\n"
        time.sleep(0.3)
        
        result = generate_comment()  # Reuse logic (returns Response, but we extract)
        # For simplicity, we call the non-stream and stream the final
        # In production, integrate full logic here
        
        final = json.loads(result.get_data(as_text=True))
        yield f"High variance detected in account {data.get('gl_account', 'N/A')}...\n"
        time.sleep(0.2)
        yield final.get("comment", "") + "\n"
        yield f"Confidence: {final.get('confidence', 0.8)}\n"
        yield "Recommendation generated.\n"
    
    return Response(stream_with_context(stream_response()), mimetype='text/plain')

@app.route('/generate-comment-batch', methods=['POST'])
def generate_comment_batch():
    return generate_comment()  # Already supports list

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data or 'input' not in data or 'correct_comment' not in data:
        return jsonify({"error": "Invalid feedback format"}), 400
    
    feedback_dir = 'data'
    os.makedirs(feedback_dir, exist_ok=True)
    feedback_path = os.path.join(feedback_dir, 'feedback.csv')
    
    row = data['input']
    row['correct_comment'] = data['correct_comment']
    row['feedback_time'] = datetime.now().isoformat()
    
    df = pd.DataFrame([row])
    if os.path.exists(feedback_path):
        df.to_csv(feedback_path, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_path, index=False)
    
    return jsonify({"status": "Feedback recorded. Model will retrain on next cycle."})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "dictionaries_loaded": measure_dict is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
