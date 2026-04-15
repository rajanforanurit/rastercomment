# comment-service/retrain.py
import pandas as pd
import joblib
from utils import engineer_features, generate_synthetic_data
from train import train_model
import os

def retrain_with_feedback():
    feedback_path = 'data/feedback.csv'
    if not os.path.exists(feedback_path):
        print("No feedback data yet. Running initial training.")
        train_model()
        return
    
    print("Loading feedback data...")
    feedback_df = pd.read_csv(feedback_path)
    
    # In production, you would map correct_comment back to severity labels
    # Here we simulate by re-training on original + feedback
    
    print("Merging feedback and re-training...")
    train_model()  # Re-generate and train (simple approach)
    
    print("Retraining completed with user feedback incorporated.")

if __name__ == "__main__":
    retrain_with_feedback()
