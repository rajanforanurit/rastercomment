import pandas as pd
import numpy as np
import os
import random
from datetime import datetime

def load_dictionaries(silent=False):
    """Load REBA Data Dictionaries for rich commenting"""
    measures_path = 'Data Dictionary - Measures.xlsx'
    attrs_path = 'Data Dictionary - Attributes.xlsx'
    
    measure_dict = {}
    attr_dict = {}
    rich_mode = False
    
    if os.path.exists(measures_path) and os.path.exists(attrs_path):
        try:
            measure_df = pd.read_excel(measures_path, sheet_name=0, header=1)
            attr_df = pd.read_excel(attrs_path, sheet_name=0, header=1)
            
            for _, row in measure_df.iterrows():
                if pd.notna(row.get('Measure Name')):
                    measure_name = str(row['Measure Name']).strip()
                    measure_dict[measure_name] = {
                        'description': str(row.get('Description', '')),
                        'formula': str(row.get('Formula', '')),
                        'table': str(row.get('Table Name', ''))
                    }
            
            for _, row in attr_df.iterrows():
                if pd.notna(row.get('Attribute Name')):
                    attr_name = str(row['Attribute Name']).strip()
                    attr_dict[attr_name] = {
                        'description': str(row.get('Description', '')),
                        'table': str(row.get('Connected Fact Tables', ''))
                    }
            
            rich_mode = True
            if not silent:
                print(f"✅ RICH MODE ACTIVATED: Loaded {len(measure_dict)} measures and {len(attr_dict)} attributes.")
                
        except Exception as e:
            if not silent:
                print(f"⚠️ Failed to load Excel files: {e}. Running in safe fallback mode.")
    else:
        if not silent:
            print("⚠️ Data Dictionary Excel files not found. Running in SAFE FALLBACK mode.")
    
    return measure_dict, attr_dict, rich_mode


def get_category_from_dict(gl_account, measure_dict):
    """Use dictionary for better category detection"""
    if not measure_dict:
        return get_simple_category_fallback(gl_account)
    
    gl_lower = str(gl_account).lower()
    
    for measure_name, info in measure_dict.items():
        name_lower = measure_name.lower()
        desc_lower = info['description'].lower()
        
        if gl_lower in name_lower or gl_lower in desc_lower:
            desc = info['description'].lower()
            if any(word in desc for word in ['rent', 'revenue', 'income', 'gross', 'effective']):
                return 'revenue'
            elif any(word in desc for word in ['concession', 'discount', 'expense', 'cost', 'payroll', 'maintenance']):
                return 'expense'
            elif 'profit' in desc:
                return 'profit'
    return get_simple_category_fallback(gl_account)


def get_simple_category_fallback(gl_account):
    """Safe fallback when dictionaries are not loaded"""
    gl_lower = str(gl_account).lower()
    if any(x in gl_lower for x in ['rent', 'revenue', 'income', 'sales', 'gross', 'effective']):
        return 'revenue'
    elif any(x in gl_lower for x in ['expense', 'cost', 'payroll', 'maintenance', 'utility', 'concession', 'insurance', 'tax']):
        return 'expense'
    elif any(x in gl_lower for x in ['profit', 'margin', 'net']):
        return 'profit'
    return 'other'


def engineer_features(row, measure_dict, attr_dict, rich_mode):
    """Safe feature engineering"""
    try:
        amount = float(row.get('amount', 0))
    except (TypeError, ValueError):
        amount = 0.0
    
    gl = str(row.get('gl_account', 'Unknown'))
    
    # Get category safely
    category = get_category_from_dict(gl, measure_dict) if rich_mode else get_simple_category_fallback(gl)
    
    features = {
        'abs_amount': abs(amount),
        'is_loss': 1 if amount < 0 else 0,
        'amount_sign': np.sign(amount),
        'log_amount': np.log1p(abs(amount) + 1),
        'category_type': category,           # This will be encoded later
        'gl_length': len(gl),
        'is_high_value': 1 if abs(amount) > 75000 else 0,
        'is_low_value': 1 if abs(amount) < 5000 else 0,
        'deviation_from_mean': random.uniform(-2.5, 2.5),
        'z_score': random.uniform(-3.0, 3.0),
        'rolling_trend': random.choice([-1, 0, 1]),
    }
    return features


def generate_synthetic_data(n=12000, measure_dict=None, rich_mode=False):
    """Generate synthetic data for training"""
    np.random.seed(42)
    data = []
    
    gl_accounts = ["Effective Rent", "Gross Rent", "Lease Concession", "Maintenance Expense", 
                   "Payroll Expense", "Marketing Cost", "Property Tax", "Insurance", "Unknown GL"]
    
    for _ in range(n):
        gl = np.random.choice(gl_accounts)
        amount = np.random.normal(0, 65000)
        
        # Bias expenses and concessions toward negative
        if any(x in gl for x in ["Expense", "Concession", "Tax", "Insurance", "Maintenance"]):
            amount = -abs(amount) * random.uniform(0.9, 1.6)
        
        row = {
            'gl_account': gl,
            'amount': round(amount, 2)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df['severity'] = df['amount'].apply(lambda x: 
        'CRITICAL' if x < -100000 else
        'HIGH' if x < -20000 else
        'HIGH' if x > 150000 else
        'MEDIUM' if abs(x) > 30000 else 'LOW'
    )
    return df
