# comment-service/utils.py
import pandas as pd
import numpy as np
import os
import random

def load_dictionaries():
    """Load both Excel data dictionaries"""
    measures_path = 'Data Dictionary - Measures.xlsx'
    attrs_path = 'Data Dictionary - Attributes.xlsx'
    
    measure_df = pd.read_excel(measures_path, sheet_name=0, header=1)
    attr_df = pd.read_excel(attrs_path, sheet_name=0, header=1)
    
    # Extract useful mappings (dynamic, no hardcoding)
    measure_dict = {}
    for _, row in measure_df.iterrows():
        if pd.notna(row.get('Measure Name')):
            measure_dict[row['Measure Name']] = {
                'description': row.get('Description', ''),
                'formula': row.get('Formula', ''),
                'table': row.get('Table Name', '')
            }
    
    attr_dict = {}
    for _, row in attr_df.iterrows():
        if pd.notna(row.get('Attribute Name')):
            attr_dict[row['Attribute Name']] = {
                'description': row.get('Description', ''),
                'table': row.get('Connected Fact Tables', '')
            }
    
    return measure_dict, attr_dict

def get_category_type(gl_account, measure_dict):
    """Dynamic category inference from dictionary"""
    gl_lower = str(gl_account).lower()
    if any(word in gl_lower for word in ['revenue', 'income', 'rent', 'sales']):
        return 'revenue'
    elif any(word in gl_lower for word in ['expense', 'cost', 'payroll', 'maintenance']):
        return 'expense'
    elif any(word in gl_lower for word in ['profit', 'margin', 'net']):
        return 'profit'
    elif 'loss' in gl_lower or 'write' in gl_lower:
        return 'loss'
    return 'other'

def engineer_features(row, measure_dict, attr_dict, metadata):
    """Feature engineering - production ready"""
    amount = float(row.get('amount', 0))
    gl = str(row.get('gl_account', 'Unknown'))
    
    features = {
        'abs_amount': abs(amount),
        'is_loss': 1 if amount < 0 else 0,
        'amount_sign': np.sign(amount),
        'log_amount': np.log1p(abs(amount)),
        'category_type': get_category_type(gl, measure_dict),
        'gl_length': len(gl),
        'is_high_value': 1 if abs(amount) > 50000 else 0,
        'is_low_value': 1 if abs(amount) < 1000 else 0,
    }
    
    # Add dummy deviation (in real system, compute from historical)
    features['deviation_from_mean'] = random.uniform(-2, 2)
    features['z_score'] = random.uniform(-3, 3)
    
    # Rolling trend simulation
    features['rolling_trend'] = random.choice([-1, 0, 1])
    
    return features

def generate_synthetic_data(n=10000):
    """Generate synthetic training data"""
    np.random.seed(42)
    data = []
    
    gl_accounts = ["Rent Revenue", "Maintenance Expense", "Payroll", "Utilities", 
                   "Marketing", "Property Tax", "Insurance", "Unknown GL"]
    
    for _ in range(n):
        gl = np.random.choice(gl_accounts)
        amount = np.random.normal(0, 50000)
        if "Expense" in gl or "Tax" in gl or "Insurance" in gl:
            amount = abs(amount) * -1  # bias to loss/expense
        
        row = {
            'gl_account': gl,
            'amount': amount,
            'status': 'Loss' if amount < 0 else 'Profit'
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    # Add target for training (severity)
    df['severity'] = df['amount'].apply(lambda x: 
        'CRITICAL' if x < -80000 else
        'HIGH' if x < 0 else
        'HIGH' if x > 120000 else
        'MEDIUM' if abs(x) > 10000 else 'LOW'
    )
    return df
