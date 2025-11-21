# preprocessing.py
import pandas as pd
import numpy as np
from config import C_TOE_ASSETS

def load_and_filter_data(filepath):
    """Loads dataset and filters for C-ToE assets."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter for specific assets based on 'product' or 'vendor' columns
    # This simulates the C-ToE definition phase [cite: 337]
    mask = df['product'].apply(lambda x: any(asset in str(x) for asset in C_TOE_ASSETS))
    c_toe_data = df[mask].copy()
    
    print(f"Filtered {len(c_toe_data)} records relevant to C-ToE assets.")
    return c_toe_data

def preprocess_features(df):
    """
    Cleans data and engineers features. 
    Crucially removes 'epss' from predictors to avoid target leakage.
    """
    data = df.copy()
    
    # Target variable
    y = data['epss']
    
    # Drop target and identifiers from inputs
    drop_cols = ['epss', 'cve_id', 'description'] # Ensure EPSS is dropped 
    X_raw = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

    # Feature Engineering [cite: 347]
    if 'cve_published_date' in X_raw.columns:
        X_raw['cve_published_date'] = pd.to_datetime(X_raw['cve_published_date'])
        X_raw['days_since_published'] = (pd.Timestamp.now() - X_raw['cve_published_date']).dt.days
        X_raw.drop(columns=['cve_published_date'], inplace=True)

    if 'exploit_count' in X_raw.columns:
        X_raw['exploit_count_log'] = np.log1p(X_raw['exploit_count'].fillna(0))
        X_raw['has_public_exploit'] = (X_raw['exploit_count'] > 0).astype(int)

    # Simple encoding for categorical variables (simplified for this snippet)
    X = pd.get_dummies(X_raw, drop_first=True)
    
    # Fill missing values
    X = X.fillna(0)
    
    return X, y
