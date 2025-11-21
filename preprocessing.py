# preprocessing.py
import pandas as pd
import numpy as np
from config import C_TOE_ASSETS

def load_and_filter_data(filepath):
    """
    Loads CVEJoin dataset and filters for C-ToE assets (Phase 1).
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # For testing without the csv
        print("Warning: CSV not found. Generating dummy data for demonstration.")
        return _generate_dummy_data()

    # Filter logic: Keep rows where 'product' matches C-ToE list
    mask = df['product'].fillna('').apply(lambda x: any(asset.lower() in str(x).lower() for asset in C_TOE_ASSETS))
    c_toe_data = df[mask].copy()
    
    print(f"Filtered {len(c_toe_data)} records relevant to C-ToE assets.")
    return c_toe_data

def preprocess_features(df):
    """
    Engineers features and separates Target (y) from Inputs (X).
    """
    data = df.copy()
    
    # --- 1. TARGET ISOLATION ---
    if 'epss' not in data.columns:
        raise ValueError("Dataset missing 'epss' target column")
    y = data['epss']
    
    # --- 2. FEATURE EXTRACTION ---
    X = pd.DataFrame()

    # CVSS Core Metrics
    X['base_score'] = data.get('base_score', 0).fillna(0)
    X['exploitability_score'] = data.get('exploitability_score', 0).fillna(0)
    X['impact_score'] = data.get('impact_score', 0).fillna(0)
    
    # Temporal Features
    if 'cve_published_date' in data.columns:
        pub_date = pd.to_datetime(data['cve_published_date'], errors='coerce').fillna(pd.Timestamp.now())
        X['days_since_published'] = (pd.Timestamp.now() - pub_date).dt.days
    else:
        X['days_since_published'] = 0

    # Exploit Intelligence
    if 'exploit_count' in data.columns:
        X['exploit_count_log'] = np.log1p(data['exploit_count'].fillna(0))
        X['has_public_exploit'] = (data['exploit_count'] > 0).astype(int)
    else:
        X['exploit_count_log'] = 0.0
        X['has_public_exploit'] = 0

    # Vendor Indicators (Pattern Matching)
    vendor_col = data['vendor'].fillna('').str.lower() if 'vendor' in data.columns else pd.Series(['']*len(data))
    X['vendor_microsoft'] = vendor_col.str.contains('microsoft').astype(int)
    X['vendor_oracle'] = vendor_col.str.contains('oracle').astype(int)
    X['vendor_cisco'] = vendor_col.str.contains('cisco').astype(int)
    
    # Attack Vector Encoding
    if 'attack_vector' in data.columns:
        vector_map = {'NETWORK': 3, 'ADJACENT_NETWORK': 2, 'LOCAL': 1, 'PHYSICAL': 0}
        X['attack_vector'] = data['attack_vector'].str.upper().map(vector_map).fillna(0)
    
    return X, y

def _generate_dummy_data():
    """Helper to create dummy data if CSV is missing during review."""
    data = {
        'product': ['Microsoft Windows Server 2019'] * 10,
        'epss': np.random.uniform(0.1, 0.99, 10),
        'base_score': np.random.uniform(5, 10, 10),
        'exploitability_score': np.random.uniform(1, 10, 10),
        'impact_score': np.random.uniform(1, 10, 10),
        'exploit_count': [5, 0, 2, 10, 0, 1, 5, 20, 0, 1],
        'vendor': ['Microsoft'] * 10,
        'attack_vector': ['NETWORK'] * 10,
        'cve_published_date': ['2020-01-01'] * 10
    }
    return pd.DataFrame(data)
