# ensemble_selection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from config import RF_PARAMS, GB_PARAMS, ELASTICNET_PARAMS

def select_key_features(X, y, top_n=25):
    """
    Executes ensemble feature selection: 0.4 RF + 0.4 GB + 0.2 EN.
    """
    print("Starting Ensemble Feature Selection...")
    
    # 1. Random Forest
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)
    
    # 2. Gradient Boosting
    gb = GradientBoostingRegressor(**GB_PARAMS)
    gb.fit(X, y)
    gb_imp = pd.Series(gb.feature_importances_, index=X.columns)
    
    # 3. ElasticNet
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    en = ElasticNet(**ELASTICNET_PARAMS)
    en.fit(X_scaled, y)
    en_imp = pd.Series(np.abs(en.coef_), index=X.columns)
    
    # Weighted Combination (Formula from Paper)
    combined_imp = (0.4 * rf_imp) + (0.4 * gb_imp) + (0.2 * en_imp)
    
    top_features = combined_imp.nlargest(top_n).index.tolist()
    
    print(f"Selected Top {top_n} Features: {top_features}")
    return top_features, combined_imp
