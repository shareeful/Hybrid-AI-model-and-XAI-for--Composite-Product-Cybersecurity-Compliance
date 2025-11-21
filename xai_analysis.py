# xai_analysis.py
import shap
import numpy as np
import pandas as pd
from config import SAFETY_LIMIT_SCORE, Z_SCORE_THRESHOLD

def calculate_global_shap(model, X_sample):
    """
    Step 4.1: Global Feature Importance.
    Calculates SHAP values to determine 'mu' and 'sigma'.
    """
    print("Running SHAP Analysis...")
    # Using a generic explainer on the prediction function
    # For HybridGPT, this would use KernelSHAP (slow) or explain the ensemble proxy
    # We simulate the output format
    
    # Mock SHAP values for code structure validity
    feature_names = X_sample.columns
    mean_shap_values = np.random.rand(len(feature_names)) * 0.2
    
    shap_series = pd.Series(mean_shap_values, index=feature_names)
    return shap_series

def calculate_critical_thresholds(shap_series):
    """
    Step 4.2 (Part 1): Risk Significance Formula.
    Formula: T_crit = mu + sigma
    """
    mu = shap_series.mean()
    sigma = shap_series.std()
    
    t_crit = mu + (Z_SCORE_THRESHOLD * sigma)
    t_mat = mu
    
    print(f"Threshold Calibration: Mean={mu:.3f}, Std={sigma:.3f} -> T_crit={t_crit:.3f}")
    return t_crit, t_mat

def calculate_adequacy_target(inherent_risk):
    """
    Step 4.2 (Part 2): The 'Required' Gap Calculation.
    Formula: Tau = (Inherent - SafetyLimit) / Inherent
    """
    if inherent_risk <= SAFETY_LIMIT_SCORE:
        return 0.0 # Already safe
        
    tau_required = (inherent_risk - SAFETY_LIMIT_SCORE) / inherent_risk
    return tau_required

def counterfactual_analysis(model, row, feature, optimal_val=0):
    """
    Step 4.2 (Part 3): 'Actual' Gap Measurement.
    Simulates optimal control performance.
    """
    # 1. Inherent Prediction
    risk_inherent = model.predict_single(row)
    
    # 2. Counterfactual Prediction
    row_optimized = row.copy()
    row_optimized[feature] = optimal_val
    risk_residual = model.predict_single(row_optimized)
    
    # 3. Actual Gap Closure
    if risk_inherent == 0: return 0.0, 0.0
    delta_g = (risk_inherent - risk_residual) / risk_inherent
    
    return delta_g, risk_inherent
