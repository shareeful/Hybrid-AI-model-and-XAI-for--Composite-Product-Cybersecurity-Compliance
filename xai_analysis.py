# xai_analysis.py
import shap
import numpy as np
import pandas as pd
from config import SAFETY_LIMIT_SCORE, Z_SCORE_THRESHOLD

def calculate_global_shap(model, X_sample):
    """
    Step 4.1: Influential Features Extraction via Real SHAP Analysis.
    
    This function wraps the Hybrid Model to compute actual Shapley values.
    It connects the model's prediction logic (Input -> Prompt -> Score) 
    to the SHAP explainer to measure the marginal contribution of each feature.
    """
    print(f"Initializing SHAP Explainer on {len(X_sample)} samples...")

    # 1. Define a wrapper function that bridges SHAP (Matrix) to HybridModel (Row-based Prompting)
    # SHAP passes a numpy array or DataFrame; we need to return an array of predictions.
    def prediction_wrapper(X_matrix):
        # Handle cases where SHAP passes numpy array instead of DataFrame
        if isinstance(X_matrix, np.ndarray):
            X_df = pd.DataFrame(X_matrix, columns=X_sample.columns)
        else:
            X_df = X_matrix
            
        scores = []
        for _, row in X_df.iterrows():
            # Call the ACTUAL model prediction (OpenAI API or Regressor)
            try:
                pred = model.predict_single(row)
                scores.append(pred)
            except Exception as e:
                print(f"Prediction error in SHAP wrapper: {e}")
                scores.append(0.0) # Fallback
        return np.array(scores)

    # 2. Initialize the Explainer
    # We use PermutationExplainer or KernelExplainer as they are model-agnostic
    # (Necessary because our 'model' is a black-box GPT wrapper, not a standard sklearn tree)
    explainer = shap.Explainer(prediction_wrapper, X_sample)
    
    # 3. Calculate SHAP Values
    # This performs the permutations to calculate marginal contributions
    print("Calculating SHAP values (this may take time for API-based models)...")
    shap_values = explainer(X_sample)
    
    # 4. Compute Global Importance (Mean Absolute SHAP Value)
    # |SHAP| represents the magnitude of impact on risk
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    
    shap_series = pd.Series(feature_importance, index=X_sample.columns).sort_values(ascending=False)
    
    print("SHAP Analysis Complete. Top features identified.")
    return shap_series

def calculate_critical_thresholds(shap_series):
    """
    Step 4.2 (Part 1): Risk Significance Classification (The Formula).
    
    Formula: T_crit = mu + sigma
    Calculates dynamic thresholds based on the statistical distribution of 
    the actual SHAP values derived above.
    """
    mu = shap_series.mean()
    sigma = shap_series.std()
    
    # The Critical Threshold (Z-Score > 1.0)
    t_crit = mu + (Z_SCORE_THRESHOLD * sigma)
    
    # The Material Threshold (Mean)
    t_mat = mu
    
    print(f"\n[Statistical Calibration]")
    print(f"Mean Influence (mu): {mu:.4f}")
    print(f"Standard Dev (sigma): {sigma:.4f}")
    print(f"CRITICAL THRESHOLD (mu + sigma): {t_crit:.4f}")
    
    return t_crit, t_mat

def calculate_adequacy_target(inherent_risk):
    """
    Step 4.2 (Part 2): The 'Required' Gap Calculation.
    
    Formula: Tau = (Inherent - SafetyLimit) / Inherent
    Determines the 'Passing Grade' for a control.
    """
    # If the asset is already safe (risk <= limit), no reduction is strictly required
    if inherent_risk <= SAFETY_LIMIT_SCORE:
        return 0.0 
        
    tau_required = (inherent_risk - SAFETY_LIMIT_SCORE) / inherent_risk
    return tau_required

def counterfactual_analysis(model, row, feature, optimal_val=0):
    """
    Step 4.2 (Part 3): 'Actual' Gap Measurement via Counterfactual Sensitivity.
    
    Simulates the "Optimal State" to measure control efficacy.
    """
    # 1. Measure Inherent Risk (Current State)
    risk_inherent = model.predict_single(row)
    
    # 2. Simulate Counterfactual (Control is Optimized)
    row_optimized = row.copy()
    # We force the risk driver feature to its 'safe' value (e.g., 0 for exploit)
    row_optimized[feature] = optimal_val 
    
    risk_residual = model.predict_single(row_optimized)
    
    # 3. Calculate Actual Gap Closure (Delta G)
    if risk_inherent <= 0: 
        delta_g = 0.0
    else:
        delta_g = (risk_inherent - risk_residual) / risk_inherent
    
    # Sanity check: Gap cannot be negative (unless control increases risk)
    delta_g = max(delta_g, 0.0)
    
    return delta_g, risk_inherent
