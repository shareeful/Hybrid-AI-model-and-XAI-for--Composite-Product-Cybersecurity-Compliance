# evidence_generation.py

def generate_audit_report(asset_name, feature, shap_val, t_crit, control_name, 
                          gap_actual, gap_required, risk_score):
    """
    Step 4.3: Structured Template Generation.
    """
    
    # 1. Classify Risk Driver
    if shap_val >= t_crit:
        driver_class = "Critical Risk Driver"
    else:
        driver_class = "Contributing Factor"
        
    # 2. Classify Adequacy (The Comparison Check)
    # Logic: Actual >= Required -> Adequate
    if gap_actual >= gap_required:
        status = "Adequate"
        impact_text = "effectively transitions the asset to the safety target"
    elif gap_actual >= 0.10: # Minimum tolerance
        status = "Moderate"
        impact_text = "provides mitigation but fails to fully bridge the safety gap"
    else:
        status = "Inadequate"
        impact_text = "fails to impact the risk posture materially"

    # 3. Instantiate Template
    template = (
        f"AUDIT EVIDENCE FOR ASSET: [{asset_name}]\n"
        f"1. Risk Analysis: The risk score ({risk_score:.3f}) is driven by [{feature}] "
        f"(Importance: {shap_val:.3f}), classified as a **{driver_class}**.\n"
        f"2. Requirement: To reach the safety limit (0.69), a reduction of **{gap_required*100:.1f}%** is required.\n"
        f"3. Performance: Control [{control_name}] demonstrates an actual gap closure of **{gap_actual*100:.1f}%**.\n"
        f"4. Conclusion: Since {gap_actual*100:.1f}% {'=>' if status=='Adequate' else '<'} {gap_required*100:.1f}%, "
        f"the control is rated as **{status}**. It {impact_text}."
    )
    
    return template
