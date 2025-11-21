# Hybrid AI Framework for Composite Product Cybersecurity Certification

## Overview
This repository contains the source code implementation for the research paper: **"Hybrid AI-Based Dynamic Risk Assessment Framework with Explainable AI Practices for Composite Product Cybersecurity Certification."**

The framework integrates **Ensemble Learning** for robust feature selection, **Large Language Models (GPT-3.5)** for contextual vulnerability prediction, and **Explainable AI (XAI)** to automate the generation of audit evidence for European Cybersecurity Certification (EUCC) schemes.

### Key Features
* **Strict Data Hygiene:** Implements rigorous preprocessing to isolate the target variable (`EPSS`) and prevent data leakage.
* **Hybrid Architecture:** Combines Random Forest, Gradient Boosting, and ElasticNet with GPT-3.5 fine-tuning.
* **Counterfactual Sensitivity Analysis:** Validates security controls by measuring the "Gap Closure" ($\Delta G$) between inherent risk and optimized safety targets.
* **Statistical Risk Calibration:** Dynamically calculates risk thresholds using Z-score principles ($T_{crit} = \mu + \sigma$).
* **Automated Evidence Generation:** Produces deterministic, template-based audit trails compatible with **AVA_VAN** assurance requirements.

---

## Project Structure

The project is modularized into seven core components:

| File | Description |
| :--- | :--- |
| `main.py` | The orchestration script. Runs the full pipeline from data loading to report generation. |
| `config.py` | Central configuration for file paths, API keys, hyperparameters, and risk thresholds. |
| `preprocessing.py` | Handles data loading, cleaning, and feature engineering. **Includes logic to prevent target leakage.** |
| `ensemble_selection.py` | Implements the weighted voting mechanism (RF + GB + ElasticNet) to select key risk drivers. |
| `hybrid_model.py` | Manages the OpenAI API integration for Fine-Tuning and Prediction. |
| `xai_analysis.py` | The analytical core. Calculates SHAP values, Z-score thresholds, and Counterfactual Sensitivity. |
| `evidence_generation.py` | Contains the string-based logic templates to generate reproducible audit text. |

---

## Installation

### Prerequisites
* Python 3.8+
* An OpenAI API Key (for GPT-3.5 functionality)

### Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-repo/hybrid-certification-framework.git](https://github.com/your-repo/hybrid-certification-framework.git)
   cd hybrid-certification-framework
