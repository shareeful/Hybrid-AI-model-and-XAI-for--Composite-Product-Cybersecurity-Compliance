# config.py
import os

# --- Data Settings ---
DATA_PATH = 'vulnerabilities.csv'
C_TOE_ASSETS = [
    'Microsoft Windows Server 2019',
    'Oracle Database 19c Enterprise',
    'Cisco ASA 5525-X',
    'Red Hat Enterprise Linux 8.2',
    'IBM Openslice OSS'
]

# --- OpenAI API Settings ---
# Requires environment variable or direct insertion
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
GPT_MODEL_BASE = "gpt-3.5-turbo-0125"

# --- Risk Management Parameters (Methodology Step 4.2) ---
# The "Safety Limit" defined in the Security Target (Upper bound of Medium Risk)
SAFETY_LIMIT_SCORE = 0.69 

# Default Z-Score parameter for Critical Drivers (Mean + 1 Std Dev)
Z_SCORE_THRESHOLD = 1.0 

# --- Ensemble Hyperparameters ---
RF_PARAMS = {'n_estimators': 500, 'max_depth': 25, 'min_samples_split': 5, 'random_state': 42}
GB_PARAMS = {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 8, 'random_state': 42}
ELASTICNET_PARAMS = {'alpha': 0.001, 'l1_ratio': 0.5, 'max_iter': 2000, 'random_state': 42}
