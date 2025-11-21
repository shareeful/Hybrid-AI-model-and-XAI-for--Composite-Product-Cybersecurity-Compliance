# config.py

DATA_PATH = 'vulnerabilities.csv'  # Path to your CVEJoin dataset
C_TOE_ASSETS = [
    'Microsoft Windows Server 2019',
    'Oracle Database 19c Enterprise',
    'Cisco ASA 5525-X',
    'Red Hat Enterprise Linux 8.2',
    'IBM Openslice OSS'
]

# Feature selection parameters
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 25,
    'min_samples_split': 5,
    'random_state': 42
}

GB_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 8,
    'random_state': 42
}

ELASTICNET_PARAMS = {
    'alpha': 0.001,
    'l1_ratio': 0.5,
    'max_iter': 2000,
    'random_state': 42
}

# Thresholds for Risk Classification (calculated dynamically in code, but defaults here)
RISK_THRESHOLD_CRITICAL = 0.9
RISK_THRESHOLD_HIGH = 0.7
RISK_THRESHOLD_MEDIUM = 0.4

# Control Adequacy Thresholds
ADEQUATE_THRESHOLD = 0.15 # 15% Gap Closure
MODERATE_THRESHOLD = 0.10 # 10% Gap Closure
