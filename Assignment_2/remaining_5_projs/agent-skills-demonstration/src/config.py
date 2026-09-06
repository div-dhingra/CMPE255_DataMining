"""Centralized Configuration for Agent Skills Demonstration (Project 5).

Structured according to the 6 CRISP-DM phases on the IBM Telco Customer Churn dataset.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SKILLS_DIR = BASE_DIR / "skills"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MPL_CACHE_DIR = BASE_DIR.parent / ".mplcache"

# Set matplotlib cache dir immediately to avoid font rebuilding
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

# Ensure directories exist
for directory in [DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASET_PATH = DATA_DIR / "Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

# Feature Categorization
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Engineered Features to generate in Phase 3
ENGINEERED_FEATURES = [
    "TenureCohort",
    "ServiceBundleCount",
    "HasFiberOptic",
    "IsMonthToMonth",
    "IsElectronicCheck",
    "MonthlyToTotalRatio",
    "HighSpendRisk",
]

# Modeling & Evaluation Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Drift & Observability Thresholds
KS_PVALUE_THRESHOLD = 0.05  # Below this p-value indicates significant distribution shift
PSI_THRESHOLD_WARN = 0.10   # Slight shift
PSI_THRESHOLD_CRIT = 0.25   # Significant shift

# Business & Financial Economics Assumptions (Phase 1 & Phase 6)
AVERAGE_ARPU = 64.76                  # Average Monthly Revenue Per User ($)
ANNUAL_ARPU = AVERAGE_ARPU * 12       # $777.12 / year
RETENTION_CAMPAIGN_COST = 90.00       # $15/month credit for 6 months per targeted user
INTERVENTION_EFFECTIVENESS = 0.45     # 45% of targeted high-risk churners are retained
TOTAL_CUSTOMER_BASE = 7043
ANNUAL_MRR_AT_STAKE = 1869 * ANNUAL_ARPU  # ~1,869 churners * $777 = $1.45M
