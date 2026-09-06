"""CRISP-DM Phase 3: Data Preparation.

Orchestrates:
- data-cleaning (param087/agent-ml-skills)
- feature-engineering (param087/agent-ml-skills)
- imbalanced-data (param087/agent-ml-skills)
- context-packager (nimrodfisher/data-analytics-skills)
"""

import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.config import ARTIFACTS_DIR, DATASET_PATH
from src.modules.eda_profiler import load_raw_dataset
from src.modules.preprocessor import prepare_data_splits


def run_phase3_data_preparation(
    dataset_path: Path = DATASET_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR / "phase3_data_preparation",
) -> Dict[str, Any]:
    """Execute all Phase 3 Data Preparation skills and generate artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset(dataset_path)

    # Execute full preparation pipeline
    splits = prepare_data_splits(df)

    cleaning_meta = splits["cleaning_metadata"]
    imbalance_meta = splits["imbalance_metadata"]
    context_pkg = splits["context_package"]

    phase3_summary = {
        "crisp_dm_phase": "Phase 3: Data Preparation",
        "status": "COMPLETED",
        "skills_executed": [
            "data-cleaning",
            "feature-engineering",
            "imbalanced-data",
            "context-packager",
        ],
        "training_samples": len(splits["X_train"]),
        "testing_samples": len(splits["X_test"]),
        "engineered_feature_count": len(splits["feature_cols"]),
        "engineered_features": splits["feature_cols"],
        "cleaning_metadata": cleaning_meta,
        "imbalance_handling": imbalance_meta,
        "context_package": context_pkg,
    }

    # Persist JSON artifact
    json_path = artifacts_dir / "phase3_data_preparation.json"
    with open(json_path, "w") as f:
        json.dump(phase3_summary, f, indent=2)

    # Persist Markdown report
    md_path = artifacts_dir / "phase3_report.md"
    with open(md_path, "w") as f:
        f.write(f"""# CRISP-DM Phase 3: Data Preparation Report

## Leakage-Safe Cleaning
- **Imputation Strategy**: `{cleaning_meta['imputation_strategy']}`
- **Imputed Value**: `${cleaning_meta['imputed_value']}` (learned strictly from training split)
- **Train Imputed Missing**: {cleaning_meta['train_imputed_missing_count']} records
- **Test Imputed Missing**: {cleaning_meta['test_imputed_missing_count']} records
- **Target Encoding**: `No` -> 0, `Yes` -> 1

## Feature Engineering
- **Total Features Prepared**: {len(splits['feature_cols'])}
- **Domain Features Added**:
  - `TenureCohort`: Binned customer lifecycle stages (`0-12m`, `13-24m`, `25-48m`, `49-72m`)
  - `ServiceBundleCount`: Aggregated security & streaming add-ons count (0 to 6)
  - `HasFiberOptic`: InternetService indicator (high churn risk factor)
  - `IsMonthToMonth`: Short-term contract indicator
  - `IsElectronicCheck`: Manual payment indicator
  - `MonthlyToTotalRatio`: Ratio of monthly bill to cumulative lifetime billing
  - `HighSpendRisk`: Interaction flag (Month-to-month and MonthlyCharges > $70)

## Imbalanced Data Management
- **Original Imbalance**: {imbalance_meta['imbalance_ratio']} (Retained: {imbalance_meta['original_distribution']['retained_0']:,}, Churned: {imbalance_meta['original_distribution']['churned_1']:,})
- **Class Weights**: {imbalance_meta['balanced_class_weights']}
- **SMOTE Oversampling**: Synthesized {imbalance_meta['synthetic_samples_generated']:,} minority samples for training.
""")

    # Attach the actual prepared DataFrames for downstream phase usage
    phase3_summary["_splits"] = splits
    return phase3_summary


if __name__ == "__main__":
    res = run_phase3_data_preparation()
    print("Phase 3 completed! Engineered features:", res["engineered_feature_count"])
