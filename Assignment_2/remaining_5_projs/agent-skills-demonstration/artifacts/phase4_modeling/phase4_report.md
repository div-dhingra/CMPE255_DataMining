# CRISP-DM Phase 4: Modeling Report

## Model Leaderboard (Stratified 5-Fold Cross-Validation)
| Rank | Model Pipeline | CV ROC-AUC | PR-AUC | F1 Score | Overfitting Delta |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **LogisticRegression** | 0.8504 ± 0.0122 | 0.6730 | 0.6299 | 0.0045 |
| 2 | **Tuned_RandomForest** | 0.8490 ± 0.0050 | 0.6500 | 0.6200 | 0.0400 |
| 3 | **RandomForest** | 0.8486 ± 0.0084 | 0.6650 | 0.6340 | 0.0583 |
| 4 | **HistGradientBoosting** | 0.8444 ± 0.0102 | 0.6601 | 0.5784 | 0.0741 |

## Champion Model
- **Selected Model**: `LogisticRegression`
- **CV ROC-AUC**: 0.8504
- **Pipeline Architecture**: Scikit-Learn Pipeline combining `StandardScaler` on numerical features, `OneHotEncoder(handle_unknown='ignore')` on categorical attributes, and cost-sensitive balanced classification.

## Hyperparameter Tuning Summary
- **Search Space**: Tested tree depths 6, 8, 10 and estimators 100, 150, 200 with leaf regularization.
- **Optimal Hyperparameters**: {'name': 'RF_depth_10_n200', 'max_depth': 10, 'min_samples_leaf': 15, 'n_estimators': 200}
