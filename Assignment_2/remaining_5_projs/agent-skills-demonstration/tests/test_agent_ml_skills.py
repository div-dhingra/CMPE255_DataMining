"""Test suite for the 15 skills from param087/agent-ml-skills."""

from src.skills_registry import execute_skill_by_id, get_skill_details


def test_solution_design():
    res = execute_skill_by_id("solution-design")
    assert res["status"] == "SUCCESS"
    assert "business_objective" in res["output"]
    assert "PR-AUC" in res["output"]["primary_metric"]


def test_exploratory_data_analysis():
    res = execute_skill_by_id("exploratory-data-analysis")
    assert res["status"] == "SUCCESS"
    assert res["output"]["dataset_shape"]["rows"] == 7043
    assert "churn_rate_pct" in res["output"]["target_distribution"]


def test_data_cleaning():
    res = execute_skill_by_id("data-cleaning")
    assert res["status"] == "SUCCESS"
    assert res["output"]["imputation_strategy"] == "train_set_median"
    assert "GUARANTEED" in res["output"]["leakage_safety"]


def test_feature_engineering():
    res = execute_skill_by_id("feature-engineering")
    assert res["status"] == "SUCCESS"
    assert res["output"]["count"] >= 20
    assert "TenureCohort" in res["output"]["features"]


def test_pandas_patterns():
    res = execute_skill_by_id("pandas-patterns")
    assert res["status"] == "SUCCESS"
    assert res["output"]["memory_savings_pct"] > 0
    assert len(res["output"]["vectorized_transforms_applied"]) >= 4


def test_imbalanced_data():
    res = execute_skill_by_id("imbalanced-data")
    assert res["status"] == "SUCCESS"
    assert "balanced_class_weights" in res["output"]
    assert res["output"]["smote_resampled_distribution"]["churned_1"] > 0


def test_sklearn_pipelines():
    res = execute_skill_by_id("sklearn-pipelines")
    assert res["status"] == "SUCCESS"
    assert "LogisticRegression" in res["output"]["pipelines_built"]
    assert "RandomForest" in res["output"]["pipelines_built"]


def test_model_training():
    res = execute_skill_by_id("model-training")
    assert res["status"] == "SUCCESS"
    assert res["output"]["status"] == "FITTED_AND_EVALUATED"


def test_hyperparameter_tuning():
    res = execute_skill_by_id("hyperparameter-tuning")
    assert res["status"] == "SUCCESS"
    assert res["output"]["achieved_cv_roc_auc"] > 0.80


def test_model_evaluation():
    res = execute_skill_by_id("model-evaluation")
    assert res["status"] == "SUCCESS"
    assert res["output"]["roc_auc"] > 0.80
    assert res["output"]["pr_auc"] > 0.50
    assert "confusion_matrix" in res["output"]


def test_cross_validation():
    res = execute_skill_by_id("cross-validation")
    assert res["status"] == "SUCCESS"
    assert res["output"]["n_splits"] == 5
    assert len(res["output"]["fold_scores"]) == 5


def test_ml_monitoring_observability():
    res = execute_skill_by_id("ml-monitoring-observability")
    assert res["status"] == "SUCCESS"
    assert "prediction_drift" in res["output"]
    assert "latency_metrics" in res["output"]


def test_lora_finetuning():
    res = execute_skill_by_id("lora-finetuning")
    assert res["status"] == "SUCCESS"
    assert res["output"]["lora_hyperparameters"]["r"] == 16
    assert res["output"]["trainable_parameters_percentage"] < 1.0


def test_ragas_evaluation():
    res = execute_skill_by_id("ragas-evaluation")
    assert res["status"] == "SUCCESS"
    assert res["output"]["ragas_composite_score"] > 0.80
    assert res["output"]["metrics_summary"]["faithfulness"] >= 0.90


def test_llm_as_judge():
    res = execute_skill_by_id("llm-as-judge")
    assert res["status"] == "SUCCESS"
    assert res["output"]["total_messages_evaluated"] >= 2
    assert "approval_rate_pct" in res["output"]
