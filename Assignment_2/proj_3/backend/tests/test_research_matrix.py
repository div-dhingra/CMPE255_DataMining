"""
Unit and Integration Tests for Research Literature Alignment & Benchmark Synthesis Matrix.
Tests:
- Literature citations repository, topic filtering, BibTeX export
- Algorithm trade-off taxonomy and complexity profiles
- Metric formulations and directionalities
- Cross-paradigm benchmark execution across all 6 models
- Bootstrap stability and standard deviation calculation
- Baseline vs. Hill-climbed ablation matrix generation
- LaTeX and Markdown table exporters
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import json

# Ensure backend/src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from crisp_dm.data_understanding import generate_synthetic_credit_card_data
from crisp_dm.data_preparation import PipelineConfig
from research.literature import (
    LITERATURE_REPOSITORY,
    ALGORITHM_TAXONOMY,
    METRIC_TAXONOMY,
    AUTORESEARCH_PRINCIPLES,
    get_literature_repository,
    get_citations,
    get_citation_by_id,
    get_bibtex,
    get_algorithm_taxonomy,
    get_metric_formulations,
    get_autoresearch_principles,
)
from research.benchmark_matrix import (
    BenchmarkRunner,
    BenchmarkModelResult,
    BenchmarkMatrixSummary,
    AblationMatrixSummary,
    compute_composite_fitness,
    run_benchmark_matrix,
    generate_ablation_matrix,
    to_latex_table,
    to_markdown_table,
    to_latex_ablation_table,
    to_markdown_ablation_table,
    to_dict,
    to_json,
)


# ============================================================================
# 1. Literature Repository Tests
# ============================================================================

def test_literature_repository_completeness():
    """Verify presence of all seminal research papers."""
    repo = get_literature_repository()
    assert len(repo) >= 9

    authors = [c["author"] for c in repo]
    assert "Rousseeuw" in authors
    assert "Davies & Bouldin" in authors
    assert "Calinski & Harabasz" in authors
    assert "Arthur & Vassilvitskii" in authors
    assert "Campello et al." in authors
    assert "McInnes et al." in authors
    assert "Von Luxburg" in authors
    assert "Satopaa et al." in authors
    assert "Sakana AI" in authors


def test_literature_topic_filtering():
    """Verify topic-based query filtering."""
    val_citations = get_citations(topic="validation")
    assert len(val_citations) >= 3
    authors = [c["author"] for c in val_citations]
    assert "Rousseeuw" in authors
    assert "Davies & Bouldin" in authors

    density_citations = get_citations(topic="density")
    assert any("Campello" in c["author"] for c in density_citations)

    auto_citations = get_citations(topic="autoresearch")
    assert any("Sakana" in c["author"] for c in auto_citations)


def test_citation_by_id_retrieval():
    """Verify direct citation retrieval by ID."""
    c = get_citation_by_id("rousseeuw1987")
    assert c is not None
    assert c["year"] == 1987
    assert "Silhouettes" in c["title"]
    assert c["venue"] == "Journal of Computational and Applied Mathematics"

    non_existent = get_citation_by_id("unknown_paper_999")
    assert non_existent is None


def test_bibtex_generation():
    """Verify single and full BibTeX export formatting."""
    single_bib = get_bibtex("rousseeuw1987")
    assert "@article{rousseeuw1987silhouettes" in single_bib
    assert "author={Rousseeuw, Peter J}" in single_bib

    full_bib = get_bibtex()
    assert "@article" in full_bib
    assert "@inproceedings" in full_bib
    assert "calinski1974dendrite" in full_bib
    assert "davies1979cluster" in full_bib
    assert "mcinnes2018umap" in full_bib


def test_algorithm_taxonomy_tradeoffs():
    """Verify computational complexity and inductive biases for all 6 models."""
    taxonomy = get_algorithm_taxonomy()
    assert len(taxonomy) == 6

    expected_models = ["kmeans", "kmedoids", "dbscan", "hdbscan", "agglomerative", "gmm"]
    for m in expected_models:
        assert m in taxonomy
        entry = taxonomy[m]
        assert "time_complexity" in entry
        assert "space_complexity" in entry
        assert "assumption" in entry
        assert "inductive_bias" in entry
        assert len(entry["pros"]) > 0
        assert len(entry["cons"]) > 0

    assert "O(k*n*d)" in taxonomy["kmeans"]["time_complexity"]
    assert taxonomy["kmeans"]["paradigm"] == "Partitioning"
    assert taxonomy["dbscan"]["paradigm"] == "Density-Based"
    assert taxonomy["agglomerative"]["paradigm"] == "Hierarchical"
    assert taxonomy["gmm"]["paradigm"] == "Probabilistic"


def test_metric_taxonomy_formulations():
    """Verify directionality and mathematical properties of metrics."""
    taxonomy = get_metric_formulations()
    assert "silhouette" in taxonomy
    assert "davies_bouldin" in taxonomy
    assert "calinski_harabasz" in taxonomy
    assert "cluster_stability" in taxonomy
    assert "inertia_elbow" in taxonomy
    assert "hopkins_statistic" in taxonomy

    assert taxonomy["silhouette"]["direction"] == "maximize"
    assert taxonomy["silhouette"]["bounds"] == [-1.0, 1.0]

    assert taxonomy["davies_bouldin"]["direction"] == "minimize"
    assert taxonomy["davies_bouldin"]["bounds"][0] == 0.0

    assert taxonomy["calinski_harabasz"]["direction"] == "maximize"
    assert taxonomy["cluster_stability"]["direction"] == "maximize"


def test_autoresearch_principles():
    """Verify core autoresearch methodology principles."""
    principles = get_autoresearch_principles()
    assert len(principles) == 6
    p_ids = [p["id"] for p in principles]
    assert "discrete_neighborhood_mutation" in p_ids
    assert "continuous_parameter_perturbation" in p_ids
    assert "composite_objective_evaluation" in p_ids
    assert "simulated_annealing_acceptance" in p_ids
    assert "random_restarts_on_plateau" in p_ids
    assert "ablation_isolation" in p_ids


# ============================================================================
# 2. Benchmark Matrix Execution Tests
# ============================================================================

@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Fixture providing synthetic customer dataframe."""
    return generate_synthetic_credit_card_data(n_samples=120, random_state=42)


def test_composite_fitness_bounds():
    """Verify composite fitness is strictly bounded in [0, 1]."""
    f1 = compute_composite_fitness(silhouette=0.5, davies_bouldin=0.8, calinski_harabasz=1500.0, stability=0.9, noise_ratio=0.0)
    assert 0.0 <= f1 <= 1.0

    # Negative silhouette edge case
    f_neg = compute_composite_fitness(silhouette=-0.5, davies_bouldin=4.0, calinski_harabasz=10.0, stability=0.2, noise_ratio=0.5)
    assert 0.0 <= f_neg <= 1.0
    assert f_neg < f1


def test_benchmark_matrix_all_6_models(sample_dataset: pd.DataFrame):
    """Verify benchmark matrix executes all 6 models across 4 paradigms on real data."""
    summary = run_benchmark_matrix(
        data=sample_dataset,
        pipeline_config=PipelineConfig(scaling_strategy="yeo_johnson", engineer_ratios=True),
        n_bootstraps=2,
        random_state=42,
        dataset_name="Test Synthetic CC",
    )

    assert isinstance(summary, BenchmarkMatrixSummary)
    assert summary.n_samples == 120
    assert summary.n_features > 0
    assert len(summary.results) == 6

    model_names = [r.model_name for r in summary.results]
    assert "K-Means" in model_names
    assert "K-Medoids" in model_names
    assert "DBSCAN" in model_names
    assert "HDBSCAN" in model_names
    assert "Agglomerative" in model_names
    assert "GMM" in model_names

    paradigms = set(r.paradigm for r in summary.results)
    assert len(paradigms) == 4
    assert "Partitioning" in paradigms
    assert "Density-Based" in paradigms
    assert "Hierarchical" in paradigms
    assert "Probabilistic" in paradigms

    # Check metrics validity
    for r in summary.results:
        assert -1.0 <= r.silhouette <= 1.0
        assert r.davies_bouldin >= 0.0
        assert r.calinski_harabasz >= 0.0
        assert 0.0 <= r.stability <= 1.0
        assert r.runtime_ms > 0.0
        assert 0.0 <= r.noise_ratio <= 1.0
        assert 0.0 <= r.composite_score <= 1.0

    assert summary.best_model_by_silhouette != ""
    assert summary.best_model_by_composite != ""


def test_benchmark_matrix_bootstrap_variances(sample_dataset: pd.DataFrame):
    """Verify standard deviations are calculated over bootstrap resamples."""
    runner = BenchmarkRunner(random_state=42)
    summary = runner.run_benchmark(
        data=sample_dataset,
        pipeline_config=PipelineConfig(scaling_strategy="standard", engineer_ratios=False),
        n_bootstraps=3,
    )

    for r in summary.results:
        assert hasattr(r, "silhouette_std")
        assert hasattr(r, "davies_bouldin_std")
        assert hasattr(r, "calinski_harabasz_std")
        assert hasattr(r, "stability_std")
        assert hasattr(r, "runtime_ms_std")
        assert r.stability_std >= 0.0


# ============================================================================
# 3. Ablation Matrix Tests
# ============================================================================

def test_ablation_matrix_synthesis(sample_dataset: pd.DataFrame):
    """Verify baseline vs. optimized ablation matrix calculation."""
    runner = BenchmarkRunner(random_state=42)

    # 1. Baseline: raw features, standard scaler, default k
    base_summary = runner.run_benchmark(
        data=sample_dataset,
        pipeline_config=PipelineConfig(scaling_strategy="standard", engineer_ratios=False),
        model_factories=runner.create_default_models(k=3),
        n_bootstraps=2,
    )

    # 2. Optimized: power transformed, engineered ratios, tuned models
    opt_summary = runner.run_benchmark(
        data=sample_dataset,
        pipeline_config=PipelineConfig(scaling_strategy="yeo_johnson", engineer_ratios=True),
        model_factories=runner.create_optimized_models(k=4),
        n_bootstraps=2,
    )

    ablation = generate_ablation_matrix(base_summary, opt_summary)

    assert isinstance(ablation, AblationMatrixSummary)
    assert len(ablation.ablation_entries) == 6

    for e in ablation.ablation_entries:
        assert e.delta_silhouette == pytest.approx(e.optimized_silhouette - e.baseline_silhouette, abs=1e-3)
        assert e.delta_davies_bouldin == pytest.approx(e.optimized_davies_bouldin - e.baseline_davies_bouldin, abs=1e-3)
        assert e.delta_calinski_harabasz == pytest.approx(e.optimized_calinski_harabasz - e.baseline_calinski_harabasz, abs=1e-1)
        assert e.delta_composite == pytest.approx(e.optimized_composite - e.baseline_composite, abs=1e-3)


# ============================================================================
# 4. LaTeX & Markdown Table Export Tests
# ============================================================================

def test_latex_table_export_structure(sample_dataset: pd.DataFrame):
    """Verify LaTeX table export contains required LaTeX elements."""
    summary = run_benchmark_matrix(data=sample_dataset, n_bootstraps=2)
    latex_out = to_latex_table(summary, include_std=False)

    assert r"\begin{table}" in latex_out
    assert r"\begin{tabular}" in latex_out
    assert r"\hline" in latex_out
    assert r"\end{tabular}" in latex_out
    assert r"\end{table}" in latex_out
    assert "Silhouette" in latex_out
    assert "K-Means" in latex_out
    assert "GMM" in latex_out

    # Test with standard deviations
    latex_std_out = to_latex_table(summary, include_std=True)
    assert r"\pm" in latex_std_out


def test_markdown_table_export_structure(sample_dataset: pd.DataFrame):
    """Verify Markdown table output format."""
    summary = run_benchmark_matrix(data=sample_dataset, n_bootstraps=2)
    md_out = to_markdown_table(summary, include_std=False)

    assert "| Model |" in md_out
    assert "| K-Means |" in md_out
    assert "| GMM |" in md_out
    assert "| HDBSCAN |" in md_out
    lines = [l.strip() for l in md_out.strip().split("\n")]
    assert len(lines) == 8  # header + separator + 6 model rows


def test_ablation_table_exporters(sample_dataset: pd.DataFrame):
    """Verify LaTeX and Markdown ablation tables."""
    runner = BenchmarkRunner(random_state=42)
    base_summary = runner.run_benchmark(data=sample_dataset, n_bootstraps=2)
    opt_summary = runner.run_benchmark(data=sample_dataset, n_bootstraps=2)

    ablation = generate_ablation_matrix(base_summary, opt_summary)

    latex_ablation = to_latex_ablation_table(ablation)
    assert r"\begin{table}" in latex_ablation
    assert r"\textbf{$\Delta$ Sil" in latex_ablation

    md_ablation = to_markdown_ablation_table(ablation)
    assert "| Model |" in md_ablation
    assert "Δ Sil" in md_ablation
    assert "Δ DB" in md_ablation


def test_json_serialization(sample_dataset: pd.DataFrame):
    """Verify summary dataclass serialization to dict and JSON string."""
    summary = run_benchmark_matrix(data=sample_dataset, n_bootstraps=2)
    d = to_dict(summary)
    assert isinstance(d, dict)
    assert "results" in d
    assert len(d["results"]) == 6

    json_str = to_json(summary)
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["dataset_name"] == summary.dataset_name
