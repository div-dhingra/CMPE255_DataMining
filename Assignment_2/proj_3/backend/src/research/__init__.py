"""
Research Literature Alignment & Benchmark Synthesis Package
Provides:
- Literature repository & citations (Rousseeuw 1987, Davies-Bouldin 1979, Calinski-Harabasz 1974,
  Arthur-Vassilvitskii 2007, Campello 2013, McInnes 2018, Von Luxburg 2010, Satopaa 2011, Sakana AI 2024)
- Algorithm & Metric Taxonomies
- Cross-Paradigm Benchmark Matrix Execution & Statistical Evaluation
- Publication-quality LaTeX & Markdown Table Generators
- Baseline vs. Hill-Climbed Ablation Matrix
"""

from .literature import (
    Citation,
    AlgorithmTaxonomyEntry,
    MetricTaxonomyEntry,
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

from .benchmark_matrix import (
    BenchmarkModelResult,
    BenchmarkMatrixSummary,
    AblationEntry,
    AblationMatrixSummary,
    BenchmarkRunner,
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

__all__ = [
    # Literature
    "Citation",
    "AlgorithmTaxonomyEntry",
    "MetricTaxonomyEntry",
    "LITERATURE_REPOSITORY",
    "ALGORITHM_TAXONOMY",
    "METRIC_TAXONOMY",
    "AUTORESEARCH_PRINCIPLES",
    "get_literature_repository",
    "get_citations",
    "get_citation_by_id",
    "get_bibtex",
    "get_algorithm_taxonomy",
    "get_metric_formulations",
    "get_autoresearch_principles",
    # Benchmark Matrix
    "BenchmarkModelResult",
    "BenchmarkMatrixSummary",
    "AblationEntry",
    "AblationMatrixSummary",
    "BenchmarkRunner",
    "compute_composite_fitness",
    "run_benchmark_matrix",
    "generate_ablation_matrix",
    "to_latex_table",
    "to_markdown_table",
    "to_latex_ablation_table",
    "to_markdown_ablation_table",
    "to_dict",
    "to_json",
]
