"""
Research Literature, Benchmark Matrix & Academic Synthesis Route Handlers.
Endpoints:
- GET  /api/v1/research/literature: Citations repository, algorithm taxonomies, metric formulations
- GET  /api/v1/research/benchmark-matrix: Cross-paradigm 6-model benchmark with bootstrap bounds
- GET  /api/v1/research/ablations: Baseline vs. Hill-Climbed ablation synthesis matrix
- POST /api/v1/research/export: Export publication-ready LaTeX / Markdown tables
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict, is_dataclass
from fastapi import APIRouter, Query, HTTPException, status
import numpy as np

from ..schemas import (
    LiteratureResponse,
    CitationSchema,
    AlgorithmTaxonomySchema,
    MetricTaxonomySchema,
    BenchmarkMatrixResponse,
    BenchmarkModelResultSchema,
    AblationMatrixResponse,
    AblationEntrySchema,
    TableExportRequest,
    TableExportResponse,
)
from ..state import state
from crisp_dm.data_preparation import PipelineConfig
from research.literature import (
    get_citations,
    get_bibtex,
    get_algorithm_taxonomy,
    get_metric_formulations,
    get_autoresearch_principles,
    ALGORITHM_TAXONOMY,
    METRIC_TAXONOMY,
    AUTORESEARCH_PRINCIPLES,
)
from research.benchmark_matrix import (
    BenchmarkRunner,
    run_benchmark_matrix,
    generate_ablation_matrix,
    to_latex_table,
    to_markdown_table,
    to_latex_ablation_table,
    to_markdown_ablation_table,
    BenchmarkMatrixSummary,
    AblationMatrixSummary,
)

router = APIRouter(prefix="/research", tags=["Research & Benchmarks"])

# In-memory benchmark cache
_cached_benchmark: Optional[BenchmarkMatrixSummary] = None
_cached_ablation: Optional[AblationMatrixSummary] = None


@router.get(
    "/literature",
    response_model=LiteratureResponse,
    summary="Get citations repository, algorithmic taxonomies, and metric formulations",
)
def get_literature(
    topic: Optional[str] = Query(default=None, description="Filter citations by topic (e.g. 'validation', 'density', 'partitioning', 'autoresearch')")
) -> LiteratureResponse:
    """
    Returns peer-reviewed citations, BibTeX bibliography, algorithmic complexity trade-offs,
    mathematical metric formulations, and autonomous research principles.
    """
    if not isinstance(topic, str):
        topic = None

    citations_raw = get_citations(topic=topic)

    citations: List[CitationSchema] = []
    for c in citations_raw:
        if isinstance(c, dict):
            citations.append(CitationSchema(**c))
        elif is_dataclass(c):
            citations.append(CitationSchema(**asdict(c)))

    algorithms: Dict[str, AlgorithmTaxonomySchema] = {}
    for k, a in ALGORITHM_TAXONOMY.items():
        if isinstance(a, dict):
            algorithms[k] = AlgorithmTaxonomySchema(**a)
        elif is_dataclass(a):
            algorithms[k] = AlgorithmTaxonomySchema(**asdict(a))

    metrics: Dict[str, MetricTaxonomySchema] = {}
    for k, m in METRIC_TAXONOMY.items():
        if isinstance(m, dict):
            metrics[k] = MetricTaxonomySchema(**m)
        elif is_dataclass(m):
            metrics[k] = MetricTaxonomySchema(**asdict(m))

    principles_raw = get_autoresearch_principles()
    if isinstance(principles_raw, list):
        principles_dict = {
            item.get("principle", f"principle_{i}"): item.get("description", "")
            for i, item in enumerate(principles_raw)
        }
    elif isinstance(principles_raw, dict):
        principles_dict = principles_raw
    else:
        principles_dict = {}

    return LiteratureResponse(
        citations=citations,
        algorithms=algorithms,
        metrics=metrics,
        principles=principles_dict,
    )


def _compute_benchmark(df, n_bootstraps: int = 3, sample_size: int = 800) -> BenchmarkMatrixSummary:
    """Helper to run benchmark matrix cleanly."""
    runner = BenchmarkRunner(random_state=42)
    sample_df = df.head(sample_size) if len(df) > sample_size else df
    return runner.run_benchmark(
        data=sample_df,
        n_bootstraps=n_bootstraps,
        dataset_name="Kaggle Credit Card Customer Dataset",
    )


def _compute_ablation(df) -> AblationMatrixSummary:
    """Helper to run ablation matrix cleanly."""
    runner = BenchmarkRunner(random_state=42)
    sample_df = df.head(800) if len(df) > 800 else df
    base_conf = PipelineConfig(scaling_strategy="standard", engineer_ratios=False)
    opt_conf = PipelineConfig(scaling_strategy="yeo_johnson", engineer_ratios=True)
    base_sum = runner.run_benchmark(data=sample_df, pipeline_config=base_conf, n_bootstraps=2, dataset_name="Baseline")
    opt_sum = runner.run_benchmark(data=sample_df, pipeline_config=opt_conf, n_bootstraps=2, dataset_name="Optimized")
    return generate_ablation_matrix(base_sum, opt_sum)


@router.get(
    "/benchmark-matrix",
    response_model=BenchmarkMatrixResponse,
    summary="Get cross-paradigm benchmark matrix with bootstrap variance bounds",
)
def get_benchmark_matrix(
    recompute: bool = Query(default=False, description="Force recomputation of the benchmark matrix"),
    n_bootstraps: int = Query(default=3, ge=1, le=10, description="Bootstrap resampling iterations"),
    sample_size: int = Query(default=800, ge=100, le=5000, description="Sample size for bootstrap evaluation"),
) -> BenchmarkMatrixResponse:
    """
    Executes cross-paradigm benchmark evaluation across all 6 clustering models
    (KMeans, KMedoids, DBSCAN, HDBSCAN, Agglomerative, GMM) computing empirical
    Silhouette, Davies-Bouldin, Calinski-Harabasz, Stability (ARI), and composite fitness.
    """
    global _cached_benchmark

    if _cached_benchmark is None or recompute:
        df = state.get_dataset()
        _cached_benchmark = _compute_benchmark(df, n_bootstraps=n_bootstraps, sample_size=sample_size)

    b = _cached_benchmark
    models_list: List[BenchmarkModelResultSchema] = []
    for m in b.results:
        m_key = m.model_name.lower().replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("+", "plus")
        models_list.append(
            BenchmarkModelResultSchema(
                model_key=m_key,
                model_name=m.model_name,
                paradigm=m.paradigm,
                n_clusters=m.n_clusters,
                silhouette_mean=round(float(m.silhouette), 4),
                silhouette_std=round(float(m.silhouette_std), 4),
                davies_bouldin_mean=round(float(m.davies_bouldin), 4),
                davies_bouldin_std=round(float(m.davies_bouldin_std), 4),
                calinski_harabasz_mean=round(float(m.calinski_harabasz), 2),
                calinski_harabasz_std=round(float(m.calinski_harabasz_std), 2),
                stability_ari_mean=round(float(m.stability), 4),
                stability_ari_std=round(float(m.stability_std), 4),
                runtime_ms=round(float(m.runtime_ms), 2),
                noise_ratio=round(float(m.noise_ratio), 4),
                composite_fitness=round(float(m.composite_score), 4),
                hyperparameters=m.hyperparameters,
            )
        )

    best_fit = max(r.composite_score for r in b.results) if b.results else 0.0

    return BenchmarkMatrixResponse(
        n_samples=b.n_samples,
        n_features=b.n_features,
        timestamp=b.timestamp,
        models=models_list,
        best_model_key=b.best_model_by_composite or "kmeans",
        best_fitness=round(float(best_fit), 4),
    )


@router.get(
    "/ablations",
    response_model=AblationMatrixResponse,
    summary="Get baseline vs. hill-climbed ablation matrix",
)
def get_ablation_matrix(
    recompute: bool = Query(default=False, description="Force recomputation of the ablation matrix")
) -> AblationMatrixResponse:
    """
    Computes performance improvement deltas between default baselines and
    Autoresearch-optimized configurations across all paradigms.
    """
    global _cached_ablation

    if _cached_ablation is None or recompute:
        df = state.get_dataset()
        _cached_ablation = _compute_ablation(df)

    a = _cached_ablation
    ablations_list: List[AblationEntrySchema] = []
    for item in a.ablation_entries:
        m_key = item.model_name.lower().replace(" ", "_").replace("-", "_")
        ablations_list.append(
            AblationEntrySchema(
                model_key=m_key,
                model_name=item.model_name,
                paradigm="Clustering",
                baseline_config={"description": item.baseline_config},
                optimized_config={"description": item.optimized_config},
                baseline_fitness=round(float(item.baseline_composite), 4),
                optimized_fitness=round(float(item.optimized_composite), 4),
                delta_fitness=round(float(item.delta_composite), 4),
                baseline_silhouette=round(float(item.baseline_silhouette), 4),
                optimized_silhouette=round(float(item.optimized_silhouette), 4),
                delta_silhouette=round(float(item.delta_silhouette), 4),
                baseline_davies_bouldin=round(float(item.baseline_davies_bouldin), 4),
                optimized_davies_bouldin=round(float(item.optimized_davies_bouldin), 4),
                delta_davies_bouldin=round(float(item.delta_davies_bouldin), 4),
                baseline_calinski_harabasz=round(float(item.baseline_calinski_harabasz), 2),
                optimized_calinski_harabasz=round(float(item.optimized_calinski_harabasz), 2),
                delta_calinski_harabasz=round(float(item.delta_calinski_harabasz), 2),
                baseline_stability_ari=round(float(item.baseline_stability), 4),
                optimized_stability_ari=round(float(item.optimized_stability), 4),
                delta_stability_ari=round(float(item.delta_stability), 4),
                percentage_improvement=round(float(item.improvement_pct), 2),
            )
        )

    return AblationMatrixResponse(
        timestamp=time_str_now(),
        ablations=ablations_list,
        overall_mean_fitness_improvement_pct=round(float(a.mean_composite_gain * 100.0), 2),
    )


def time_str_now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@router.post(
    "/export",
    response_model=TableExportResponse,
    summary="Export benchmark or ablation tables to LaTeX or Markdown format",
)
def export_research_table(request: TableExportRequest) -> TableExportResponse:
    """
    Generates publication-quality LaTeX or Markdown comparison tables.
    """
    global _cached_benchmark, _cached_ablation

    df = state.get_dataset()
    fmt = request.format.lower()
    t_type = request.table_type.lower()

    if t_type == "benchmark":
        if _cached_benchmark is None:
            _cached_benchmark = _compute_benchmark(df, n_bootstraps=2, sample_size=600)

        if fmt == "latex":
            content = to_latex_table(_cached_benchmark, caption="Multi-Paradigm Clustering Benchmark", label="tab:benchmark")
        elif fmt == "markdown":
            content = to_markdown_table(_cached_benchmark)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{request.format}'. Choose 'latex' or 'markdown'.",
            )

    elif t_type == "ablation":
        if _cached_ablation is None:
            _cached_ablation = _compute_ablation(df)

        if fmt == "latex":
            content = to_latex_ablation_table(_cached_ablation, caption="Clustering Pipeline Ablation Table", label="tab:ablation")
        elif fmt == "markdown":
            # Generate markdown table from ablation summary
            lines = [
                "| Model | Base Sil | Opt Sil | Δ Sil ↑ | Base DB | Opt DB | Δ DB ↓ | Base Fitness | Opt Fitness | Gain % |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for e in _cached_ablation.ablation_entries:
                lines.append(
                    f"| {e.model_name} | {e.baseline_silhouette:.3f} | {e.optimized_silhouette:.3f} | {e.delta_silhouette:+.3f} | {e.baseline_davies_bouldin:.3f} | {e.optimized_davies_bouldin:.3f} | {e.delta_davies_bouldin:+.3f} | {e.baseline_composite:.3f} | {e.optimized_composite:.3f} | {e.improvement_pct:+.1f}% |"
                )
            content = "\n".join(lines) + "\n"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{request.format}'. Choose 'latex' or 'markdown'.",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported table_type '{request.table_type}'. Choose 'benchmark' or 'ablation'.",
        )

    return TableExportResponse(
        format=fmt,
        table_type=t_type,
        content=content,
    )
