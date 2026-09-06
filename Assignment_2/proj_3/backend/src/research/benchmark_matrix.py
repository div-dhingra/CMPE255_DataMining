"""
Cross-Paradigm Clustering Benchmark Matrix & Ablation Synthesis Engine
Executes genuine multi-model clustering benchmarks across all 4 paradigms (6 models),
computes statistical validation metrics with bootstrap standard deviation bounds,
generates publication-ready LaTeX and Markdown tables, and calculates baseline vs.
hill-climbed ablation matrices.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field, asdict
import time
import json
import numpy as np
import pandas as pd

try:
    from ..crisp_dm.data_preparation import (
        PipelineConfig,
        DataPreparationPipeline,
    )
    from ..crisp_dm.models import (
        ClusteringModelBase,
        KMeansModel,
        KMedoidsModel,
        DBSCANModel,
        HDBSCANModel,
        AgglomerativeModel,
        GaussianMixtureModel,
    )
    from ..crisp_dm.evaluation import (
        compute_silhouette_score,
        compute_davies_bouldin_index,
        compute_calinski_harabasz_score,
        compute_subsampling_stability,
        adjusted_rand_index,
        evaluate_clustering_solution,
    )
except (ImportError, ValueError):
    from crisp_dm.data_preparation import (
        PipelineConfig,
        DataPreparationPipeline,
    )
    from crisp_dm.models import (
        ClusteringModelBase,
        KMeansModel,
        KMedoidsModel,
        DBSCANModel,
        HDBSCANModel,
        AgglomerativeModel,
        GaussianMixtureModel,
    )
    from crisp_dm.evaluation import (
        compute_silhouette_score,
        compute_davies_bouldin_index,
        compute_calinski_harabasz_score,
        compute_subsampling_stability,
        adjusted_rand_index,
        evaluate_clustering_solution,
    )


@dataclass
class BenchmarkModelResult:
    """Individual model evaluation result within benchmark matrix."""
    model_name: str
    paradigm: str
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability: float
    runtime_ms: float
    noise_ratio: float
    n_clusters: int
    composite_score: float = 0.0
    # Bootstrap standard deviations (optional)
    silhouette_std: float = 0.0
    davies_bouldin_std: float = 0.0
    calinski_harabasz_std: float = 0.0
    stability_std: float = 0.0
    runtime_ms_std: float = 0.0
    noise_ratio_std: float = 0.0
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMatrixSummary:
    """Complete summary container for a benchmark run."""
    dataset_name: str
    n_samples: int
    n_features: int
    results: List[BenchmarkModelResult] = field(default_factory=list)
    timestamp: str = ""
    best_model_by_silhouette: str = ""
    best_model_by_davies_bouldin: str = ""
    best_model_by_calinski_harabasz: str = ""
    best_model_by_stability: str = ""
    best_model_by_composite: str = ""


@dataclass
class AblationEntry:
    """Ablation delta entry comparing baseline configuration to optimized."""
    factor_name: str
    model_name: str
    baseline_config: str
    optimized_config: str
    baseline_silhouette: float
    optimized_silhouette: float
    delta_silhouette: float
    baseline_davies_bouldin: float
    optimized_davies_bouldin: float
    delta_davies_bouldin: float
    baseline_calinski_harabasz: float
    optimized_calinski_harabasz: float
    delta_calinski_harabasz: float
    baseline_stability: float
    optimized_stability: float
    delta_stability: float
    baseline_composite: float
    optimized_composite: float
    delta_composite: float
    improvement_pct: float = 0.0


@dataclass
class AblationMatrixSummary:
    """Summary of baseline vs. hill-climbed ablation comparisons."""
    dataset_name: str
    ablation_entries: List[AblationEntry] = field(default_factory=list)
    mean_silhouette_gain: float = 0.0
    mean_db_reduction: float = 0.0
    mean_ch_gain: float = 0.0
    mean_stability_gain: float = 0.0
    mean_composite_gain: float = 0.0


# ============================================================================
# 1. Composite Fitness Helper
# ============================================================================

def compute_composite_fitness(
    silhouette: float,
    davies_bouldin: float,
    calinski_harabasz: float,
    stability: float,
    noise_ratio: float = 0.0,
    w_sil: float = 0.40,
    w_db: float = 0.25,
    w_ch: float = 0.15,
    w_stab: float = 0.20,
) -> float:
    """
    Computes bounded composite fitness in [0, 1]:
    F = w1 * S_norm + w2 * (1 - DB_norm) + w3 * CH_norm + w4 * Stability - P_noise
    """
    # Normalize silhouette from [-1, 1] to [0, 1]
    s_norm = max(0.0, min(1.0, (silhouette + 1.0) / 2.0))
    
    # Normalize DB index (lower is better, cap at 5.0)
    db_norm = max(0.0, min(1.0, davies_bouldin / 5.0))
    db_score = 1.0 - db_norm
    
    # Normalize CH index via log scale
    ch_val = max(0.0, calinski_harabasz)
    ch_norm = min(1.0, np.log1p(ch_val) / np.log1p(10000.0))
    
    # Stability is already in [0, 1]
    stab_norm = max(0.0, min(1.0, stability))
    
    # Noise penalty for density models if noise exceeds 10%
    p_noise = 0.5 * max(0.0, noise_ratio - 0.10)
    
    fitness = (w_sil * s_norm) + (w_db * db_score) + (w_ch * ch_norm) + (w_stab * stab_norm) - p_noise
    return float(max(0.0, min(1.0, fitness)))


# ============================================================================
# 2. Benchmark Runner
# ============================================================================

class BenchmarkRunner:
    """
    Orchestrates end-to-end benchmark matrix runs across models and preprocessing.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def create_default_models(self, k: int = 4) -> List[Tuple[str, str, Callable[[], ClusteringModelBase]]]:
        """
        Creates standard model factories for all 6 models across 4 paradigms.
        Returns: List of (model_name, paradigm, factory_fn)
        """
        return [
            ("K-Means", "Partitioning", lambda: KMeansModel(n_clusters=k, random_state=self.random_state)),
            ("K-Medoids", "Partitioning", lambda: KMedoidsModel(n_clusters=k, metric="euclidean", random_state=self.random_state)),
            ("DBSCAN", "Density-Based", lambda: DBSCANModel(eps=1.2, min_samples=5)),
            ("HDBSCAN", "Density-Based", lambda: HDBSCANModel(min_cluster_size=10, min_samples=5)),
            ("Agglomerative", "Hierarchical", lambda: AgglomerativeModel(n_clusters=k, linkage="ward")),
            ("GMM", "Probabilistic", lambda: GaussianMixtureModel(n_clusters=k, covariance_type="full", random_state=self.random_state)),
        ]

    def create_optimized_models(self, k: int = 5) -> List[Tuple[str, str, Callable[[], ClusteringModelBase]]]:
        """
        Creates optimized model factories (hill-climbed hyperparameters).
        Returns: List of (model_name, paradigm, factory_fn)
        """
        return [
            ("K-Means", "Partitioning", lambda: KMeansModel(n_clusters=k, init="k-means++", max_iter=500, random_state=self.random_state)),
            ("K-Medoids", "Partitioning", lambda: KMedoidsModel(n_clusters=k, metric="manhattan", max_iter=200, random_state=self.random_state)),
            ("DBSCAN", "Density-Based", lambda: DBSCANModel(eps=0.85, min_samples=4)),
            ("HDBSCAN", "Density-Based", lambda: HDBSCANModel(min_cluster_size=8, min_samples=3)),
            ("Agglomerative", "Hierarchical", lambda: AgglomerativeModel(n_clusters=k, linkage="ward")),
            ("GMM", "Probabilistic", lambda: GaussianMixtureModel(n_clusters=k, covariance_type="diag", reg_covar=1e-5, random_state=self.random_state)),
        ]

    def evaluate_model(
        self,
        X: np.ndarray,
        model_name: str,
        paradigm: str,
        factory_fn: Callable[[], ClusteringModelBase],
        n_bootstraps: int = 5,
        subsample_ratio: float = 0.80,
    ) -> BenchmarkModelResult:
        """
        Evaluates a single model on feature matrix X with bootstrap standard deviations.
        """
        X_arr = np.asarray(X, dtype=float)
        n_samples = len(X_arr)
        rng = np.random.default_rng(self.random_state)

        # 1. Main model execution & runtime measurement
        t0 = time.perf_counter()
        main_model = factory_fn()
        labels = main_model.fit_predict(X_arr)
        t1 = time.perf_counter()
        runtime_ms = (t1 - t0) * 1000.0

        # Primary metrics
        eval_dict = evaluate_clustering_solution(X_arr, labels, main_model)
        sil = eval_dict["silhouette_score"]
        db = eval_dict["davies_bouldin_index"]
        ch = eval_dict["calinski_harabasz_score"]
        noise_ratio = eval_dict["noise_ratio"]
        n_clusters = eval_dict["n_clusters"]

        # 2. Stability and bootstrap metrics
        ari_list: List[float] = []
        sil_list: List[float] = [sil]
        db_list: List[float] = [db]
        ch_list: List[float] = [ch]
        time_list: List[float] = [runtime_ms]
        noise_list: List[float] = [noise_ratio]

        sub_size = max(10, int(round(n_samples * subsample_ratio)))

        if n_bootstraps > 1 and n_samples >= 15:
            for b in range(n_bootstraps):
                sub_indices = rng.choice(n_samples, size=sub_size, replace=False)
                X_sub = X_arr[sub_indices]

                tb0 = time.perf_counter()
                b_model = factory_fn()
                b_labels = b_model.fit_predict(X_sub)
                tb1 = time.perf_counter()

                time_list.append((tb1 - tb0) * 1000.0)

                # Stability ARI
                base_sub_labels = labels[sub_indices]
                ari = adjusted_rand_index(base_sub_labels, b_labels)
                ari_list.append(ari)

                # Bootstrap internal metrics
                b_eval = evaluate_clustering_solution(X_sub, b_labels, b_model)
                sil_list.append(b_eval["silhouette_score"])
                db_list.append(b_eval["davies_bouldin_index"])
                ch_list.append(b_eval["calinski_harabasz_score"])
                noise_list.append(b_eval["noise_ratio"])

            stability = float(np.mean(ari_list))
            stability_std = float(np.std(ari_list))
        else:
            stability = 1.0
            stability_std = 0.0

        composite = compute_composite_fitness(sil, db, ch, stability, noise_ratio)

        params = main_model.get_params() if hasattr(main_model, "get_params") else {}

        return BenchmarkModelResult(
            model_name=model_name,
            paradigm=paradigm,
            silhouette=round(sil, 4),
            davies_bouldin=round(db, 4),
            calinski_harabasz=round(ch, 2),
            stability=round(stability, 4),
            runtime_ms=round(runtime_ms, 2),
            noise_ratio=round(noise_ratio, 4),
            n_clusters=n_clusters,
            composite_score=round(composite, 4),
            silhouette_std=round(float(np.std(sil_list)), 4),
            davies_bouldin_std=round(float(np.std(db_list)), 4),
            calinski_harabasz_std=round(float(np.std(ch_list)), 2),
            stability_std=round(stability_std, 4),
            runtime_ms_std=round(float(np.std(time_list)), 2),
            noise_ratio_std=round(float(np.std(noise_list)), 4),
            hyperparameters=params,
        )

    def run_benchmark(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        pipeline_config: Optional[PipelineConfig] = None,
        model_factories: Optional[List[Tuple[str, str, Callable[[], ClusteringModelBase]]]] = None,
        dataset_name: str = "Credit Card Customer Dataset",
        n_bootstraps: int = 5,
    ) -> BenchmarkMatrixSummary:
        """
        Executes benchmark matrix on dataset across all models.
        """
        # Data preparation
        if isinstance(data, pd.DataFrame):
            cfg = pipeline_config or PipelineConfig()
            pipeline = DataPreparationPipeline(cfg)
            X = pipeline.fit_transform(data)
            n_samples, n_features = X.shape
        else:
            X = np.asarray(data, dtype=float)
            n_samples, n_features = X.shape
            cfg = pipeline_config or PipelineConfig()

        factories = model_factories or self.create_default_models()
        results: List[BenchmarkModelResult] = []

        for name, paradigm, factory in factories:
            res = self.evaluate_model(
                X=X,
                model_name=name,
                paradigm=paradigm,
                factory_fn=factory,
                n_bootstraps=n_bootstraps,
            )
            res.preprocessing_config = asdict(cfg) if isinstance(cfg, PipelineConfig) else {}
            results.append(res)

        # Determine best models
        best_sil = max(results, key=lambda r: r.silhouette).model_name if results else ""
        best_db = min(results, key=lambda r: r.davies_bouldin).model_name if results else ""
        best_ch = max(results, key=lambda r: r.calinski_harabasz).model_name if results else ""
        best_stab = max(results, key=lambda r: r.stability).model_name if results else ""
        best_comp = max(results, key=lambda r: r.composite_score).model_name if results else ""

        return BenchmarkMatrixSummary(
            dataset_name=dataset_name,
            n_samples=n_samples,
            n_features=n_features,
            results=results,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            best_model_by_silhouette=best_sil,
            best_model_by_davies_bouldin=best_db,
            best_model_by_calinski_harabasz=best_ch,
            best_model_by_stability=best_stab,
            best_model_by_composite=best_comp,
        )


def run_benchmark_matrix(
    data: Union[pd.DataFrame, np.ndarray],
    pipeline_config: Optional[PipelineConfig] = None,
    n_bootstraps: int = 5,
    random_state: int = 42,
    dataset_name: str = "Credit Card Customer Dataset",
) -> BenchmarkMatrixSummary:
    """Convenience function to run benchmark matrix directly."""
    runner = BenchmarkRunner(random_state=random_state)
    return runner.run_benchmark(
        data=data,
        pipeline_config=pipeline_config,
        dataset_name=dataset_name,
        n_bootstraps=n_bootstraps,
    )


# ============================================================================
# 3. Baseline vs. Hill-Climbed Ablation Matrix
# ============================================================================

def generate_ablation_matrix(
    baseline_summary: BenchmarkMatrixSummary,
    optimized_summary: BenchmarkMatrixSummary,
    factor_name: str = "Autoresearch Full Optimization",
) -> AblationMatrixSummary:
    """
    Computes comparative ablation metrics between baseline and optimized models.
    """
    base_map = {r.model_name: r for r in baseline_summary.results}
    opt_map = {r.model_name: r for r in optimized_summary.results}

    entries: List[AblationEntry] = []

    for name, opt_r in opt_map.items():
        if name in base_map:
            base_r = base_map[name]
            d_sil = opt_r.silhouette - base_r.silhouette
            d_db = opt_r.davies_bouldin - base_r.davies_bouldin  # negative is improvement
            d_ch = opt_r.calinski_harabasz - base_r.calinski_harabasz
            d_stab = opt_r.stability - base_r.stability
            d_comp = opt_r.composite_score - base_r.composite_score
            imp_pct = ((opt_r.composite_score - base_r.composite_score) / max(0.001, base_r.composite_score)) * 100.0

            entry = AblationEntry(
                factor_name=factor_name,
                model_name=name,
                baseline_config="Default (StandardScaler, Raw)",
                optimized_config="Autoresearch (Yeo-Johnson, Ratios, Tuned)",
                baseline_silhouette=base_r.silhouette,
                optimized_silhouette=opt_r.silhouette,
                delta_silhouette=round(d_sil, 4),
                baseline_davies_bouldin=base_r.davies_bouldin,
                optimized_davies_bouldin=opt_r.davies_bouldin,
                delta_davies_bouldin=round(d_db, 4),
                baseline_calinski_harabasz=base_r.calinski_harabasz,
                optimized_calinski_harabasz=opt_r.calinski_harabasz,
                delta_calinski_harabasz=round(d_ch, 2),
                baseline_stability=base_r.stability,
                optimized_stability=opt_r.stability,
                delta_stability=round(d_stab, 4),
                baseline_composite=base_r.composite_score,
                optimized_composite=opt_r.composite_score,
                delta_composite=round(d_comp, 4),
                improvement_pct=round(imp_pct, 2),
            )
            entries.append(entry)

    m_sil = float(np.mean([e.delta_silhouette for e in entries])) if entries else 0.0
    m_db = float(np.mean([e.delta_davies_bouldin for e in entries])) if entries else 0.0
    m_ch = float(np.mean([e.delta_calinski_harabasz for e in entries])) if entries else 0.0
    m_stab = float(np.mean([e.delta_stability for e in entries])) if entries else 0.0
    m_comp = float(np.mean([e.delta_composite for e in entries])) if entries else 0.0

    return AblationMatrixSummary(
        dataset_name=baseline_summary.dataset_name,
        ablation_entries=entries,
        mean_silhouette_gain=round(m_sil, 4),
        mean_db_reduction=round(-m_db, 4),  # positive reduction
        mean_ch_gain=round(m_ch, 2),
        mean_stability_gain=round(m_stab, 4),
        mean_composite_gain=round(m_comp, 4),
    )


# ============================================================================
# 4. LaTeX and Markdown Exporters
# ============================================================================

def to_latex_table(
    results: Union[List[BenchmarkModelResult], BenchmarkMatrixSummary],
    caption: str = "CRISP-DM Multi-Paradigm Clustering Benchmark Comparison",
    label: str = "tab:clustering_benchmark",
    include_std: bool = False,
) -> str:
    """
    Generates a publication-quality LaTeX table string.
    """
    rows = results.results if isinstance(results, BenchmarkMatrixSummary) else results

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
    ]

    if include_std:
        lines.extend([
            r"\begin{tabular}{llccccc}",
            r"\hline",
            r"\textbf{Model} & \textbf{Paradigm} & \textbf{Silhouette $\uparrow$} & \textbf{DB Index $\downarrow$} & \textbf{CH Score $\uparrow$} & \textbf{Stability $\uparrow$} & \textbf{Time (ms)} \\",
            r"\hline",
        ])
        for r in rows:
            sil_str = f"{r.silhouette:.2f} $\\pm$ {r.silhouette_std:.2f}"
            db_str = f"{r.davies_bouldin:.2f} $\\pm$ {r.davies_bouldin_std:.2f}"
            ch_str = f"{r.calinski_harabasz:.0f} $\\pm$ {r.calinski_harabasz_std:.0f}"
            stab_str = f"{r.stability:.2f} $\\pm$ {r.stability_std:.2f}"
            time_str = f"{r.runtime_ms:.1f} $\\pm$ {r.runtime_ms_std:.1f}"
            lines.append(f"{r.model_name} & {r.paradigm} & {sil_str} & {db_str} & {ch_str} & {stab_str} & {time_str} \\\\")
    else:
        lines.extend([
            r"\begin{tabular}{llccccc}",
            r"\hline",
            r"\textbf{Model} & \textbf{Paradigm} & \textbf{Silhouette $\uparrow$} & \textbf{DB Index $\downarrow$} & \textbf{CH Score $\uparrow$} & \textbf{Stability $\uparrow$} & \textbf{Time (ms)} \\",
            r"\hline",
        ])
        for r in rows:
            lines.append(f"{r.model_name} & {r.paradigm} & {r.silhouette:.2f} & {r.davies_bouldin:.2f} & {r.calinski_harabasz:.0f} & {r.stability:.2f} & {r.runtime_ms:.1f} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def to_markdown_table(
    results: Union[List[BenchmarkModelResult], BenchmarkMatrixSummary],
    include_std: bool = False,
) -> str:
    """
    Generates a formatted Markdown comparative table string.
    """
    rows = results.results if isinstance(results, BenchmarkMatrixSummary) else results

    lines = [
        "| Model | Paradigm | Silhouette ↑ | DB Index ↓ | CH Score ↑ | Stability ↑ | Noise Ratio | Time (ms) | Composite |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in rows:
        if include_std and r.silhouette_std > 0:
            sil = f"{r.silhouette:.3f} ± {r.silhouette_std:.3f}"
            db = f"{r.davies_bouldin:.3f} ± {r.davies_bouldin_std:.3f}"
            ch = f"{r.calinski_harabasz:.1f} ± {r.calinski_harabasz_std:.1f}"
            stab = f"{r.stability:.3f} ± {r.stability_std:.3f}"
            time_str = f"{r.runtime_ms:.1f} ± {r.runtime_ms_std:.1f}"
        else:
            sil = f"{r.silhouette:.3f}"
            db = f"{r.davies_bouldin:.3f}"
            ch = f"{r.calinski_harabasz:.1f}"
            stab = f"{r.stability:.3f}"
            time_str = f"{r.runtime_ms:.1f}"

        noise = f"{r.noise_ratio:.1%}" if r.noise_ratio > 0 else "0.0%"
        comp = f"{r.composite_score:.3f}"

        lines.append(f"| {r.model_name} | {r.paradigm} | {sil} | {db} | {ch} | {stab} | {noise} | {time_str} | {comp} |")

    return "\n".join(lines) + "\n"


def to_latex_ablation_table(
    ablation: AblationMatrixSummary,
    caption: str = "Systematic Ablation: Baseline vs. Autoresearch Hill-Climbed Configurations",
    label: str = "tab:clustering_ablation",
) -> str:
    """
    Generates LaTeX ablation comparison table with delta indicators.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"\textbf{Model} & \textbf{Base Sil} & \textbf{Opt Sil} & \textbf{$\Delta$ Sil $\uparrow$} & \textbf{Base DB} & \textbf{Opt DB} & \textbf{$\Delta$ DB $\downarrow$} \\",
        r"\hline",
    ]

    for e in ablation.ablation_entries:
        d_sil_sign = f"+{e.delta_silhouette:.2f}" if e.delta_silhouette >= 0 else f"{e.delta_silhouette:.2f}"
        d_db_sign = f"{e.delta_davies_bouldin:.2f}" if e.delta_davies_bouldin <= 0 else f"+{e.delta_davies_bouldin:.2f}"
        lines.append(f"{e.model_name} & {e.baseline_silhouette:.2f} & {e.optimized_silhouette:.2f} & {d_sil_sign} & {e.baseline_davies_bouldin:.2f} & {e.optimized_davies_bouldin:.2f} & {d_db_sign} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def to_markdown_ablation_table(ablation: AblationMatrixSummary) -> str:
    """
    Generates Markdown ablation comparison table.
    """
    lines = [
        "| Model | Base Sil | Opt Sil | Δ Sil ↑ | Base DB | Opt DB | Δ DB ↓ | Base CH | Opt CH | Δ CH ↑ | Base ARI | Opt ARI | Δ ARI ↑ | Base Fitness | Opt Fitness | Δ Fitness |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for e in ablation.ablation_entries:
        d_sil = f"+{e.delta_silhouette:.3f}" if e.delta_silhouette >= 0 else f"{e.delta_silhouette:.3f}"
        d_db = f"{e.delta_davies_bouldin:.3f}" if e.delta_davies_bouldin <= 0 else f"+{e.delta_davies_bouldin:.3f}"
        d_ch = f"+{e.delta_calinski_harabasz:.1f}" if e.delta_calinski_harabasz >= 0 else f"{e.delta_calinski_harabasz:.1f}"
        d_stab = f"+{e.delta_stability:.3f}" if e.delta_stability >= 0 else f"{e.delta_stability:.3f}"
        d_comp = f"+{e.delta_composite:.3f}" if e.delta_composite >= 0 else f"{e.delta_composite:.3f}"

        lines.append(
            f"| {e.model_name} | {e.baseline_silhouette:.3f} | {e.optimized_silhouette:.3f} | **{d_sil}** | "
            f"{e.baseline_davies_bouldin:.3f} | {e.optimized_davies_bouldin:.3f} | **{d_db}** | "
            f"{e.baseline_calinski_harabasz:.1f} | {e.optimized_calinski_harabasz:.1f} | **{d_ch}** | "
            f"{e.baseline_stability:.3f} | {e.optimized_stability:.3f} | **{d_stab}** | "
            f"{e.baseline_composite:.3f} | {e.optimized_composite:.3f} | **{d_comp}** |"
        )

    return "\n".join(lines) + "\n"


def to_dict(summary: Union[BenchmarkMatrixSummary, AblationMatrixSummary]) -> Dict[str, Any]:
    """Serializes summary dataclasses into JSON-compatible dictionaries."""
    return asdict(summary)


def to_json(summary: Union[BenchmarkMatrixSummary, AblationMatrixSummary], indent: int = 2) -> str:
    """Serializes summary dataclasses into pretty-printed JSON string."""
    return json.dumps(to_dict(summary), indent=indent)
