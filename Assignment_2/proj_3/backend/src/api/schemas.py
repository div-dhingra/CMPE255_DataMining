"""
Pydantic V2 Schemas for FastAPI Analytical Backend.
Defines strict, typed data models for:
- System health & error handling
- Dataset understanding, statistical profiling, correlations, distributions, and uploads
- Clustering execution, 2D/3D projections, personas, silhouette samples, and elbow curves
- Autoresearch triggers, telemetry streams, job status, leaderboards, and ablations
- Research literature citations, algorithm trade-offs, benchmark matrix, and table exports
- Real-time single and batch customer inference
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict


# ==========================================
# 1. Generic & System Health Models
# ==========================================

class HealthResponse(BaseModel):
    """System health check and state metadata."""
    model_config = ConfigDict(extra="ignore")

    status: str = Field(default="ok", description="Service health status")
    version: str = Field(default="1.0.0", description="API version")
    dataset_loaded: bool = Field(description="Whether a dataset is currently loaded in memory")
    dataset_rows: int = Field(default=0, description="Total rows in active dataset")
    model_cached: bool = Field(description="Whether an active clustering model is fitted in memory")
    active_algorithm: Optional[str] = Field(default=None, description="Name of currently cached model")
    active_autoresearch_jobs: int = Field(default=0, description="Number of currently running autoresearch jobs")


class ErrorResponse(BaseModel):
    """Standard error response payload."""
    error: str = Field(description="Error code or category")
    detail: str = Field(description="Detailed error message")
    status_code: int = Field(description="HTTP status code")


# ==========================================
# 2. Dataset & EDA Schemas
# ==========================================

class DescriptiveStat(BaseModel):
    """Descriptive statistics for a single numerical feature."""
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q25: float
    q75: float
    iqr: float
    skewness: float
    kurtosis: float
    zero_pct: float
    missing_count: int
    missing_pct: float


class MissingColumnInfo(BaseModel):
    """Missing value information for a single column."""
    missing_count: int
    missing_pct: float


class MissingSummary(BaseModel):
    """Dataset-wide missingness audit."""
    total_missing_cells: int
    columns_with_missing: Dict[str, MissingColumnInfo] = Field(default_factory=dict)


class SkewnessEntry(BaseModel):
    """Feature skewness ranking item."""
    feature: str
    skewness: float


class DatasetSummaryResponse(BaseModel):
    """Complete dataset overview and statistical profiling response."""
    model_config = ConfigDict(extra="ignore")

    n_samples: int = Field(description="Total sample count")
    n_columns: int = Field(description="Total column count")
    columns: List[str] = Field(description="List of all column names")
    numeric_columns: List[str] = Field(description="List of numerical column names")
    descriptive_statistics: Dict[str, DescriptiveStat] = Field(description="Per-feature descriptive stats")
    missing_summary: MissingSummary = Field(description="Missing value audit")
    skewness_ranking: List[List[Union[str, float]]] = Field(description="Ordered list of [feature, skewness]")
    correlation_pearson: Dict[str, Dict[str, float]] = Field(description="Pearson correlation matrix")
    correlation_spearman: Dict[str, Dict[str, float]] = Field(description="Spearman correlation matrix")
    hopkins_statistic: float = Field(description="Hopkins clustering tendency statistic")


class CorrelationResponse(BaseModel):
    """Pearson and Spearman correlation matrices."""
    columns: List[str]
    pearson: Dict[str, Dict[str, float]]
    spearman: Dict[str, Dict[str, float]]


class HistogramBin(BaseModel):
    """Histogram bin specification."""
    bin_start: float
    bin_end: float
    count: int
    density: float


class FeatureDistribution(BaseModel):
    """Distribution metrics and histogram bins for a feature."""
    feature: str
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    skewness: float
    bins: List[HistogramBin]


class DistributionsResponse(BaseModel):
    """Distribution histograms across all numeric features."""
    features: Dict[str, FeatureDistribution]


class DataSampleResponse(BaseModel):
    """Sample preview of the active dataset."""
    total_records: int
    sample_size: int
    columns: List[str]
    records: List[Dict[str, Any]]


class DataUploadResponse(BaseModel):
    """Confirmation payload for dataset upload."""
    success: bool
    message: str
    n_samples: int
    n_columns: int
    columns: List[str]
    hopkins_statistic: float


# ==========================================
# 3. Clustering & Profiling Schemas
# ==========================================

class ClusteringPipelineConfig(BaseModel):
    """Preprocessing and feature engineering configuration."""
    imputer: str = Field(default="median", description="Imputation strategy: median, mean, knn, mice")
    outlier_method: str = Field(default="winsorize", description="Outlier treatment: winsorize, iqr, isolation_forest, none")
    scaler: str = Field(default="yeo_johnson", description="Scaler: yeo_johnson, standard, robust, minmax")
    power_transform: bool = Field(default=False, description="Apply Yeo-Johnson power transform")
    pca_components: Optional[int] = Field(default=None, description="Dimensionality reduction dimensions (2, 3, 5, 10, or None)")
    feature_engineering: bool = Field(default=True, description="Compute derived behavioral financial ratios")


class ClusteringRequest(BaseModel):
    """Request payload to train a clustering model."""
    algorithm: str = Field(default="kmeans", description="Algorithm: kmeans, kmedoids, dbscan, hdbscan, agglomerative, gmm")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model-specific hyperparameters")
    pipeline: ClusteringPipelineConfig = Field(default_factory=ClusteringPipelineConfig, description="Preprocessing configuration")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class DistinguishingFeature(BaseModel):
    """Distinguishing feature metric for a cluster persona."""
    feature: str
    cluster_mean: float
    global_mean: float
    z_score: float


class ClusterPersona(BaseModel):
    """Actionable business customer persona synthesized for a cluster."""
    cluster_id: int
    persona_name: str
    sample_count: int
    percentage: float
    business_description: str
    marketing_strategy: str
    top_distinguishing_features: List[DistinguishingFeature]
    radar_metrics: Dict[str, float]


class ClusteringResponse(BaseModel):
    """Full execution response for a clustering run."""
    algorithm: str
    n_clusters: int
    labels: List[int]
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_score: float
    stability_ari: Optional[float] = None
    inertia: Optional[float] = None
    bic: Optional[float] = None
    aic: Optional[float] = None
    noise_points_count: int
    noise_ratio: float
    cluster_sizes: Dict[str, int]
    personas: List[ClusterPersona]
    sample_projections_2d: Optional[List[List[float]]] = None
    sample_projections_3d: Optional[List[List[float]]] = None
    execution_time_ms: float


class ProjectionsResponse(BaseModel):
    """2D and 3D projection coordinates for visualization."""
    method: str = Field(description="Projection method: pca, umap, tsne")
    explained_variance_ratio: Optional[List[float]] = None
    coords_2d: List[List[float]] = Field(description="List of [x, y] coordinates")
    coords_3d: Optional[List[List[float]]] = Field(default=None, description="List of [x, y, z] coordinates")
    labels: List[int] = Field(description="Cluster assignments matching the samples")


class ClusterProfilesResponse(BaseModel):
    """Cluster personas, radar charts, ANOVA feature importance, and centroids."""
    personas: List[ClusterPersona]
    radar_profiles: Dict[str, Dict[str, float]]
    feature_importance: Dict[str, float]
    centroids: Dict[str, Dict[str, float]]


class SilhouetteSampleItem(BaseModel):
    """Individual sample silhouette value."""
    sample_index: int
    cluster_id: int
    silhouette_value: float


class SilhouetteSamplesResponse(BaseModel):
    """Per-sample silhouette ribbon distribution."""
    global_score: float
    samples: List[SilhouetteSampleItem]
    per_cluster_means: Dict[str, float]


class ElbowResponse(BaseModel):
    """WCSS Inertia curve and detected Kneedle elbow point."""
    k_values: List[int]
    inertias: List[float]
    elbow_k: int
    normalized_differences: List[float]


# ==========================================
# 4. Autoresearch & Optimization Schemas
# ==========================================

class AutoresearchObjectiveWeights(BaseModel):
    """Weights for the composite multi-objective optimization function."""
    w_silhouette: float = Field(default=0.40, ge=0.0, le=1.0)
    w_davies_bouldin: float = Field(default=0.25, ge=0.0, le=1.0)
    w_calinski_harabasz: float = Field(default=0.15, ge=0.0, le=1.0)
    w_stability: float = Field(default=0.20, ge=0.0, le=1.0)
    noise_penalty_weight: float = Field(default=0.50, ge=0.0)
    imbalance_penalty_weight: float = Field(default=0.20, ge=0.0)


class AutoresearchStartRequest(BaseModel):
    """Trigger payload to launch an autonomous hill-climbing search."""
    max_steps: int = Field(default=30, ge=1, le=500, description="Total iterations to explore")
    patience: int = Field(default=8, ge=1, le=50, description="Stagnation steps before random restart")
    target_algorithm: Optional[str] = Field(default=None, description="Fix to single algorithm (or None for cross-algorithm)")
    allow_algorithm_mutation: bool = Field(default=True, description="Allow mutating algorithm family during search")
    initial_config: Optional[Dict[str, Any]] = Field(default=None, description="Starting configuration vector")
    weights: Optional[AutoresearchObjectiveWeights] = Field(default=None, description="Objective component weights")
    initial_temperature: float = Field(default=0.5, ge=0.0, description="Initial simulated annealing temperature")
    cooling_rate: float = Field(default=0.95, ge=0.1, le=0.99, description="Geometric cooling factor")
    random_state: int = Field(default=42, description="Random seed")


class StepLogSchema(BaseModel):
    """Telemetry payload for a single hill-climbing optimization step."""
    step: int
    timestamp: str
    algorithm: str
    candidate_theta: Dict[str, Any]
    current_theta: Dict[str, Any]
    best_theta: Dict[str, Any]
    fitness: float
    delta_fitness: float
    current_fitness: float
    best_fitness: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    n_clusters: int
    noise_ratio: float
    temperature: float
    accepted: bool
    restart: bool
    stagnation_count: int
    execution_time_ms: float
    error: Optional[str] = None


class AutoresearchStartResponse(BaseModel):
    """Immediate response on launching an optimization job."""
    job_id: str
    status: str
    message: str
    max_steps: int


class AutoresearchStatusResponse(BaseModel):
    """Current job status, telemetry progress, and best candidate found."""
    job_id: str
    status: str = Field(description="Job status: running, paused, stopped, completed, error")
    current_step: int
    max_steps: int
    best_fitness: float
    best_algorithm: Optional[str] = None
    best_config: Optional[Dict[str, Any]] = None
    restart_count: int
    total_time_ms: float
    history: List[StepLogSchema]
    error: Optional[str] = None


class LeaderboardEntry(BaseModel):
    """Single evaluated configuration entry in the optimization leaderboard."""
    rank: int
    config_hash: str
    algorithm: str
    config: Dict[str, Any]
    fitness: float
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    stability_ari: float
    n_clusters: int
    noise_ratio: float
    step: int


class LeaderboardResponse(BaseModel):
    """Ranked leaderboard of the top-performing pipeline configurations."""
    total_unique_evaluated: int
    top_entries: List[LeaderboardEntry]


class AutoresearchAblationSummary(BaseModel):
    """Marginal contribution of pipeline components and baseline comparison."""
    stage_contributions: Dict[str, float]
    parameter_importance_variance: Dict[str, float]
    baseline_vs_best_comparison: Dict[str, Any]


# ==========================================
# 5. Research Literature & Benchmark Schemas
# ==========================================

class CitationSchema(BaseModel):
    """Academic citation with BibTeX and research relevance."""
    id: str
    author: str
    year: int
    title: str
    venue: str
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi_or_url: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    abstract: str = ""
    bibtex: str = ""


class AlgorithmTaxonomySchema(BaseModel):
    """Algorithmic trade-offs, theoretical assumptions, and complexity."""
    model_config = ConfigDict(extra="ignore")

    algorithm_key: Optional[str] = Field(default=None)
    name: str
    paradigm: str
    time_complexity: str
    space_complexity: str
    assumption: str
    inductive_bias: str
    hyperparameter_sensitivity: str
    robustness_to_noise: str
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    optimal_regime: str = ""


class MetricTaxonomySchema(BaseModel):
    """Validation metric mathematical definitions and bounds."""
    model_config = ConfigDict(extra="ignore")

    metric_key: Optional[str] = Field(default=None)
    name: str
    reference: Optional[str] = None
    direction: str
    bounds: Any
    formula_latex: Optional[str] = None
    formula_ascii: Optional[str] = None
    description: Optional[str] = None
    interpretation: str
    computational_complexity: Optional[str] = None


class LiteratureResponse(BaseModel):
    """Complete literature repository, taxonomies, and methodology principles."""
    citations: List[CitationSchema]
    algorithms: Dict[str, AlgorithmTaxonomySchema]
    metrics: Dict[str, MetricTaxonomySchema]
    principles: Dict[str, str]


class BenchmarkModelResultSchema(BaseModel):
    """Cross-paradigm model evaluation with bootstrap confidence bounds."""
    model_key: str
    model_name: str
    paradigm: str
    n_clusters: int
    silhouette_mean: float
    silhouette_std: float
    davies_bouldin_mean: float
    davies_bouldin_std: float
    calinski_harabasz_mean: float
    calinski_harabasz_std: float
    stability_ari_mean: float
    stability_ari_std: float
    runtime_ms: float
    noise_ratio: float
    composite_fitness: float
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkMatrixResponse(BaseModel):
    """Full cross-paradigm benchmark evaluation matrix."""
    n_samples: int
    n_features: int
    timestamp: str
    models: List[BenchmarkModelResultSchema]
    best_model_key: str
    best_fitness: float


class AblationEntrySchema(BaseModel):
    """Comparison deltas between baseline and Autoresearch-optimized model."""
    model_key: str
    model_name: str
    paradigm: str
    baseline_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    baseline_fitness: float
    optimized_fitness: float
    delta_fitness: float
    baseline_silhouette: float
    optimized_silhouette: float
    delta_silhouette: float
    baseline_davies_bouldin: float
    optimized_davies_bouldin: float
    delta_davies_bouldin: float
    baseline_calinski_harabasz: float
    optimized_calinski_harabasz: float
    delta_calinski_harabasz: float
    baseline_stability_ari: float
    optimized_stability_ari: float
    delta_stability_ari: float
    percentage_improvement: float


class AblationMatrixResponse(BaseModel):
    """Ablation matrix quantifying hill-climbing performance deltas."""
    timestamp: str
    ablations: List[AblationEntrySchema]
    overall_mean_fitness_improvement_pct: float


class TableExportRequest(BaseModel):
    """Request to generate publication-ready LaTeX or Markdown tables."""
    format: str = Field(default="latex", description="Export format: 'latex' or 'markdown'")
    table_type: str = Field(default="benchmark", description="Table type: 'benchmark' or 'ablation'")
    include_caption: bool = Field(default=True, description="Include LaTeX caption and label")


class TableExportResponse(BaseModel):
    """Exported table text content."""
    format: str
    table_type: str
    content: str


# ==========================================
# 6. Real-Time Inference Schemas
# ==========================================

class SingleInferenceRequest(BaseModel):
    """Customer financial feature vector for single-sample inference."""
    features: Dict[str, float] = Field(
        description="Feature dictionary (e.g. {'BALANCE': 1200.0, 'PURCHASES': 450.0, ...})"
    )


class BatchInferenceRequest(BaseModel):
    """Batch customer records for bulk inference."""
    customers: List[Dict[str, float]] = Field(
        description="List of feature dictionaries per customer"
    )


class InferencePersonaMatch(BaseModel):
    """Predicted persona match and business action tags."""
    cluster_id: int
    persona_name: str
    business_description: str
    marketing_strategy: str
    archetype_tags: List[str] = Field(default_factory=list)


class SingleInferenceResponse(BaseModel):
    """Prediction result for a single customer vector."""
    cluster_id: int
    probabilities: Dict[str, float] = Field(description="Soft cluster membership probabilities")
    distances_to_centroids: Optional[Dict[str, float]] = Field(default=None, description="Euclidean distances to cluster centroids")
    persona: InferencePersonaMatch
    outlier_score: Optional[float] = Field(default=None, description="Anomaly score if applicable")


class BatchInferenceResultItem(BaseModel):
    """Prediction item for a single customer in a batch."""
    index: int
    cluster_id: int
    probabilities: Dict[str, float]
    persona_name: str


class BatchInferenceResponse(BaseModel):
    """Batch inference predictions and aggregate cluster distribution."""
    total_samples: int
    predictions: List[BatchInferenceResultItem]
    cluster_counts: Dict[str, int]
