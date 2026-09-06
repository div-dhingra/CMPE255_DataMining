"""
CRISP-DM Machine Learning & Data Mining Package for Customer Segmentation.
Integrates full CRISP-DM lifecycle:
- Data Understanding (Kaggle CC loader, synthetic generator, Hopkins statistic, EDA summaries)
- Data Preparation (Imputation, Outlier Handling, Power Transformations, Ratios)
- Multi-Paradigm Modeling (K-Means, K-Medoids, DBSCAN, HDBSCAN, Agglomerative, GMM)
- Comprehensive Evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz, Kneedle Elbow, Subsampling Stability)
- Projections (PCA, UMAP, t-SNE)
- Personas & Profiling (Centroids, Radars, Feature Importance, Business Personas)
"""

from .data_understanding import (
    CREDIT_CARD_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    generate_synthetic_credit_card_data,
    load_credit_card_data,
    compute_hopkins_statistic,
    compute_data_understanding_summary,
)

from .data_preparation import (
    PipelineConfig,
    Imputer,
    OutlierHandler,
    FeatureScaler,
    FeatureEngineer,
    DataPreparationPipeline,
)

from .models import (
    ClusteringModelBase,
    KMeansModel,
    KMedoidsModel,
    DBSCANModel,
    HDBSCANModel,
    AgglomerativeModel,
    GaussianMixtureModel,
)

from .evaluation import (
    compute_silhouette_score,
    compute_davies_bouldin_index,
    compute_calinski_harabasz_score,
    compute_inertia_elbow_curve,
    compute_subsampling_stability,
    adjusted_rand_index,
    evaluate_clustering_solution,
)

from .projections import (
    compute_pca_projections,
    compute_umap_projections,
    compute_tsne_projections,
    project_coordinates,
)

from .profiling import (
    compute_cluster_centroids,
    compute_radar_profiles,
    compute_feature_importance,
    generate_cluster_personas,
)

__all__ = [
    # Data Understanding
    "CREDIT_CARD_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "generate_synthetic_credit_card_data",
    "load_credit_card_data",
    "compute_hopkins_statistic",
    "compute_data_understanding_summary",
    # Data Preparation
    "PipelineConfig",
    "Imputer",
    "OutlierHandler",
    "FeatureScaler",
    "FeatureEngineer",
    "DataPreparationPipeline",
    # Models
    "ClusteringModelBase",
    "KMeansModel",
    "KMedoidsModel",
    "DBSCANModel",
    "HDBSCANModel",
    "AgglomerativeModel",
    "GaussianMixtureModel",
    # Evaluation
    "compute_silhouette_score",
    "compute_davies_bouldin_index",
    "compute_calinski_harabasz_score",
    "compute_inertia_elbow_curve",
    "compute_subsampling_stability",
    "adjusted_rand_index",
    "evaluate_clustering_solution",
    # Projections
    "compute_pca_projections",
    "compute_umap_projections",
    "compute_tsne_projections",
    "project_coordinates",
    # Profiling
    "compute_cluster_centroids",
    "compute_radar_profiles",
    "compute_feature_importance",
    "generate_cluster_personas",
]
