"""
Research Literature Alignment & Citations Repository
Integrates formal academic citations, algorithmic trade-off taxonomies,
mathematical metric formulations, and autonomous research methodology principles.

Citations include:
- Rousseeuw (1987) [Silhouette Coefficient]
- Davies & Bouldin (1979) [Davies-Bouldin Index]
- Calinski & Harabasz (1974) [Variance Ratio Criterion]
- Arthur & Vassilvitskii (2007) [k-means++]
- Campello et al. (2013) [HDBSCAN]
- McInnes et al. (2018) [UMAP]
- Von Luxburg (2010) [Clustering Stability]
- Satopaa et al. (2011) [Kneedle Algorithm]
- Sakana AI / Lu et al. (2024) [The AI Scientist]
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Citation:
    """Academic Citation entry with BibTeX and metadata."""
    id: str
    author: str
    year: int
    title: str
    venue: str
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi_or_url: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    abstract: str = ""
    bibtex: str = ""


@dataclass
class AlgorithmTaxonomyEntry:
    """Algorithmic trade-off and complexity taxonomy."""
    algorithm_key: str
    name: str
    paradigm: str
    time_complexity: str
    space_complexity: str
    assumption: str
    inductive_bias: str
    hyperparameter_sensitivity: str
    robustness_to_noise: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    optimal_regime: str = ""


@dataclass
class MetricTaxonomyEntry:
    """Internal and external clustering validation metric taxonomy."""
    metric_key: str
    name: str
    reference: str
    direction: str  # 'maximize', 'minimize', 'elbow_knee'
    bounds: List[float]  # e.g. [-1.0, 1.0] or [0.0, float('inf')]
    formula_latex: str
    formula_ascii: str
    description: str
    interpretation: str
    computational_complexity: str


# ============================================================================
# 1. Authoritative Literature Repository
# ============================================================================

LITERATURE_REPOSITORY: List[Citation] = [
    Citation(
        id="rousseeuw1987",
        author="Rousseeuw",
        year=1987,
        title="Silhouettes: A graphical aid to the interpretation and validation of cluster analysis",
        venue="Journal of Computational and Applied Mathematics",
        volume="20",
        pages="53--65",
        doi_or_url="https://doi.org/10.1016/0377-0427(87)90125-7",
        topics=["validation", "silhouette", "interpretation", "cluster_quality"],
        abstract=(
            "Presents a graphical display technique ('silhouettes') to evaluate the quality of clusterings. "
            "For each object, a silhouette value s(i) is calculated representing how well-matched the object is "
            "to its cluster compared to neighboring clusters. The global silhouette width serves as an intrinsic "
            "validation index for selecting optimal cluster counts without ground-truth labels."
        ),
        bibtex="""@article{rousseeuw1987silhouettes,
  title={Silhouettes: a graphical aid to the interpretation and validation of cluster analysis},
  author={Rousseeuw, Peter J},
  journal={Journal of Computational and Applied Mathematics},
  volume={20},
  pages={53--65},
  year={1987},
  publisher={Elsevier}
}""",
    ),
    Citation(
        id="daviesbouldin1979",
        author="Davies & Bouldin",
        year=1979,
        title="A Cluster Separation Measure",
        venue="IEEE Transactions on Pattern Analysis and Machine Intelligence",
        volume="PAMI-1",
        issue="2",
        pages="224--227",
        doi_or_url="https://doi.org/10.1109/TPAMI.1979.4766909",
        topics=["validation", "davies_bouldin", "separation", "dispersion"],
        abstract=(
            "Introduces a non-parametric cluster separation measure based on the ratio of the sum of within-cluster "
            "scatters to between-cluster separation across all cluster pairs. Lower values signify compact, "
            "well-separated clusters. The measure is computationally efficient and widely applied for model selection."
        ),
        bibtex="""@article{davies1979cluster,
  title={A cluster separation measure},
  author={Davies, David L and Bouldin, Donald W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={PAMI-1},
  number={2},
  pages={224--227},
  year={1979},
  publisher={IEEE}
}""",
    ),
    Citation(
        id="calinskiharabasz1974",
        author="Calinski & Harabasz",
        year=1974,
        title="A dendrite method for cluster analysis",
        venue="Communications in Statistics",
        volume="3",
        issue="1",
        pages="1--27",
        doi_or_url="https://doi.org/10.1080/03610927408827101",
        topics=["validation", "calinski_harabasz", "variance_ratio", "anova"],
        abstract=(
            "Proposes the Variance Ratio Criterion (VRC), now known as the Calinski-Harabasz index. It evaluates "
            "the ratio of the between-cluster dispersion matrix trace to the within-cluster dispersion matrix trace, "
            "analogous to an F-statistic in ANOVA. Higher values indicate denser, better-separated clusters."
        ),
        bibtex="""@article{calinski1974dendrite,
  title={A dendrite method for cluster analysis},
  author={Cali{\'n}ski, Tadeusz and Harabasz, Jerzy},
  journal={Communications in Statistics-theory and Methods},
  volume={3},
  number={1},
  pages={1--27},
  year={1974},
  publisher={Taylor \\& Francis}
}""",
    ),
    Citation(
        id="arthurvassilvitskii2007",
        author="Arthur & Vassilvitskii",
        year=2007,
        title="k-means++: The Advantages of Careful Seeding",
        venue="Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA '07)",
        pages="1027--1035",
        doi_or_url="https://doi.org/10.5555/1283383.1283494",
        topics=["partitioning", "kmeans", "seeding", "approximation_algorithms"],
        abstract=(
            "Formulates the k-means++ randomized seeding algorithm which selects initial cluster centers with "
            "probability proportional to their squared distance to the nearest existing center. Guarantees an "
            "expected approximation ratio of O(log k) relative to the optimal clustering and achieves faster convergence."
        ),
        bibtex="""@inproceedings{arthur2007k,
  title={k-means++: The advantages of careful seeding},
  author={Arthur, David and Vassilvitskii, Sergei},
  booktitle={Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms},
  pages={1027--1035},
  year={2007}
}""",
    ),
    Citation(
        id="campello2013",
        author="Campello et al.",
        year=2013,
        title="Density-Based Clustering Based on Hierarchical Density Estimates",
        venue="Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)",
        pages="160--172",
        doi_or_url="https://doi.org/10.1007/978-3-642-37456-2_14",
        topics=["density", "hdbscan", "hierarchical_density", "variable_density"],
        abstract=(
            "Introduces HDBSCAN, which transforms data into a mutual reachability space, constructs a minimum "
            "spanning tree, converts it into a condensed cluster tree, and extracts stable clusters based on "
            "excess of mass (stability). Handles variable density and automatically identifies noise."
        ),
        bibtex="""@inproceedings{campello2013density,
  title={Density-based clustering based on hierarchical density estimates},
  author={Campello, Ricardo JGB and Moulavi, Davoud and Zimek, Arthur},
  booktitle={Pacific-Asia conference on knowledge discovery and data mining},
  pages={160--172},
  year={2013},
  organization={Springer}
}""",
    ),
    Citation(
        id="mcinnes2018",
        author="McInnes et al.",
        year=2018,
        title="UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction",
        venue="Journal of Open Source Software / arXiv:1802.03426",
        pages="1--63",
        doi_or_url="https://doi.org/10.48550/arXiv.1802.03426",
        topics=["dimensionality_reduction", "manifold_learning", "umap", "projections"],
        abstract=(
            "Presents Uniform Manifold Approximation and Projection (UMAP), a non-linear dimension reduction "
            "technique grounded in Riemannian geometry and algebraic topology. Preserves both local and global "
            "topological structure while offering scalable runtime performance compared to t-SNE."
        ),
        bibtex="""@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}""",
    ),
    Citation(
        id="vonluxburg2010",
        author="Von Luxburg",
        year=2010,
        title="Clustering Stability: An Overview",
        venue="Foundations and Trends in Machine Learning",
        volume="2",
        issue="3",
        pages="235--274",
        doi_or_url="https://doi.org/10.1561/2200000008",
        topics=["validation", "stability", "bootstrap", "model_selection"],
        abstract=(
            "Provides an in-depth theoretical overview of clustering stability as a model selection and validation "
            "principle. Demonstrates how stability under subsampling perturbations measures algorithm robustness and "
            "governs optimal parameter selection (e.g. k) across different data geometries."
        ),
        bibtex="""@article{von2010clustering,
  title={Clustering stability: An overview},
  author={Von Luxburg, Ulrike},
  journal={Foundations and Trends{\\registered} in Machine Learning},
  volume={2},
  number={3},
  pages={235--274},
  year={2010},
  publisher={Now Publishers, Inc.}
}""",
    ),
    Citation(
        id="satopaa2011",
        author="Satopaa et al.",
        year=2011,
        title="Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System Behavior",
        venue="IEEE 31st International Conference on Distributed Computing Systems Workshops (ICDCSW)",
        pages="166--171",
        doi_or_url="https://doi.org/10.1109/ICDCSW.2011.20",
        topics=["model_selection", "elbow", "kneedle", "curvature"],
        abstract=(
            "Formulates the Kneedle algorithm for detecting knee/elbow points in discrete curves based on mathematical "
            "curvature in normalized coordinate space. Applied to within-cluster sum of squares (inertia) curves to "
            "determine the optimal number of clusters k without subjective visual inspection."
        ),
        bibtex="""@inproceedings{satopaa2011finding,
  title={Finding a" kneedle" in a haystack: Detecting knee points in system behavior},
  author={Satopaa, Ville and Albrecht, Jeannie and Irwin, David and Raghavan, Barath},
  booktitle={2011 31st international conference on distributed computing systems workshops},
  pages={166--171},
  year={2011},
  organization={IEEE}
}""",
    ),
    Citation(
        id="sakana2024",
        author="Sakana AI",
        year=2024,
        title="The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
        venue="arXiv preprint arXiv:2408.06292",
        pages="1--45",
        doi_or_url="https://doi.org/10.48550/arXiv.2408.06292",
        topics=["autoresearch", "automl", "metaheuristic", "scientific_discovery"],
        abstract=(
            "Introduces the first comprehensive framework for automated scientific discovery ('The AI Scientist'). "
            "Demonstrates autonomous hypothesis generation, iterative code execution, metric evaluation, "
            "systematic ablations, and paper generation, inspiring the autonomous clustering autoresearch loop."
        ),
        bibtex="""@article{lu2024aiscientist,
  title={The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}""",
    ),
]


# ============================================================================
# 2. Algorithm Taxonomy & Trade-off Specifications
# ============================================================================

ALGORITHM_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "kmeans": {
        "name": "K-Means++",
        "paradigm": "Partitioning",
        "time_complexity": "O(k*n*d)",
        "space_complexity": "O((n + k) * d)",
        "assumption": "Spherical / equal variance",
        "inductive_bias": "Voronoi convex polyhedra, isotropic variance, Euclidean metric",
        "hyperparameter_sensitivity": "Moderate (k selection via Elbow/Silhouette; sensitive to outlier distortion)",
        "robustness_to_noise": "Low (forces all points into clusters, pull centroids towards outliers)",
        "pros": [
            "Linear time complexity O(k*n*d) scalable to massive datasets",
            "k-means++ seeding guarantees O(log k) competitive ratio",
            "Easily interpretable spherical centroid coordinates",
            "Guaranteed convergence via Lloyd's alternating minimization"
        ],
        "cons": [
            "Fails on arbitrary non-convex or manifold geometries",
            "Fixed k must be specified a priori",
            "Heavily distorted by extreme financial power-law outliers"
        ],
        "optimal_regime": "Large-scale isotropic spherical clusters with standardized/power-transformed features."
    },
    "kmedoids": {
        "name": "K-Medoids (FasterPAM)",
        "paradigm": "Partitioning",
        "time_complexity": "O(k*(n-k))",
        "space_complexity": "O(n*d)",
        "assumption": "Exemplar data point representability, arbitrary metrics",
        "inductive_bias": "Centroid must be an observed data point (medoid), L1/L2 metric flexibility",
        "hyperparameter_sensitivity": "Moderate (k parameter, metric choice)",
        "robustness_to_noise": "High (medoids are robust against extreme outlier pull)",
        "pros": [
            "Direct interpretability: clusters centered at actual customer records",
            "Supports Manhattan (L1) metric robust to heavy-tailed distributions",
            "FasterPAM optimization eliminates O(n^2) distance caching bottleneck"
        ],
        "cons": [
            "Higher computational latency than K-Means",
            "Still assumes roughly equal-sized convex partitions"
        ],
        "optimal_regime": "Customer segmentation where real customer exemplars are required and financial skew is severe."
    },
    "dbscan": {
        "name": "DBSCAN",
        "paradigm": "Density-Based",
        "time_complexity": "O(n*log(n))",
        "space_complexity": "O(n)",
        "assumption": "Uniform density",
        "inductive_bias": "Clusters are contiguous regions of high point density separated by low-density noise",
        "hyperparameter_sensitivity": "High (eps and min_samples must be precisely calibrated)",
        "robustness_to_noise": "High (explicit noise label -1 isolates outliers)",
        "pros": [
            "Discovers arbitrary non-linear shapes and geometric contours",
            "No predefined cluster count required",
            "Explicit noise isolation prevents outlier contamination"
        ],
        "cons": [
            "Fails when clusters have substantially different local densities",
            "Curse of dimensionality causes distance concentration in high D"
        ],
        "optimal_regime": "Low-to-moderate dimensional data with uniform density and significant noise/outliers."
    },
    "hdbscan": {
        "name": "HDBSCAN",
        "paradigm": "Density-Based",
        "time_complexity": "O(n*log(n))",
        "space_complexity": "O(n)",
        "assumption": "Variable density",
        "inductive_bias": "Clusters correspond to prominent peaks in the mutual reachability density hierarchy",
        "hyperparameter_sensitivity": "Low-to-Moderate (governed primarily by min_cluster_size)",
        "robustness_to_noise": "Very High (GLOSH outlier scoring and soft cluster probabilities)",
        "pros": [
            "Overcomes DBSCAN's uniform density limitation via mutual reachability graph",
            "Produces continuous outlier scores (GLOSH) and soft membership probabilities",
            "Extracts globally optimal flat clusters using Excess of Mass (EOM)"
        ],
        "cons": [
            "Can classify borderline samples as noise if min_cluster_size is too high",
            "More memory-intensive than standard K-Means"
        ],
        "optimal_regime": "Complex behavioral data with multi-scale densities and distinct transactional outlier populations."
    },
    "agglomerative": {
        "name": "Agglomerative Hierarchical",
        "paradigm": "Hierarchical",
        "time_complexity": "O(n^2*log(n))",
        "space_complexity": "O(n^2)",
        "assumption": "Hierarchical cluster nesting (Ward assumes minimum variance)",
        "inductive_bias": "Bottom-up greedy tree linkage merge criterion (Ward, Complete, Average)",
        "hyperparameter_sensitivity": "Moderate (linkage metric, distance cutoff / k)",
        "robustness_to_noise": "Moderate (Ward is robust; Single linkage suffers from chaining)",
        "pros": [
            "Produces rich hierarchical dendrogram tree of customer relationships",
            "Ward linkage minimizes total within-cluster variance effectively",
            "Deterministic output without random initialization instability"
        ],
        "cons": [
            "Quadratic O(n^2) space and O(n^2 log n) time complexity limits scalability to >20k samples",
            "Greedy merges cannot be revised downstream"
        ],
        "optimal_regime": "Medium-sized customer cohorts where nested sub-segment hierarchies are required."
    },
    "gmm": {
        "name": "Gaussian Mixture Models (EM)",
        "paradigm": "Probabilistic",
        "time_complexity": "O(k*n*d^2)",
        "space_complexity": "O(k*d^2 + n*k)",
        "assumption": "Multivariate Gaussian mixtures",
        "inductive_bias": "Data generated from K multivariate Gaussian distributions with covariance Sigma_k",
        "hyperparameter_sensitivity": "Moderate (k components, covariance type: full/tied/diag/spherical)",
        "robustness_to_noise": "Moderate (soft responsibilities accommodate overlapping boundaries)",
        "pros": [
            "Rigorous probabilistic foundation providing soft posterior membership probabilities",
            "Models complex elliptical cluster orientations and correlations via full covariance",
            "Principled model selection criteria (BIC, AIC, Log-Likelihood)"
        ],
        "cons": [
            "EM algorithm can converge to local maxima (requires multiple initializations)",
            "Covariance matrices can become ill-conditioned without regularization (reg_covar)"
        ],
        "optimal_regime": "Continuous behavioral dimensions with soft customer boundaries and elliptical covariances."
    }
}


# ============================================================================
# 3. Validation Metric Taxonomy
# ============================================================================

METRIC_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "silhouette": {
        "name": "Silhouette Coefficient",
        "reference": "Rousseeuw (1987)",
        "direction": "maximize",
        "bounds": [-1.0, 1.0],
        "formula_latex": r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad S = \frac{1}{N}\sum_{i=1}^N s(i)",
        "formula_ascii": "s(i) = (b(i) - a(i)) / max(a(i), b(i)); S = mean(s(i))",
        "description": "Measures intra-cluster cohesion a(i) versus nearest neighbor inter-cluster separation b(i).",
        "interpretation": "Values near +1 indicate dense, well-separated clusters; values near 0 indicate overlapping clusters; negative values indicate misassignments.",
        "computational_complexity": "O(n^2 * d) pairwise distances (subsampling recommended for N > 10,000)"
    },
    "davies_bouldin": {
        "name": "Davies-Bouldin Index",
        "reference": "Davies & Bouldin (1979)",
        "direction": "minimize",
        "bounds": [0.0, float("inf")],
        "formula_latex": r"DB = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i} \left(\frac{s_i + s_j}{d(\mu_i, \mu_j)}\right)",
        "formula_ascii": "DB = (1/k) * sum_i max_{j!=i} [(s_i + s_j) / d(mu_i, mu_j)]",
        "description": "Calculates the maximum similarity ratio of within-cluster dispersion (s_i + s_j) to centroid distance d(mu_i, mu_j).",
        "interpretation": "Lower values (closer to 0.0) indicate superior clustering with compact, widely separated clusters.",
        "computational_complexity": "O(n * d + k^2 * d), extremely fast to evaluate"
    },
    "calinski_harabasz": {
        "name": "Calinski-Harabasz Index (Variance Ratio Criterion)",
        "reference": "Calinski & Harabasz (1974)",
        "direction": "maximize",
        "bounds": [0.0, float("inf")],
        "formula_latex": r"CH = \frac{\text{Tr}(B_k)/(k-1)}{\text{Tr}(W_k)/(n-k)}",
        "formula_ascii": "CH = [Tr(B_k) / (k - 1)] / [Tr(W_k) / (n - k)]",
        "description": "Ratio of between-cluster dispersion (Tr(B_k)) to within-cluster dispersion (Tr(W_k)), weighted by degrees of freedom.",
        "interpretation": "Higher values indicate tighter clusters with greater separation between centroids.",
        "computational_complexity": "O(n * d + k * d), linear in sample count and very fast"
    },
    "cluster_stability": {
        "name": "Subsampling Bootstrap Stability (ARI)",
        "reference": "Von Luxburg (2010)",
        "direction": "maximize",
        "bounds": [0.0, 1.0],
        "formula_latex": r"\text{Stability} = \frac{1}{B}\sum_{b=1}^B \text{ARI}(Y_{\text{orig}|S_b}, Y_{\text{boot}, b})",
        "formula_ascii": "Stability = (1/B) * sum_{b=1}^B ARI(Y_orig[S_b], Y_boot[S_b])",
        "description": "Measures the reproducibility of clustering assignments under B bootstrap subsampling perturbations evaluated via Adjusted Rand Index.",
        "interpretation": "Values > 0.85 indicate robust, generalizable clusters; values < 0.70 suggest high sensitivity to sample variance.",
        "computational_complexity": "O(B * Cost(Model))"
    },
    "inertia_elbow": {
        "name": "Within-Cluster Sum of Squares (Inertia) & Kneedle Elbow",
        "reference": "Satopaa et al. (2011)",
        "direction": "elbow_knee",
        "bounds": [0.0, float("inf")],
        "formula_latex": r"\text{Inertia}(k) = \sum_{j=1}^k \sum_{x \in C_j} \|x - \mu_j\|^2, \quad D(k) = y_{\text{norm}}(k) - x_{\text{norm}}(k)",
        "formula_ascii": "Inertia(k) = sum_j sum_{x in C_j} ||x - mu_j||^2; Knee = argmax D(k)",
        "description": "Total within-cluster squared distance across k in [2, 15], with automated knee point detection via Kneedle difference curvature.",
        "interpretation": "Identifies the optimal k where marginal reduction in WCSS diminishes.",
        "computational_complexity": "O(K_max * n * d)"
    },
    "hopkins_statistic": {
        "name": "Hopkins Statistic (Clustering Tendency)",
        "reference": "Hopkins & Skellam (1954)",
        "direction": "maximize",
        "bounds": [0.0, 1.0],
        "formula_latex": r"H = \frac{\sum_{i=1}^m u_i^d}{\sum_{i=1}^m u_i^d + \sum_{i=1}^m w_i^d}",
        "formula_ascii": "H = sum(u_i^d) / (sum(u_i^d) + sum(w_i^d))",
        "description": "Compares nearest neighbor distances of synthetic uniform sample points (u_i) against actual dataset sample points (w_i).",
        "interpretation": "H > 0.75 confirms strong spatial clustering tendency; H ~ 0.5 indicates a uniform/random spatial distribution lacking clusters.",
        "computational_complexity": "O(m * n * d) where m is subsample size"
    }
}


# ============================================================================
# 4. Autoresearch Principles
# ============================================================================

AUTORESEARCH_PRINCIPLES: List[Dict[str, str]] = [
    {
        "id": "discrete_neighborhood_mutation",
        "name": "Discrete Neighborhood Mutation",
        "description": "Mutates categorical pipeline stages (e.g. Scaler: standard -> yeo_johnson; Outlier: none -> winsorize) using single-step graph transitions.",
        "rationale": "Enables granular exploration of the preprocessing discrete topological lattice without unbounded combinatorial jumps."
    },
    {
        "id": "continuous_parameter_perturbation",
        "name": "Continuous Parameter Perturbation",
        "description": "Applies Gaussian drift (+/- 10% to 20%) or adaptive step scales to continuous hyperparameters (e.g. DBSCAN eps, contamination, reg_covar).",
        "rationale": "Locally refines hyperparameter boundaries around promising basins of attraction."
    },
    {
        "id": "composite_objective_evaluation",
        "name": "Composite Multi-Objective Evaluation",
        "description": "Evaluates candidate pipelines using a bounded, normalized scalar fitness F(theta) balancing Silhouette, DB, CH, and Stability with penalty terms.",
        "rationale": "Prevents single-metric gaming (e.g. CH favoring tiny spherical clusters or Silhouette favoring k=2)."
    },
    {
        "id": "simulated_annealing_acceptance",
        "name": "Simulated Annealing Acceptance Criteria",
        "description": "Accepts strictly improving mutations (Delta F > 0) unconditionally, and accepts inferior steps with probability P = exp(Delta F / T).",
        "rationale": "Allows the optimizer to escape shallow local optima during early exploration phases while converging asymptotically as temperature cools."
    },
    {
        "id": "random_restarts_on_plateau",
        "name": "Random Restarts on Plateau Stagnation",
        "description": "Upon exhausting patience limit M with zero objective gain, resets search state by sampling a uniform random point in configuration space Theta.",
        "rationale": "Guarantees global exploration of disparate algorithm paradigms and distant preprocessing subspaces."
    },
    {
        "id": "ablation_isolation",
        "name": "Systematic Component-Wise Ablation",
        "description": "Isolates the individual marginal contributions of preprocessing transforms, feature engineering, and hyperparameter tuning against baseline models.",
        "rationale": "Provides empirical justification and scientific attribution for each pipeline decision."
    }
]


# ============================================================================
# 5. Retrieval & Query Functions
# ============================================================================

def get_literature_repository() -> List[Dict[str, Any]]:
    """Returns all literature citations as structured dictionaries."""
    return [asdict(c) for c in LITERATURE_REPOSITORY]


def get_citations(topic: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves citations, optionally filtered by topic tag.
    
    Args:
        topic: Tag string (e.g. 'validation', 'density', 'partitioning', 'autoresearch').
        
    Returns:
        List of matching citation dictionaries.
    """
    if topic is None:
        return [asdict(c) for c in LITERATURE_REPOSITORY]
    topic_clean = topic.strip().lower()
    return [
        asdict(c) for c in LITERATURE_REPOSITORY
        if any(topic_clean in t.lower() for t in c.topics)
    ]


def get_citation_by_id(citation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a specific citation by ID."""
    cid = citation_id.strip().lower()
    for c in LITERATURE_REPOSITORY:
        if c.id.lower() == cid:
            return asdict(c)
    return None


def get_bibtex(citation_id: Optional[str] = None) -> str:
    """
    Returns BibTeX string representation.
    If citation_id is provided, returns that citation's BibTeX;
    otherwise returns a concatenated multi-entry BibTeX bibliography.
    """
    if citation_id is not None:
        c = get_citation_by_id(citation_id)
        return c["bibtex"] if c else ""
    return "\n\n".join(c.bibtex for c in LITERATURE_REPOSITORY)


def get_algorithm_taxonomy(algorithm_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves algorithm trade-off taxonomy.
    If algorithm_key is provided, returns taxonomy entry for that algorithm;
    otherwise returns full taxonomy dictionary.
    """
    if algorithm_key is None:
        return ALGORITHM_TAXONOMY
    key = algorithm_key.strip().lower()
    return ALGORITHM_TAXONOMY.get(key, {})


def get_metric_formulations(metric_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves metric formulations and mathematical definitions.
    If metric_key is provided, returns formulation for that metric;
    otherwise returns full metric taxonomy dictionary.
    """
    if metric_key is None:
        return METRIC_TAXONOMY
    key = metric_key.strip().lower()
    return METRIC_TAXONOMY.get(key, {})


def get_autoresearch_principles() -> List[Dict[str, str]]:
    """Returns the list of core autoresearch methodology principles."""
    return list(AUTORESEARCH_PRINCIPLES)
