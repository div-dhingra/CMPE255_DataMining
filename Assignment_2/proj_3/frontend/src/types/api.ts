/**
 * API & Data Science Type Definitions
 * Aligned with FastAPI Pydantic schemas and frontend visualization requirements.
 */

export interface DatasetHealth {
  totalRows: number;
  totalFeatures: number;
  missingValuesCount: number;
  missingValuesRate: number;
  outliersDetected: number;
  outlierRate: number;
  memoryUsageMb: number;
  hopkinsStatistic: number;
  duplicateRows: number;
  lastUpdated: string;
}

export interface FeatureSummary {
  name: string;
  dataType: string;
  mean: number;
  std: number;
  min: number;
  median: number;
  max: number;
  skewness: number;
  kurtosis: number;
  missingCount: number;
  missingRate: number;
  description: string;
  histogramBins: { bin: string; count: number }[];
}

export interface CorrelationMatrix {
  features: string[];
  pearson: number[][];
  spearman: number[][];
}

export interface CrispDmPhase {
  phaseId: number;
  code: string;
  name: string;
  status: 'completed' | 'in_progress' | 'ready';
  badge: string;
  description: string;
  kpis: { label: string; value: string }[];
  details: string[];
  actionRoute: string;
}

export interface ClusterPoint2D {
  id: string;
  x: number;
  y: number;
  cluster: number;
  balance: number;
  purchases: number;
  creditLimit: number;
  cashAdvance: number;
  payments: number;
}

export interface ClusterPoint3D extends ClusterPoint2D {
  z: number;
}

export interface ClusterPersona {
  clusterId: number;
  name: string;
  tagline: string;
  size: number;
  percentage: number;
  color: string;
  silhouetteScore: number;
  description: string;
  radarMetrics: {
    feature: string;
    value: number; // Normalized [0, 1]
    rawValue: number;
  }[];
  recommendations: string[];
  keyAttributes: { label: string; value: string }[];
}

export interface SilhouetteSample {
  cluster: number;
  score: number;
  sampleIndex: number;
}

export interface ElbowPoint {
  k: number;
  inertia: number;
  isElbow: boolean;
}

export interface ClusteringModelSummary {
  id: string;
  name: string;
  paradigm: 'Partitioning' | 'Density' | 'Hierarchical' | 'Probabilistic' | 'Autoresearch';
  k: number | string;
  silhouette: number;
  daviesBouldin: number;
  calinskiHarabasz: number;
  stabilityAri: number;
  runtimeMs: number;
  noisePoints: number;
  description: string;
}

export interface AutoresearchStep {
  step: number;
  timestamp: string;
  algorithm: string;
  hyperparameters: Record<string, string | number | boolean>;
  preprocessing: Record<string, string | number | boolean>;
  silhouetteScore: number;
  daviesBouldinIndex: number;
  calinskiHarabaszIndex: number;
  stabilityScore: number;
  compositeFitness: number;
  status: 'ACCEPTED' | 'REJECTED' | 'PERTURBED' | 'RESTART';
  mutationDescription: string;
  deltaSilhouette: number;
  temperature: number;
}

export interface AutoresearchLeaderboardItem {
  rank: number;
  trialId: number;
  algorithm: string;
  preprocessing: string;
  hyperparameters: string;
  silhouette: number;
  daviesBouldin: number;
  calinskiHarabasz: number;
  stability: number;
  compositeFitness: number;
  runtimeMs: number;
  isPareto: boolean;
}

export interface BenchmarkRow {
  algorithm: string;
  paradigm: 'Partitioning' | 'Density' | 'Hierarchical' | 'Probabilistic';
  isOptimized: boolean;
  kOrEps: string;
  silhouetteMean: number;
  silhouetteStd: number;
  daviesBouldinMean: number;
  daviesBouldinStd: number;
  calinskiHarabaszMean: number;
  calinskiHarabaszStd: number;
  stabilityScore: number;
  runtimeMs: number;
  outlierRatio: number;
}

export interface AblationEntry {
  stage: string;
  component: string;
  description: string;
  silhouette: number;
  deltaSilhouette: number;
  daviesBouldin: number;
  calinskiHarabasz: number;
  stability: number;
}

export interface LiteraturePaper {
  id: string;
  title: string;
  authors: string;
  year: number;
  venue: string;
  keyContribution: string;
  relevanceToProject: string;
  bibtex: string;
}

export interface CustomerVector {
  balance: number;
  balanceFrequency: number;
  purchases: number;
  oneOffPurchases: number;
  installmentsPurchases: number;
  cashAdvance: number;
  purchasesFrequency: number;
  oneOffPurchasesFrequency: number;
  purchasesInstallmentsFrequency: number;
  cashAdvanceFrequency: number;
  cashAdvanceTrx: number;
  purchasesTrx: number;
  creditLimit: number;
  payments: number;
  minimumPayments: number;
  prcFullPayment: number;
  tenure: number;
}

export interface InferenceRequest extends CustomerVector {
  customerId?: string;
}

export interface InferenceResponse {
  customerId?: string;
  predictedCluster: number;
  personaName: string;
  tagline: string;
  confidence: number;
  probabilities: {
    clusterId: number;
    personaName: string;
    probability: number;
    color: string;
  }[];
  radarComparison: {
    feature: string;
    sampleValue: number;
    clusterCentroid: number;
    populationMean: number;
  }[];
  topContributingFeatures: {
    feature: string;
    deviation: number;
    impact: 'Positive' | 'Negative' | 'Neutral';
  }[];
  recommendations: string[];
}

export interface SystemHealthStatus {
  online: boolean;
  latencyMs: number;
  version: string;
  backendUrl: string;
  mode: 'connected' | 'mock_fallback';
  modelCount: number;
  datasetLoaded: boolean;
}
