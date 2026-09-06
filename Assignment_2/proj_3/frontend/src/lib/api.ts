import {
  DatasetHealth,
  CrispDmPhase,
  FeatureSummary,
  CorrelationMatrix,
  ClusteringModelSummary,
  ClusterPersona,
  ClusterPoint2D,
  ClusterPoint3D,
  SilhouetteSample,
  ElbowPoint,
  AutoresearchStep,
  AutoresearchLeaderboardItem,
  BenchmarkRow,
  AblationEntry,
  LiteraturePaper,
  CustomerVector,
  InferenceResponse,
  SystemHealthStatus,
} from '../types/api';
import {
  mockDatasetHealth,
  mockCrispDmPhases,
  mockFeatureSummaries,
  mockCorrelationMatrix,
  mockClusteringModels,
  mockPersonas,
  mockPoints2D,
  mockPoints3D,
  mockSilhouetteSamples,
  mockElbowPoints,
  mockAutoresearchHistory,
  mockLeaderboard,
  mockBenchmarkRows,
  mockAblationEntries,
  mockLiteraturePapers,
  simulateCustomerInference,
} from './mockData';

declare const globalThis: any;

const BASE_URL =
  (typeof globalThis !== 'undefined' &&
    globalThis.process &&
    globalThis.process.env &&
    globalThis.process.env.NEXT_PUBLIC_API_BASE_URL) ||
  'http://localhost:8000/api/v1';

class ApiClient {
  private isOnline: boolean = false;
  private lastHealthCheck: number = 0;

  private async fetchWithTimeout(url: string, options: any = {}, timeoutMs = 1500): Promise<any> {
    if (typeof AbortController === 'undefined' || typeof fetch === 'undefined') {
      throw new Error('Fetch not available');
    }
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...(options.headers || {}),
        },
      });
      clearTimeout(id);
      return response;
    } catch (err) {
      clearTimeout(id);
      throw err;
    }
  }

  public async checkHealth(): Promise<SystemHealthStatus> {
    const startTime = Date.now();
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/health`, { method: 'GET' }, 1200);
      if (res.ok) {
        const data = await res.json();
        this.isOnline = true;
        this.lastHealthCheck = Date.now();
        return {
          online: true,
          latencyMs: Date.now() - startTime,
          version: data.version || '1.0.0',
          backendUrl: BASE_URL,
          mode: 'connected',
          modelCount: data.models_count || 6,
          datasetLoaded: true,
        };
      }
    } catch (e) {
      // Backend offline or unreachable
    }

    this.isOnline = false;
    this.lastHealthCheck = Date.now();
    return {
      online: false,
      latencyMs: Date.now() - startTime,
      version: '1.0.0-mock',
      backendUrl: BASE_URL,
      mode: 'mock_fallback',
      modelCount: 6,
      datasetLoaded: true,
    };
  }

  public async getDatasetHealth(): Promise<DatasetHealth> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/dataset/summary`);
      if (res.ok) {
        const data = await res.json();
        return {
          totalRows: data.total_rows || mockDatasetHealth.totalRows,
          totalFeatures: data.total_features || mockDatasetHealth.totalFeatures,
          missingValuesCount: data.missing_values_count ?? mockDatasetHealth.missingValuesCount,
          missingValuesRate: data.missing_values_rate ?? mockDatasetHealth.missingValuesRate,
          outliersDetected: data.outliers_detected ?? mockDatasetHealth.outliersDetected,
          outlierRate: data.outlier_rate ?? mockDatasetHealth.outlierRate,
          memoryUsageMb: data.memory_usage_mb ?? mockDatasetHealth.memoryUsageMb,
          hopkinsStatistic: data.hopkins_statistic ?? mockDatasetHealth.hopkinsStatistic,
          duplicateRows: data.duplicate_rows ?? mockDatasetHealth.duplicateRows,
          lastUpdated: data.last_updated || mockDatasetHealth.lastUpdated,
        };
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockDatasetHealth;
  }

  public async getCrispDmPhases(): Promise<CrispDmPhase[]> {
    return mockCrispDmPhases;
  }

  public async getFeatureSummaries(): Promise<FeatureSummary[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/dataset/distributions`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockFeatureSummaries;
  }

  public async getCorrelationMatrix(): Promise<CorrelationMatrix> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/dataset/correlation`);
      if (res.ok) {
        const data = await res.json();
        if (data.features && data.pearson) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockCorrelationMatrix;
  }

  public async getClusteringModels(): Promise<ClusteringModelSummary[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/clustering/models`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockClusteringModels;
  }

  public async getPersonas(modelId?: string): Promise<ClusterPersona[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/clustering/profiles?model=${modelId || 'kmeans_opt'}`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockPersonas;
  }

  public async getProjectionPoints2D(modelId?: string, projectionType: string = 'pca'): Promise<ClusterPoint2D[]> {
    try {
      const res = await this.fetchWithTimeout(
        `${BASE_URL}/clustering/projections?model=${modelId || 'kmeans_opt'}&type=${projectionType}&dim=2`
      );
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.points) && data.points.length > 0) return data.points;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockPoints2D;
  }

  public async getProjectionPoints3D(modelId?: string): Promise<ClusterPoint3D[]> {
    try {
      const res = await this.fetchWithTimeout(
        `${BASE_URL}/clustering/projections?model=${modelId || 'kmeans_opt'}&type=pca&dim=3`
      );
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data.points) && data.points.length > 0) return data.points;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockPoints3D;
  }

  public async getSilhouetteSamples(modelId?: string): Promise<SilhouetteSample[]> {
    try {
      const res = await this.fetchWithTimeout(
        `${BASE_URL}/clustering/silhouette-samples?model=${modelId || 'kmeans_opt'}`
      );
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockSilhouetteSamples;
  }

  public async getElbowCurve(): Promise<ElbowPoint[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/clustering/elbow`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockElbowPoints;
  }

  public async getAutoresearchHistory(): Promise<AutoresearchStep[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/autoresearch/history`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockAutoresearchHistory;
  }

  public async getAutoresearchLeaderboard(): Promise<AutoresearchLeaderboardItem[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/autoresearch/leaderboard`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockLeaderboard;
  }

  public async getBenchmarkMatrix(): Promise<BenchmarkRow[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/benchmark/matrix`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockBenchmarkRows;
  }

  public async getAblationEntries(): Promise<AblationEntry[]> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/benchmark/ablation`);
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data) && data.length > 0) return data;
      }
    } catch (err) {
      // Graceful fallback
    }
    return mockAblationEntries;
  }

  public async getLiteraturePapers(): Promise<LiteraturePaper[]> {
    return mockLiteraturePapers;
  }

  public async predictCustomer(input: CustomerVector): Promise<InferenceResponse> {
    try {
      const res = await this.fetchWithTimeout(`${BASE_URL}/inference/predict`, {
        method: 'POST',
        body: JSON.stringify(input),
      });
      if (res.ok) {
        const data = await res.json();
        return data;
      }
    } catch (err) {
      // Graceful fallback to client-side simulator
    }
    return simulateCustomerInference(input);
  }
}

export const api = new ApiClient();
