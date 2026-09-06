'use client';

import React, { useState, useEffect } from 'react';
import {
  Layers,
  Sparkles,
  Sliders,
  CheckCircle2,
  Lightbulb,
  Info,
  TrendingUp,
  Activity,
  Maximize2,
  PieChart,
} from 'lucide-react';
import { api } from '@/lib/api';
import {
  ClusteringModelSummary,
  ClusterPersona,
  ClusterPoint2D,
  ClusterPoint3D,
  SilhouetteSample,
  ElbowPoint,
} from '@/types/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import { StatCard } from '@/components/ui/StatCard';
import { ProjectionPlot2D } from '@/components/visualizations/ProjectionPlot2D';
import { ProjectionPlot3D } from '@/components/visualizations/ProjectionPlot3D';
import { RadarProfile } from '@/components/visualizations/RadarProfile';
import { SilhouettePlot } from '@/components/visualizations/SilhouettePlot';
import { formatNumber, formatPercent } from '@/lib/utils';

export default function ClusterExplorerPage() {
  const [models, setModels] = useState<ClusteringModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('kmeans_opt');
  const [personas, setPersonas] = useState<ClusterPersona[]>([]);
  const [points2D, setPoints2D] = useState<ClusterPoint2D[]>([]);
  const [points3D, setPoints3D] = useState<ClusterPoint3D[]>([]);
  const [silhouetteSamples, setSilhouetteSamples] = useState<SilhouetteSample[]>([]);
  const [elbowPoints, setElbowPoints] = useState<ElbowPoint[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null);
  const [projectionType, setProjectionType] = useState<'pca2d' | 'umap2d' | 'tsne2d' | 'pca3d'>('pca2d');

  useEffect(() => {
    async function loadData() {
      const [mList, pList, pts2D, pts3D, silSamples, elPoints] = await Promise.all([
        api.getClusteringModels(),
        api.getPersonas(selectedModelId),
        api.getProjectionPoints2D(selectedModelId, projectionType === 'umap2d' ? 'umap' : 'pca'),
        api.getProjectionPoints3D(selectedModelId),
        api.getSilhouetteSamples(selectedModelId),
        api.getElbowCurve(),
      ]);
      setModels(mList);
      setPersonas(pList);
      setPoints2D(pts2D);
      setPoints3D(pts3D);
      setSilhouetteSamples(silSamples);
      setElbowPoints(elPoints);
    }
    loadData();
  }, [selectedModelId, projectionType]);

  const selectedModel = models.find((m) => m.id === selectedModelId) || models[0];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Top Header & Model Selector */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900/80 p-5 rounded-2xl border border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-extrabold text-white flex items-center gap-2">
              <Layers className="w-6 h-6 text-emerald-400" />
              Multi-Model Cluster Explorer
            </h1>
            <Badge variant="emerald">6 Paradigms</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Compare partitioning, density, hierarchical, and probabilistic customer embeddings.
          </p>
        </div>

        {/* Algorithm Selector Dropdown */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-300">Model:</span>
            <select
              value={selectedModelId}
              onChange={(e: any) => {
                setSelectedModelId(e.target.value);
                setSelectedCluster(null);
              }}
              className="px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs font-semibold text-white focus:outline-none focus:border-emerald-500 transition-colors shadow-inner"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.paradigm})
                </option>
              ))}
            </select>
          </div>

          <Tabs
            tabs={[
              { id: 'pca2d', label: 'PCA 2D' },
              { id: 'umap2d', label: 'UMAP 2D' },
              { id: 'tsne2d', label: 't-SNE 2D' },
              { id: 'pca3d', label: '3D Orbital' },
            ]}
            activeTab={projectionType}
            onChange={(t) => setProjectionType(t as any)}
          />
        </div>
      </div>

      {/* Model Performance Telemetry Row */}
      {selectedModel && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <StatCard
            title="Silhouette Score"
            value={selectedModel.silhouette.toFixed(3)}
            subtitle="Separation ratio (↑)"
            badge={selectedModel.silhouette > 0.55 ? 'Optimal' : 'Sub-Optimal'}
            badgeVariant={selectedModel.silhouette > 0.55 ? 'emerald' : 'amber'}
            icon={<Sparkles className="w-4 h-4 text-emerald-400" />}
          />
          <StatCard
            title="Davies-Bouldin"
            value={selectedModel.daviesBouldin.toFixed(3)}
            subtitle="Cluster overlap (↓)"
            badge="Minimizing"
            badgeVariant="cyan"
            icon={<TrendingUp className="w-4 h-4 text-cyan-400" />}
          />
          <StatCard
            title="Calinski-Harabasz"
            value={selectedModel.calinskiHarabasz.toFixed(0)}
            subtitle="Variance ratio (↑)"
            icon={<Activity className="w-4 h-4 text-violet-400" />}
          />
          <StatCard
            title="Cluster Count"
            value={selectedModel.k.toString()}
            subtitle={selectedModel.paradigm}
            icon={<Layers className="w-4 h-4 text-amber-400" />}
          />
          <StatCard
            title="Bootstrap Stability"
            value={formatPercent(selectedModel.stabilityAri, 1)}
            subtitle="Mean ARI (100 runs)"
            badge="Robust"
            badgeVariant="emerald"
            icon={<CheckCircle2 className="w-4 h-4 text-emerald-400" />}
          />
          <StatCard
            title="Execution Latency"
            value={`${selectedModel.runtimeMs.toFixed(1)} ms`}
            subtitle="Inference time"
            icon={<Sliders className="w-4 h-4 text-slate-400" />}
          />
        </div>
      )}

      {/* Main Dual-Column Visualization Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: 2D / 3D Canvas Projection */}
        <div className="lg:col-span-7 space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Maximize2 className="w-4 h-4 text-emerald-400" />
                  Dimensionality Reduction Projection Canvas
                </h3>
                <p className="text-[11px] text-slate-400">
                  Interactive scatter embedding showing customer behavioral density clusters.
                </p>
              </div>

              {selectedCluster !== null && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedCluster(null)}
                >
                  Clear Filter (Show All)
                </Button>
              )}
            </div>

            {projectionType === 'pca3d' ? (
              <ProjectionPlot3D
                points={points3D}
                personas={personas}
                selectedCluster={selectedCluster}
              />
            ) : (
              <ProjectionPlot2D
                points={points2D}
                personas={personas}
                selectedCluster={selectedCluster}
                onSelectCluster={setSelectedCluster}
                projectionType={projectionType}
              />
            )}

            {/* Cluster Badges Selector */}
            <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800">
              <span className="text-xs text-slate-400 font-semibold mr-1">Filter Cluster:</span>
              {personas.map((p) => {
                const isSelected = selectedCluster === p.clusterId;
                return (
                  <button
                    key={p.clusterId}
                    onClick={() => setSelectedCluster(isSelected ? null : p.clusterId)}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${
                      isSelected
                        ? 'bg-slate-800 text-white border-emerald-500 shadow-md scale-105'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: p.color }}
                    />
                    <span>C{p.clusterId}: {p.name.split('/')[0].trim()}</span>
                    <span className="text-[10px] text-slate-500 font-mono">({p.percentage}%)</span>
                  </button>
                );
              })}
            </div>
          </Card>

          {/* Silhouette Ribbon Plot */}
          <SilhouettePlot
            samples={silhouetteSamples}
            personas={personas}
            averageScore={selectedModel?.silhouette ?? 0.584}
          />
        </div>

        {/* Right Column: Persona Details & Radar Profile */}
        <div className="lg:col-span-5 space-y-4">
          {/* Radar Profile Card */}
          <Card className="p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <PieChart className="w-4 h-4 text-cyan-400" />
                  Cluster Persona Behavioral Radar
                </h3>
                <p className="text-[11px] text-slate-400">
                  Normalized multi-axis comparison across 6 financial dimensions.
                </p>
              </div>
            </div>

            <RadarProfile personas={personas} selectedCluster={selectedCluster} />
          </Card>

          {/* Detailed Persona Narrative Cards */}
          <div className="space-y-3">
            {personas.map((persona) => {
              const isSelected = selectedCluster === persona.clusterId;
              return (
                <div
                  key={persona.clusterId}
                  onClick={() => setSelectedCluster(isSelected ? null : persona.clusterId)}
                  className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                    isSelected
                      ? 'bg-slate-900 border-emerald-500 shadow-lg shadow-emerald-950/30'
                      : 'bg-slate-950/70 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-3 h-3 rounded-full shrink-0"
                        style={{ backgroundColor: persona.color }}
                      />
                      <span className="font-bold text-xs text-white">{persona.name}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs font-mono">
                      <span className="text-slate-400">{persona.size.toLocaleString()} accounts</span>
                      <Badge variant="emerald">{persona.percentage}%</Badge>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 mt-2 leading-relaxed">{persona.description}</p>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs font-mono">
                    {persona.keyAttributes.map((attr, i) => (
                      <div key={i} className="bg-slate-900/90 p-1.5 rounded border border-slate-800">
                        <div className="text-[10px] text-slate-400 font-sans">{attr.label}</div>
                        <div className="font-bold text-slate-200">{attr.value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Recommendations */}
                  <div className="mt-3 pt-2 border-t border-slate-800/60">
                    <div className="flex items-center gap-1.5 text-[11px] font-semibold text-emerald-400 mb-1">
                      <Lightbulb className="w-3.5 h-3.5" />
                      <span>Business Actions & Marketing Recommendations:</span>
                    </div>
                    <ul className="space-y-1 text-[11px] text-slate-400 list-disc list-inside">
                      {persona.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
