'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  Activity,
  CheckCircle2,
  Database,
  ArrowRight,
  TrendingUp,
  AlertTriangle,
  FileSpreadsheet,
  Search,
  Sparkles,
  BarChart2,
  Layers,
  Cpu,
  SlidersHorizontal,
} from 'lucide-react';
import { api } from '@/lib/api';
import {
  DatasetHealth,
  CrispDmPhase,
  FeatureSummary,
  CorrelationMatrix,
} from '@/types/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { StatCard } from '@/components/ui/StatCard';
import { Tabs } from '@/components/ui/Tabs';
import { formatNumber, formatPercent } from '@/lib/utils';

export default function OverviewPage() {
  const [health, setHealth] = useState<DatasetHealth | null>(null);
  const [phases, setPhases] = useState<CrispDmPhase[]>([]);
  const [features, setFeatures] = useState<FeatureSummary[]>([]);
  const [correlation, setCorrelation] = useState<CorrelationMatrix | null>(null);
  const [selectedPhase, setSelectedPhase] = useState<number>(1);
  const [searchFeature, setSearchFeature] = useState('');
  const [corrType, setCorrType] = useState<'pearson' | 'spearman'>('pearson');
  const [hoveredCorr, setHoveredCorr] = useState<{ f1: string; f2: string; val: number } | null>(null);

  useEffect(() => {
    async function loadData() {
      const [h, p, f, c] = await Promise.all([
        api.getDatasetHealth(),
        api.getCrispDmPhases(),
        api.getFeatureSummaries(),
        api.getCorrelationMatrix(),
      ]);
      setHealth(h);
      setPhases(p);
      setFeatures(f);
      setCorrelation(c);
    }
    loadData();
  }, []);

  const filteredFeatures = features.filter(
    (f) =>
      f.name.toLowerCase().includes(searchFeature.toLowerCase()) ||
      f.description.toLowerCase().includes(searchFeature.toLowerCase())
  );

  const activePhaseData = phases.find((p) => p.phaseId === selectedPhase) || phases[0];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Hero Header */}
      <div className="relative rounded-2xl p-6 sm:p-8 overflow-hidden bg-gradient-to-r from-slate-900 via-slate-900/90 to-emerald-950/40 border border-slate-800 shadow-xl">
        <div className="absolute -right-10 -bottom-10 w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-4xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-700/50 text-xs font-semibold text-emerald-300">
            <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
            CRISP-DM Standard Data Mining Lifecycle
          </div>
          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight text-white">
            Autonomous Customer Clustering & Autoresearch Studio
          </h1>
          <p className="text-sm sm:text-base text-slate-300 leading-relaxed">
            Industrial data mining system on the Kaggle Credit Card dataset (8,950 accounts, 18 behavioral attributes).
            Engineered with a literature-backed autonomous hill-climbing optimization engine that maximises clustering validity metrics.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link href="/autoresearch">
              <Button variant="primary" icon={<Cpu className="w-4 h-4" />}>
                Launch Autoresearch Loop
              </Button>
            </Link>
            <Link href="/clusters">
              <Button variant="secondary" icon={<Layers className="w-4 h-4" />}>
                Explore 2D/3D Clusters
              </Button>
            </Link>
            <Link href="/playground">
              <Button variant="outline" icon={<SlidersHorizontal className="w-4 h-4" />}>
                Customer Profiler Playground
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Dataset Health Telemetry Row */}
      {health && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <StatCard
            title="Total Accounts"
            value={health.totalRows.toLocaleString()}
            subtitle="Customer records"
            icon={<Database className="w-4 h-4 text-emerald-400" />}
          />
          <StatCard
            title="Attributes"
            value={health.totalFeatures}
            subtitle="17 Numeric + 1 ID"
            icon={<FileSpreadsheet className="w-4 h-4 text-cyan-400" />}
          />
          <StatCard
            title="Missing Rate"
            value={formatPercent(health.missingValuesRate, 2)}
            subtitle="314 missing values"
            badge="Imputed"
            badgeVariant="emerald"
            icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}
          />
          <StatCard
            title="Outlier Rate"
            value={formatPercent(health.outlierRate, 1)}
            subtitle="376 extreme points"
            badge="Winsorized"
            badgeVariant="cyan"
            icon={<TrendingUp className="w-4 h-4 text-rose-400" />}
          />
          <StatCard
            title="Hopkins Stat"
            value={health.hopkinsStatistic.toFixed(3)}
            subtitle="Clustering tendency"
            badge="Clustered"
            badgeVariant="emerald"
            icon={<Activity className="w-4 h-4 text-emerald-400" />}
          />
          <StatCard
            title="Memory Size"
            value={`${health.memoryUsageMb} MB`}
            subtitle="In-memory cache"
            icon={<Cpu className="w-4 h-4 text-violet-400" />}
          />
        </div>
      )}

      {/* 6-Phase Interactive CRISP-DM Lifecycle Tracker */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-emerald-400" />
              CRISP-DM 6-Phase Execution Roadmap
            </h2>
            <p className="text-xs text-slate-400">Click any phase card to inspect stage deliverables and KPI metrics.</p>
          </div>
          <Badge variant="emerald">100% Phase Completed</Badge>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {phases.map((phase) => {
            const isSelected = phase.phaseId === selectedPhase;
            return (
              <button
                key={phase.phaseId}
                onClick={() => setSelectedPhase(phase.phaseId)}
                className={`text-left p-4 rounded-xl border transition-all duration-200 ${
                  isSelected
                    ? 'bg-emerald-950/60 border-emerald-500/80 shadow-lg shadow-emerald-950/50 scale-[1.02]'
                    : 'glass-card border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                    P{phase.phaseId}: {phase.code}
                  </span>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="font-bold text-xs text-white line-clamp-1">{phase.name}</div>
                <div className="text-[11px] text-slate-400 mt-1 line-clamp-2">{phase.badge}</div>
              </button>
            );
          })}
        </div>

        {/* Selected Phase Detail Panel */}
        {activePhaseData && (
          <Card className="border-emerald-800/40 bg-slate-900/90 p-6 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
              <div>
                <div className="flex items-center gap-2.5">
                  <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-emerald-950 border border-emerald-700/60 text-emerald-400">
                    PHASE {activePhaseData.phaseId}
                  </span>
                  <h3 className="text-lg font-bold text-white">{activePhaseData.name}</h3>
                  <Badge variant="emerald">{activePhaseData.badge}</Badge>
                </div>
                <p className="text-xs text-slate-300 mt-1">{activePhaseData.description}</p>
              </div>

              <Link href={activePhaseData.actionRoute}>
                <Button variant="secondary" size="sm" icon={<ArrowRight className="w-3.5 h-3.5" />}>
                  Navigate to View
                </Button>
              </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
              {/* KPIs */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Phase Key Metrics</h4>
                <div className="space-y-2">
                  {activePhaseData.kpis.map((kpi, idx) => (
                    <div key={idx} className="flex justify-between items-center p-2.5 rounded-lg bg-slate-950/60 border border-slate-800">
                      <span className="text-xs text-slate-400">{kpi.label}</span>
                      <span className="text-xs font-mono font-semibold text-emerald-400">{kpi.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Execution Deliverables */}
              <div className="md:col-span-2 space-y-2">
                <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Engineering Deliverables</h4>
                <div className="space-y-2">
                  {activePhaseData.details.map((detail, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-200">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        )}
      </div>

      {/* Feature Profiling & Distribution Table */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <FileSpreadsheet className="w-5 h-5 text-cyan-400" />
              18-Attribute Statistical Profiling
            </h2>
            <p className="text-xs text-slate-400">Descriptive statistics, distribution shapes, and missingness audits.</p>
          </div>

          <div className="relative w-full sm:w-72">
            <Search className="w-4 h-4 absolute left-3 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search feature attributes..."
              value={searchFeature}
              onChange={(e: any) => setSearchFeature(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500 transition-colors"
            />
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4">Feature Name</th>
                <th className="py-3 px-3">Type</th>
                <th className="py-3 px-3">Mean</th>
                <th className="py-3 px-3">Std Dev</th>
                <th className="py-3 px-3">Median</th>
                <th className="py-3 px-3">Skewness</th>
                <th className="py-3 px-3">Missing</th>
                <th className="py-3 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {filteredFeatures.map((feat) => (
                <tr key={feat.name} className="hover:bg-slate-900/50 transition-colors">
                  <td className="py-3 px-4 font-bold text-white flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                    {feat.name}
                  </td>
                  <td className="py-3 px-3 text-slate-400 font-sans">
                    <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-[10px]">
                      {feat.dataType}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-200">{formatNumber(feat.mean, 2)}</td>
                  <td className="py-3 px-3 text-slate-400">{formatNumber(feat.std, 2)}</td>
                  <td className="py-3 px-3 text-slate-200">{formatNumber(feat.median, 2)}</td>
                  <td className="py-3 px-3">
                    <span className={feat.skewness > 3.0 ? 'text-amber-400 font-semibold' : 'text-slate-300'}>
                      {formatNumber(feat.skewness, 2)}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    {feat.missingCount > 0 ? (
                      <span className="text-amber-400 font-bold">{feat.missingCount} ({formatPercent(feat.missingRate, 1)})</span>
                    ) : (
                      <span className="text-emerald-400 font-medium">0 (0%)</span>
                    )}
                  </td>
                  <td className="py-3 px-4 font-sans text-slate-300 text-[11px] max-w-xs truncate" title={feat.description}>
                    {feat.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Interactive Feature Correlation Matrix */}
      {correlation && (
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <BarChart2 className="w-5 h-5 text-emerald-400" />
                Inter-Feature Correlation Matrix
              </h2>
              <p className="text-xs text-slate-400">Collinearity grid revealing behavioral relationships (e.g. Purchases vs One-off r=0.91).</p>
            </div>

            <Tabs
              tabs={[
                { id: 'pearson', label: 'Pearson (Linear)' },
                { id: 'spearman', label: 'Spearman (Rank)' },
              ]}
              activeTab={corrType}
              onChange={(tab) => setCorrType(tab as 'pearson' | 'spearman')}
            />
          </div>

          <Card className="p-4 overflow-x-auto">
            <div className="min-w-[680px]">
              <div className="grid grid-cols-11 gap-1 text-[11px] font-mono text-center">
                <div className="p-2 font-bold text-slate-400 text-left">Feature</div>
                {correlation.features.map((f) => (
                  <div key={f} className="p-2 font-semibold text-slate-400 truncate" title={f}>
                    {f.slice(0, 7)}
                  </div>
                ))}

                {correlation.features.map((f1, rIdx) => (
                  <React.Fragment key={f1}>
                    <div className="p-2 font-semibold text-slate-300 text-left truncate" title={f1}>
                      {f1.slice(0, 9)}
                    </div>
                    {correlation.features.map((f2, cIdx) => {
                      const val =
                        corrType === 'pearson'
                          ? correlation.pearson[rIdx][cIdx]
                          : correlation.spearman[rIdx][cIdx];

                      let bgColor = 'bg-slate-900';
                      let textColor = 'text-slate-400';
                      if (val >= 0.7) {
                        bgColor = 'bg-emerald-600/80';
                        textColor = 'text-white font-bold';
                      } else if (val >= 0.4) {
                        bgColor = 'bg-emerald-800/60';
                        textColor = 'text-emerald-200';
                      } else if (val >= 0.15) {
                        bgColor = 'bg-emerald-950/60';
                        textColor = 'text-emerald-300';
                      } else if (val <= -0.2) {
                        bgColor = 'bg-rose-900/60';
                        textColor = 'text-rose-200';
                      }

                      return (
                        <div
                          key={`${f1}-${f2}`}
                          onMouseEnter={() => setHoveredCorr({ f1, f2, val })}
                          onMouseLeave={() => setHoveredCorr(null)}
                          className={`p-2 rounded cursor-pointer transition-all hover:scale-105 ${bgColor} ${textColor}`}
                        >
                          {val.toFixed(2)}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>

              {/* Hover indicator */}
              <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                <div>
                  {hoveredCorr ? (
                    <span>
                      Correlation between <strong className="text-white">{hoveredCorr.f1}</strong> and{' '}
                      <strong className="text-white">{hoveredCorr.f2}</strong>:{' '}
                      <strong className="text-emerald-400 font-mono">{hoveredCorr.val.toFixed(3)}</strong>
                    </span>
                  ) : (
                    <span>Hover over any grid cell to view feature correlation details.</span>
                  )}
                </div>
                <div className="flex items-center gap-3 text-[11px]">
                  <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-600 inline-block" /> Strong (+0.7 to +1.0)</span>
                  <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-emerald-950 border border-emerald-800 inline-block" /> Moderate (+0.2 to +0.6)</span>
                  <span className="flex items-center gap-1.5"><span className="w-3 h-3 rounded bg-rose-900 inline-block" /> Negative (-0.3)</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
