'use client';

import React, { useState, useEffect } from 'react';
import {
  TableProperties,
  BookOpen,
  Copy,
  Check,
  Download,
  Sparkles,
  Award,
  Layers,
  FileCode,
  CheckCircle2,
  TrendingUp,
} from 'lucide-react';
import { api } from '@/lib/api';
import {
  BenchmarkRow,
  AblationEntry,
  LiteraturePaper,
} from '@/types/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';
import { StatCard } from '@/components/ui/StatCard';
import { generateLatexTable, generateMarkdownTable, formatPercent } from '@/lib/utils';

export default function BenchmarksPage() {
  const [benchmarks, setBenchmarks] = useState<BenchmarkRow[]>([]);
  const [ablations, setAblations] = useState<AblationEntry[]>([]);
  const [papers, setPapers] = useState<LiteraturePaper[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [latexModalOpen, setLatexModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<'benchmarks' | 'ablations' | 'literature'>('benchmarks');

  useEffect(() => {
    async function loadData() {
      const [bm, ab, lp] = await Promise.all([
        api.getBenchmarkMatrix(),
        api.getAblationEntries(),
        api.getLiteraturePapers(),
      ]);
      setBenchmarks(bm);
      setAblations(ab);
      setPapers(lp);
    }
    loadData();
  }, []);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const latexCode = generateLatexTable(benchmarks);
  const markdownCode = generateMarkdownTable(benchmarks);

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Top Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900/80 p-5 rounded-2xl border border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-extrabold text-white flex items-center gap-2">
              <TableProperties className="w-6 h-6 text-emerald-400" />
              Research Benchmark & Literature Matrix
            </h1>
            <Badge variant="emerald">Paper-Style Synthesis</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Comparative performance bounds (10-fold CV), systematic ablation studies, and literature citations.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Button
            variant="primary"
            size="md"
            icon={<FileCode className="w-4 h-4" />}
            onClick={() => setLatexModalOpen(true)}
          >
            Export LaTeX Table
          </Button>
          <Button
            variant="secondary"
            size="md"
            icon={copiedId === 'md_all' ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            onClick={() => handleCopy(markdownCode, 'md_all')}
          >
            {copiedId === 'md_all' ? 'Copied Markdown' : 'Copy Markdown'}
          </Button>
        </div>
      </div>

      {/* Benchmark Summary Telemetry */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <StatCard
          title="State-of-the-Art Model"
          value="0.584 ± 0.009"
          subtitle="Autoresearch K-Means++"
          badge="Top Silhouette"
          badgeVariant="emerald"
          icon={<Award className="w-4 h-4 text-emerald-400" />}
        />
        <StatCard
          title="Optimal DB Index"
          value="0.812 ± 0.021"
          subtitle="Lowest cluster overlap"
          badge="Minimized"
          badgeVariant="cyan"
          icon={<TrendingUp className="w-4 h-4 text-cyan-400" />}
        />
        <StatCard
          title="Ablation Lift"
          value="+0.202"
          subtitle="From 0.382 raw baseline"
          badge="+52.8% Gain"
          badgeVariant="emerald"
          icon={<Sparkles className="w-4 h-4 text-amber-400" />}
        />
        <StatCard
          title="Literature Grounding"
          value="6 Seminal Papers"
          subtitle="Rousseeuw, DB, CH, Arthur"
          icon={<BookOpen className="w-4 h-4 text-violet-400" />}
        />
      </div>

      {/* Main Comparative Benchmark Table */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <TableProperties className="w-5 h-5 text-emerald-400" />
              Comparative Multi-Paradigm Benchmark Table (10-Fold Cross Validation)
            </h2>
            <p className="text-xs text-slate-400">
              Evaluated across 4 paradigms with mean and standard deviation bounds ($\mu \pm \sigma$). Asterisk (*) denotes hill-climbed optimization.
            </p>
          </div>
          <Badge variant="emerald">10-Fold Repeated CV</Badge>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-semibold uppercase tracking-wider text-[11px] font-sans">
                <th className="py-3 px-4">Clustering Model</th>
                <th className="py-3 px-3">Paradigm</th>
                <th className="py-3 px-3">Configuration</th>
                <th className="py-3 px-3">Silhouette (↑)</th>
                <th className="py-3 px-3">Davies-Bouldin (↓)</th>
                <th className="py-3 px-3">Calinski-Harabasz (↑)</th>
                <th className="py-3 px-3">Stability ARI</th>
                <th className="py-3 px-3">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {benchmarks.map((row) => (
                <tr
                  key={row.algorithm}
                  className={`hover:bg-slate-900/60 transition-colors ${
                    row.isOptimized ? 'bg-emerald-950/20 font-bold' : ''
                  }`}
                >
                  <td className="py-3 px-4 font-bold text-white flex items-center gap-2 font-sans">
                    {row.isOptimized && <Sparkles className="w-3.5 h-3.5 text-emerald-400 shrink-0" />}
                    <span>{row.algorithm}</span>
                    {row.isOptimized && <span className="text-emerald-400 font-mono text-[11px]">*</span>}
                  </td>
                  <td className="py-3 px-3 text-slate-300 font-sans">
                    <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-[10px]">
                      {row.paradigm}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-400 text-[11px]">{row.kOrEps}</td>
                  <td className="py-3 px-3">
                    <span className={row.isOptimized ? 'text-emerald-400 font-bold' : 'text-slate-200'}>
                      {row.silhouetteMean.toFixed(3)} ± {row.silhouetteStd.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-3 px-3">
                    <span className={row.isOptimized ? 'text-cyan-400 font-bold' : 'text-slate-300'}>
                      {row.daviesBouldinMean.toFixed(3)} ± {row.daviesBouldinStd.toFixed(3)}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-200">
                    {row.calinskiHarabaszMean.toFixed(1)} ± {row.calinskiHarabaszStd.toFixed(1)}
                  </td>
                  <td className="py-3 px-3 text-emerald-300 font-medium">
                    {formatPercent(row.stabilityScore, 1)}
                  </td>
                  <td className="py-3 px-3 text-slate-400">{row.runtimeMs.toFixed(1)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Systematic Ablation Matrix */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-cyan-400" />
              Systematic Pipeline Ablation Study
            </h2>
            <p className="text-xs text-slate-400">
              Isolates the performance lift contributed by data imputation, power transformations, PCA denoising, and hill-climbing search.
            </p>
          </div>
          <Badge variant="cyan">Cumulative Lift: +0.202</Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {ablations.map((ab, idx) => (
            <Card key={idx} className="p-4 space-y-2 border-slate-800 bg-slate-950/70 flex flex-col justify-between">
              <div>
                <div className="text-[11px] font-mono text-emerald-400 font-bold">{ab.stage}</div>
                <h4 className="text-xs font-bold text-white mt-1">{ab.component}</h4>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{ab.description}</p>
              </div>

              <div className="pt-2 border-t border-slate-800 space-y-1 font-mono text-xs">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Silhouette:</span>
                  <span className="font-bold text-white">{ab.silhouette.toFixed(3)}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Δ Gain:</span>
                  <span className={ab.deltaSilhouette > 0 ? 'text-emerald-400 font-bold' : 'text-slate-500'}>
                    {ab.deltaSilhouette > 0 ? `+${ab.deltaSilhouette.toFixed(3)}` : '0.000 (Base)'}
                  </span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Academic Literature Bibliography Panel */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-violet-400" />
              Academic Literature Alignment & Bibliography
            </h2>
            <p className="text-xs text-slate-400">
              Peer-reviewed foundations supporting our evaluation metrics, algorithm selections, and optimization methodology.
            </p>
          </div>
          <Badge variant="violet">6 Citations</Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {papers.map((paper) => (
            <Card key={paper.id} className="p-5 space-y-3 border-slate-800 bg-slate-900/60">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-bold text-white leading-snug">{paper.title}</h3>
                  <p className="text-xs text-emerald-400 mt-0.5">
                    {paper.authors} ({paper.year})
                  </p>
                  <p className="text-[11px] text-slate-400 italic">{paper.venue}</p>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  icon={copiedId === paper.id ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  onClick={() => handleCopy(paper.bibtex, paper.id)}
                >
                  {copiedId === paper.id ? 'Copied' : 'BibTeX'}
                </Button>
              </div>

              <div className="space-y-1.5 pt-2 border-t border-slate-800 text-xs">
                <div>
                  <strong className="text-slate-300 font-medium">Core Contribution: </strong>
                  <span className="text-slate-400">{paper.keyContribution}</span>
                </div>
                <div>
                  <strong className="text-emerald-400 font-medium">Platform Relevance: </strong>
                  <span className="text-slate-300">{paper.relevanceToProject}</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* LaTeX Exporter Modal */}
      <Modal
        isOpen={latexModalOpen}
        onClose={() => setLatexModalOpen(false)}
        title="LaTeX Benchmark Table Exporter (Paper Ready)"
        maxWidth="max-w-3xl"
      >
        <div className="space-y-4">
          <p className="text-xs text-slate-300">
            Copy this snippet directly into your LaTeX research paper manuscript (e.g. IEEE Transactions, ACM SIGKDD, NeurIPS).
          </p>

          <div className="relative">
            <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-slate-300 text-xs font-mono overflow-x-auto max-h-96">
              {latexCode}
            </pre>
            <button
              onClick={() => handleCopy(latexCode, 'latex_modal')}
              className="absolute top-3 right-3 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1.5 shadow-lg"
            >
              {copiedId === 'latex_modal' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              <span>{copiedId === 'latex_modal' ? 'Copied LaTeX' : 'Copy Code'}</span>
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
