'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Play,
  Pause,
  RotateCcw,
  FastForward,
  Flame,
  CheckCircle2,
  XCircle,
  TrendingUp,
  Award,
  Zap,
  Sliders,
  ShieldCheck,
} from 'lucide-react';
import { api } from '@/lib/api';
import {
  AutoresearchStep,
  AutoresearchLeaderboardItem,
} from '@/types/api';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Slider } from '@/components/ui/Slider';
import { StatCard } from '@/components/ui/StatCard';
import { TrajectoryChart } from '@/components/visualizations/TrajectoryChart';
import { formatNumber, formatPercent } from '@/lib/utils';

export default function AutoresearchPage() {
  const [history, setHistory] = useState<AutoresearchStep[]>([]);
  const [leaderboard, setLeaderboard] = useState<AutoresearchLeaderboardItem[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStepIdx, setCurrentStepIdx] = useState(10);
  const [maxSteps, setMaxSteps] = useState(50);
  const [patience, setPatience] = useState(5);
  const [temperature, setTemperature] = useState(0.68);
  const [bestScore, setBestScore] = useState(0.5842);
  const [statusMessage, setStatusMessage] = useState('Optimal configuration converged.');

  const intervalRef = useRef<any>(null);

  useEffect(() => {
    async function loadData() {
      const [hist, lead] = await Promise.all([
        api.getAutoresearchHistory(),
        api.getAutoresearchLeaderboard(),
      ]);
      setHistory(hist);
      setLeaderboard(lead);
    }
    loadData();
  }, []);

  // Live Step Simulator
  const executeStep = () => {
    setHistory((prev) => {
      const nextStepNum = prev.length + 1;
      const prevBest = Math.max(...prev.map((s) => s.silhouetteScore), 0.442);

      // Random stochastic mutation
      const algorithms = ['K-Means++', 'GMM', 'Agglomerative', 'HDBSCAN'];
      const alg = algorithms[Math.floor(Math.random() * algorithms.length)];
      const isPerturbation = nextStepNum % 8 === 0;

      let newScore = prevBest + (Math.random() * 0.02 - 0.009);
      if (isPerturbation) newScore = prevBest - 0.03; // temperature jump

      newScore = Math.max(0.40, Math.min(0.62, Number(newScore.toFixed(4))));
      const deltaS = Number((newScore - prevBest).toFixed(4));
      const isAccepted = deltaS > 0 || Math.random() < temperature * 0.3;

      const newStep: AutoresearchStep = {
        step: nextStepNum,
        timestamp: new Date().toISOString(),
        algorithm: alg,
        hyperparameters: { k: 4, n_init: 20 + Math.floor(Math.random() * 10) },
        preprocessing: { scaler: 'PowerTransformer (Yeo-Johnson)', dim_red: 'PCA (12)' },
        silhouetteScore: newScore,
        daviesBouldinIndex: Number((1.2 - (newScore - 0.4) * 1.5).toFixed(3)),
        calinskiHarabaszIndex: Math.round(2500 + newScore * 2400),
        stabilityScore: Number((0.90 + newScore * 0.12).toFixed(3)),
        compositeFitness: Number((newScore * 0.8 + (1 / (1 + (1.2 - newScore))) * 0.4).toFixed(3)),
        status: isPerturbation ? 'PERTURBED' : isAccepted ? 'ACCEPTED' : 'REJECTED',
        mutationDescription: isPerturbation
          ? 'Applied neighborhood perturbation (Simulated Annealing jump)'
          : isAccepted
          ? `Mutated hyperparameter on ${alg} (ΔS = +${deltaS.toFixed(4)})`
          : `Candidate evaluated on ${alg} (ΔS = ${deltaS.toFixed(4)}, Rejected)`,
        deltaSilhouette: deltaS,
        temperature: Number((temperature * 0.97).toFixed(3)),
      };

      if (isAccepted && newScore > bestScore) {
        setBestScore(newScore);
        setStatusMessage(`New best score discovered: ${newScore.toFixed(4)} on ${alg}`);
      }

      setTemperature((t) => Math.max(0.05, t * 0.97));
      return [...prev, newStep];
    });
  };

  useEffect(() => {
    if (isRunning) {
      intervalRef.current = setInterval(() => {
        executeStep();
      }, 1200);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRunning, temperature, bestScore]);

  const handleStart = () => setIsRunning(true);
  const handlePause = () => setIsRunning(false);
  const handlePerturb = () => {
    setTemperature(1.0);
    executeStep();
  };
  const handleReset = async () => {
    setIsRunning(false);
    const hist = await api.getAutoresearchHistory();
    setHistory(hist);
    setBestScore(0.5842);
    setTemperature(0.68);
    setStatusMessage('Restored baseline optimization ledger.');
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header Bar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900/80 p-5 rounded-2xl border border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-extrabold text-white flex items-center gap-2">
              <Cpu className="w-6 h-6 text-emerald-400" />
              Autoresearch Hill-Climbing Studio
            </h1>
            <Badge variant={isRunning ? 'emerald' : 'slate'}>
              {isRunning ? '🟢 Active Search Loop' : '⏸ Loop Paused'}
            </Badge>
          </div>
          <p className="text-xs text-slate-400">
            Autonomous pipeline search across preprocessing, transformations, and hyperparameters.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex flex-wrap items-center gap-2.5">
          {!isRunning ? (
            <Button variant="primary" size="md" icon={<Play className="w-4 h-4" />} onClick={handleStart}>
              Start Autoresearch
            </Button>
          ) : (
            <Button variant="secondary" size="md" icon={<Pause className="w-4 h-4" />} onClick={handlePause}>
              Pause Optimization
            </Button>
          )}
          <Button variant="secondary" size="md" icon={<FastForward className="w-4 h-4" />} onClick={executeStep}>
            Step Next
          </Button>
          <Button variant="secondary" size="md" icon={<Flame className="w-4 h-4 text-amber-400" />} onClick={handlePerturb}>
            Perturb / Restart
          </Button>
          <Button variant="outline" size="md" icon={<RotateCcw className="w-4 h-4" />} onClick={handleReset}>
            Reset
          </Button>
        </div>
      </div>

      {/* Live Telemetry Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard
          title="Best Silhouette"
          value={bestScore.toFixed(4)}
          subtitle="+0.142 vs baseline"
          badge="Maximized"
          badgeVariant="emerald"
          icon={<Award className="w-4 h-4 text-emerald-400" />}
        />
        <StatCard
          title="Current Step"
          value={`Step ${history.length}`}
          subtitle={`Max: ${maxSteps}`}
          icon={<TrendingUp className="w-4 h-4 text-cyan-400" />}
        />
        <StatCard
          title="Annealing Temp"
          value={temperature.toFixed(2)}
          subtitle="Exploration rate"
          badge={temperature > 0.5 ? 'Exploratory' : 'Exploiting'}
          badgeVariant={temperature > 0.5 ? 'amber' : 'emerald'}
          icon={<Flame className="w-4 h-4 text-amber-400" />}
        />
        <StatCard
          title="Best Algorithm"
          value="K-Means++"
          subtitle="Elkan (k=4)"
          icon={<Cpu className="w-4 h-4 text-violet-400" />}
        />
        <StatCard
          title="Search Space"
          value="480 Combos"
          subtitle="Pipeline choices"
          icon={<Sliders className="w-4 h-4 text-slate-400" />}
        />
        <StatCard
          title="Loop Status"
          value={isRunning ? 'SEARCHING' : 'IDLE'}
          subtitle={statusMessage.slice(0, 18) + '...'}
          icon={<Zap className="w-4 h-4 text-emerald-400" />}
        />
      </div>

      {/* Trajectory Curves & Search Control Parameters */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Trajectory Chart */}
        <div className="lg:col-span-8 space-y-4">
          <Card className="p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  Real-Time Optimization Trajectory
                </h3>
                <p className="text-[11px] text-slate-400">
                  Dual-axis curve tracking Silhouette maximization (Green) vs. Davies-Bouldin minimization (Cyan).
                </p>
              </div>

              <div className="flex items-center gap-2 text-xs font-mono">
                <span className="text-emerald-400 font-semibold">Accepted Steps: {history.filter((h) => h.status === 'ACCEPTED').length}</span>
                <span className="text-slate-500">|</span>
                <span className="text-rose-400">Rejected: {history.filter((h) => h.status === 'REJECTED').length}</span>
              </div>
            </div>

            <TrajectoryChart steps={history} />
          </Card>

          {/* Parameter Delta Mutation Log */}
          <Card className="p-4 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-400" />
              Step-by-Step Parameter Delta & Mutation Ledger
            </h3>

            <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
              {[...history].reverse().map((step) => {
                const isAccepted = step.status === 'ACCEPTED';
                const isPerturbed = step.status === 'PERTURBED';

                return (
                  <div
                    key={step.step}
                    className={`flex items-start justify-between gap-3 p-2.5 rounded-lg border text-xs ${
                      isPerturbed
                        ? 'bg-amber-950/40 border-amber-800/60 text-amber-200'
                        : isAccepted
                        ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200'
                        : 'bg-slate-950 border-slate-800 text-slate-400'
                    }`}
                  >
                    <div className="flex items-start gap-2">
                      {isPerturbed ? (
                        <Flame className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      ) : isAccepted ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                      )}
                      <div>
                        <div className="flex items-center gap-2 font-mono font-bold">
                          <span className="text-white">Step {step.step}</span>
                          <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-900 border border-slate-800">
                            {step.algorithm}
                          </span>
                          <span className="text-[10px] uppercase font-semibold text-slate-300">
                            [{step.status}]
                          </span>
                        </div>
                        <p className="text-[11px] text-slate-300 mt-0.5">{step.mutationDescription}</p>
                      </div>
                    </div>

                    <div className="text-right shrink-0 font-mono">
                      <div className="font-bold text-white">S = {step.silhouetteScore.toFixed(4)}</div>
                      <div className={step.deltaSilhouette >= 0 ? 'text-emerald-400 text-[10px]' : 'text-rose-400 text-[10px]'}>
                        ΔS = {step.deltaSilhouette >= 0 ? `+${step.deltaSilhouette.toFixed(4)}` : step.deltaSilhouette.toFixed(4)}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>

        {/* Right Column: Search Settings & Best Candidate Box */}
        <div className="lg:col-span-4 space-y-4">
          {/* Best Pipeline Spec Card */}
          <Card className="p-4 space-y-3 border-emerald-800/60 bg-gradient-to-b from-slate-900 to-emerald-950/30">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2 font-bold text-sm text-white">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>Champion Pipeline Candidate</span>
              </div>
              <Badge variant="emerald">Rank #1</Badge>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Algorithm:</span>
                <span className="text-white font-mono font-bold">K-Means++ (Elkan)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Clusters (k):</span>
                <span className="text-emerald-400 font-mono font-bold">k = 4</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Transformation:</span>
                <span className="text-white font-mono">Yeo-Johnson Power</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Outlier Capping:</span>
                <span className="text-white font-mono">Winsorization (1%-99%)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Denoising:</span>
                <span className="text-white font-mono">PCA (12 Components)</span>
              </div>
              <div className="flex justify-between py-1 border-b border-slate-800/60">
                <span className="text-slate-400">Silhouette Score:</span>
                <span className="text-emerald-400 font-mono font-bold">{bestScore.toFixed(4)}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-slate-400">Davies-Bouldin:</span>
                <span className="text-cyan-400 font-mono font-bold">0.8124</span>
              </div>
            </div>
          </Card>

          {/* Hyperparameter Search Tuner Controls */}
          <Card className="p-4 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Metaheuristic Search Controls
            </h3>

            <Slider
              label="Max Iterations / Step Budget"
              value={maxSteps}
              min={10}
              max={100}
              step={5}
              unit=" steps"
              onChange={setMaxSteps}
              description="Maximum exploration step horizon."
            />

            <Slider
              label="Restart Patience"
              value={patience}
              min={2}
              max={15}
              step={1}
              unit=" steps"
              onChange={setPatience}
              description="Consecutive non-improving steps before random restart."
            />

            <Slider
              label="Simulated Annealing Temp"
              value={temperature}
              min={0.05}
              max={1.0}
              step={0.05}
              formatValue={(v) => `T = ${v.toFixed(2)}`}
              onChange={setTemperature}
              description="Probability threshold for accepting suboptimal mutations."
            />
          </Card>
        </div>
      </div>

      {/* Experiment Trial Leaderboard */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Award className="w-5 h-5 text-emerald-400" />
              Autonomous Experiment Leaderboard & Pareto Frontier
            </h2>
            <p className="text-xs text-slate-400">
              Ranked top-performing pipeline configurations evaluated against composite objective F(θ).
            </p>
          </div>
          <Badge variant="emerald">Top 6 Trials</Badge>
        </div>

        <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/70">
          <table className="w-full text-left text-xs border-collapse font-mono">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-400 font-semibold uppercase tracking-wider text-[11px] font-sans">
                <th className="py-3 px-3">Rank</th>
                <th className="py-3 px-3">Trial ID</th>
                <th className="py-3 px-4">Algorithm</th>
                <th className="py-3 px-4">Preprocessing Pipeline</th>
                <th className="py-3 px-3">Silhouette (↑)</th>
                <th className="py-3 px-3">Davies-Bouldin (↓)</th>
                <th className="py-3 px-3">Calinski-H (↑)</th>
                <th className="py-3 px-3">Stability</th>
                <th className="py-3 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {leaderboard.map((item) => (
                <tr key={item.trialId} className="hover:bg-slate-900/50 transition-colors">
                  <td className="py-3 px-3 font-bold text-white">
                    <span className={`px-2 py-0.5 rounded text-xs ${item.rank === 1 ? 'bg-emerald-950 text-emerald-400 border border-emerald-700' : 'text-slate-400'}`}>
                      #{item.rank}
                    </span>
                  </td>
                  <td className="py-3 px-3 text-slate-300">#{item.trialId}</td>
                  <td className="py-3 px-4 font-bold text-white font-sans">{item.algorithm}</td>
                  <td className="py-3 px-4 text-slate-300 font-sans text-[11px] truncate max-w-xs" title={item.preprocessing}>
                    {item.preprocessing}
                  </td>
                  <td className="py-3 px-3 font-bold text-emerald-400">{item.silhouette.toFixed(4)}</td>
                  <td className="py-3 px-3 text-cyan-400">{item.daviesBouldin.toFixed(4)}</td>
                  <td className="py-3 px-3 text-slate-200">{item.calinskiHarabasz.toFixed(0)}</td>
                  <td className="py-3 px-3 text-emerald-300">{formatPercent(item.stability, 1)}</td>
                  <td className="py-3 px-3 font-sans">
                    {item.isPareto ? (
                      <span className="px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800 text-[10px] font-semibold">
                        Pareto Optimal
                      </span>
                    ) : (
                      <span className="text-slate-500 text-[11px]">Dominated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
