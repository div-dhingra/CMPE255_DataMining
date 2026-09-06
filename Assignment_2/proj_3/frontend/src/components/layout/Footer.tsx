import React from 'react';
import { Activity, BookOpen, GitBranch, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950/60 mt-16 py-8 text-slate-400 text-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-2">
            <div className="flex items-center gap-2 font-bold text-white text-sm">
              <Activity className="w-4 h-4 text-emerald-400" />
              <span>CRISP-DM Autoresearch Platform</span>
            </div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Industrial customer segmentation system built on the Kaggle Credit Card benchmark with autonomous hill-climbing search and literature alignment.
            </p>
          </div>

          <div>
            <h4 className="font-semibold text-slate-200 uppercase tracking-wider text-[11px] mb-2">Clustering Paradigms</h4>
            <ul className="space-y-1 text-slate-400 text-[11px]">
              <li>Partitioning: K-Means++, K-Medoids</li>
              <li>Density: DBSCAN, HDBSCAN</li>
              <li>Hierarchical: Agglomerative (Ward)</li>
              <li>Probabilistic: Gaussian Mixture Models</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-slate-200 uppercase tracking-wider text-[11px] mb-2">Literature Alignment</h4>
            <ul className="space-y-1 text-slate-400 text-[11px]">
              <li>Rousseeuw (1987) Silhouette Metric</li>
              <li>Davies & Bouldin (1979) Separation</li>
              <li>Caliński & Harabasz (1974) Variance</li>
              <li>Campello et al. (2013) HDBSCAN</li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-slate-200 uppercase tracking-wider text-[11px] mb-2">System Telemetry</h4>
            <div className="space-y-1 text-[11px] font-mono">
              <div className="flex justify-between">
                <span>Dataset:</span>
                <span className="text-slate-200">8,950 Records</span>
              </div>
              <div className="flex justify-between">
                <span>Optimization:</span>
                <span className="text-emerald-400">Silhouette 0.584</span>
              </div>
              <div className="flex justify-between">
                <span>Stability:</span>
                <span className="text-cyan-400">0.982 ARI</span>
              </div>
              <div className="flex justify-between">
                <span>Inference:</span>
                <span className="text-emerald-400">&lt; 2.5ms</span>
              </div>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-800/60 mt-8 pt-4 flex flex-col sm:flex-row items-center justify-between text-slate-500 text-[11px] gap-2">
          <div>CMPE 255 Data Mining — Autonomous Autoresearch Project</div>
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> High-Fidelity Dual-Mode Engine</span>
            <span>Next.js 14 + FastAPI</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
