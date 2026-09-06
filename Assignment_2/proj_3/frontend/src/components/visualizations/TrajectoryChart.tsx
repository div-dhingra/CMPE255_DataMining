'use client';

import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { AutoresearchStep } from '@/types/api';

interface TrajectoryChartProps {
  steps: AutoresearchStep[];
}

export const TrajectoryChart: React.FC<TrajectoryChartProps> = ({ steps }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="w-full h-80 flex items-center justify-center text-slate-500">Loading Trajectory...</div>;
  }

  const chartData = steps.map((s) => ({
    step: `Step ${s.step}`,
    stepNum: s.step,
    silhouette: s.silhouetteScore,
    daviesBouldin: s.daviesBouldinIndex,
    compositeFitness: s.compositeFitness,
    status: s.status,
    description: s.mutationDescription,
  }));

  return (
    <div className="w-full h-80 relative">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="silhouetteGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
            </linearGradient>
            <linearGradient id="fitnessGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} stroke="#334155" />
          <YAxis
            yAxisId="left"
            domain={[0.3, 0.7]}
            tick={{ fill: '#10b981', fontSize: 11 }}
            stroke="#10b981"
            label={{ value: 'Silhouette ↑', angle: -90, position: 'insideLeft', fill: '#10b981', fontSize: 10 }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            domain={[0.6, 1.8]}
            tick={{ fill: '#06b6d4', fontSize: 11 }}
            stroke="#06b6d4"
            label={{ value: 'Davies-Bouldin ↓', angle: 90, position: 'insideRight', fill: '#06b6d4', fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              borderColor: '#334155',
              borderRadius: '8px',
              color: '#f8fafc',
              fontSize: '12px',
            }}
          />
          <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />

          <Area
            yAxisId="left"
            type="monotone"
            dataKey="silhouette"
            name="Silhouette Score (Maximizing)"
            stroke="#10b981"
            strokeWidth={2.5}
            fill="url(#silhouetteGrad)"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="daviesBouldin"
            name="Davies-Bouldin Index (Minimizing)"
            stroke="#06b6d4"
            strokeWidth={2}
            strokeDasharray="4 4"
            dot={{ fill: '#06b6d4', r: 3 }}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="compositeFitness"
            name="Composite Fitness F(θ)"
            stroke="#8b5cf6"
            strokeWidth={2}
            dot={{ fill: '#8b5cf6', r: 4 }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};
