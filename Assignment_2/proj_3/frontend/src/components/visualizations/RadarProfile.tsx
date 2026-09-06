'use client';

import React, { useState, useEffect } from 'react';
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import { ClusterPersona } from '@/types/api';

interface RadarProfileProps {
  personas: ClusterPersona[];
  selectedCluster: number | null;
}

export const RadarProfile: React.FC<RadarProfileProps> = ({
  personas,
  selectedCluster,
}) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="w-full h-80 flex items-center justify-center text-slate-500">Loading Radar...</div>;
  }

  const featureKeys = personas[0]?.radarMetrics.map((m) => m.feature) || [
    'Balance',
    'Purchases',
    'Cash Advance',
    'Credit Limit',
    'Payments',
    'Full Payment %',
  ];

  const chartData = featureKeys.map((feature, idx) => {
    const row: Record<string, any> = { feature };
    personas.forEach((p) => {
      row[`cluster_${p.clusterId}`] = p.radarMetrics[idx]?.value ?? 0.5;
    });
    return row;
  });

  return (
    <div className="w-full h-80 relative">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={chartData} margin={{ top: 10, right: 25, bottom: 10, left: 25 }}>
          <PolarGrid stroke="#334155" strokeDasharray="3 3" />
          <PolarAngleAxis
            dataKey="feature"
            tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 500 }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 1]}
            tick={{ fill: '#64748b', fontSize: 9 }}
            stroke="#1e293b"
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
          <Legend
            wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
            formatter={(value: any) => {
              const clusterId = parseInt(String(value).replace('cluster_', ''), 10);
              const p = personas.find((pr) => pr.clusterId === clusterId);
              return p ? `${p.name.split('/')[0].trim()}` : value;
            }}
          />

          {personas.map((persona) => {
            const isHighlighted = selectedCluster === null || selectedCluster === persona.clusterId;
            if (!isHighlighted && selectedCluster !== null) return null;

            return (
              <Radar
                key={persona.clusterId}
                name={`cluster_${persona.clusterId}`}
                dataKey={`cluster_${persona.clusterId}`}
                stroke={persona.color}
                fill={persona.color}
                fillOpacity={selectedCluster === persona.clusterId ? 0.45 : 0.15}
                strokeWidth={selectedCluster === persona.clusterId ? 2.5 : 1.5}
              />
            );
          })}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
