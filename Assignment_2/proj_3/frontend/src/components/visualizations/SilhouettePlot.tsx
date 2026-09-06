'use client';

import React, { useRef, useEffect } from 'react';
import { SilhouetteSample, ClusterPersona } from '@/types/api';

interface SilhouettePlotProps {
  samples: SilhouetteSample[];
  personas: ClusterPersona[];
  averageScore?: number;
}

const CLUSTER_COLORS = ['#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#f43f5e'];

export const SilhouettePlot: React.FC<SilhouettePlotProps> = ({
  samples,
  personas,
  averageScore = 0.584,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const paddingLeft = 40;
    const paddingRight = 30;
    const paddingTop = 20;
    const paddingBottom = 30;

    const plotWidth = width - paddingLeft - paddingRight;
    const plotHeight = height - paddingTop - paddingBottom;

    // Silhouette score range: -0.1 to 1.0
    const minScore = -0.1;
    const maxScore = 1.0;
    const scoreRange = maxScore - minScore;

    const scoreToX = (score: number) => {
      return paddingLeft + ((score - minScore) / scoreRange) * plotWidth;
    };

    // Draw Grid Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let s = 0.0; s <= 1.0; s += 0.2) {
      const x = scoreToX(s);
      ctx.beginPath();
      ctx.moveTo(x, paddingTop);
      ctx.lineTo(x, height - paddingBottom);
      ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px monospace';
      ctx.fillText(s.toFixed(1), x - 8, height - paddingBottom + 16);
    }

    // Zero line
    const zeroX = scoreToX(0.0);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(zeroX, paddingTop);
    ctx.lineTo(zeroX, height - paddingBottom);
    ctx.stroke();

    // Group samples by cluster
    const clusterMap = new Map<number, SilhouetteSample[]>();
    samples.forEach((sample) => {
      const list = clusterMap.get(sample.cluster) || [];
      list.push(sample);
      clusterMap.set(sample.cluster, list);
    });

    const totalSamples = samples.length;
    let currentY = paddingTop;

    Array.from(clusterMap.entries()).forEach(([clusterId, clusterSamples]) => {
      const clusterHeight = (clusterSamples.length / totalSamples) * plotHeight;
      const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

      // Sort samples descending
      const sorted = [...clusterSamples].sort((a, b) => b.score - a.score);
      const barHeight = Math.max(1, clusterHeight / sorted.length);

      sorted.forEach((sample, i) => {
        const barY = currentY + i * barHeight;
        const barX = scoreToX(sample.score);

        ctx.fillStyle = color;
        ctx.fillRect(zeroX, barY, barX - zeroX, Math.max(1, barHeight - 0.5));
      });

      // Cluster label
      const persona = personas.find((p) => p.clusterId === clusterId);
      ctx.fillStyle = color;
      ctx.font = 'bold 11px system-ui, sans-serif';
      ctx.fillText(
        `C${clusterId}: ${persona?.name.split('/')[0].trim() || `Cluster ${clusterId}`}`,
        paddingLeft + 8,
        currentY + 16
      );

      currentY += clusterHeight + 4;
    });

    // Draw Global Average Silhouette Dashed Red Line
    const avgX = scoreToX(averageScore);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(avgX, paddingTop);
    ctx.lineTo(avgX, height - paddingBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Average line annotation
    ctx.fillStyle = '#f43f5e';
    ctx.font = 'bold 10px monospace';
    ctx.fillText(`Avg: ${averageScore.toFixed(3)}`, avgX - 25, paddingTop - 6);
  }, [samples, personas, averageScore]);

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950/80 p-3">
      <div className="flex items-center justify-between pb-2 border-b border-slate-800 text-xs font-semibold text-slate-300">
        <span>Silhouette Coefficient per Sample Distribution</span>
        <span className="text-rose-400 font-mono">Mean Score: {averageScore.toFixed(3)}</span>
      </div>
      <canvas
        ref={canvasRef}
        width={650}
        height={260}
        className="w-full h-[260px] block mt-1"
      />
    </div>
  );
};
