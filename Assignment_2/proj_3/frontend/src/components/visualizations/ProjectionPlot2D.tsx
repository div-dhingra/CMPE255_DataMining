'use client';

import React, { useRef, useEffect, useState } from 'react';
import { ClusterPoint2D, ClusterPersona } from '@/types/api';
import { Button } from '@/components/ui/Button';
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react';
import { formatCurrency } from '@/lib/utils';

interface ProjectionPlot2DProps {
  points: ClusterPoint2D[];
  personas: ClusterPersona[];
  selectedCluster: number | null;
  onSelectCluster?: (clusterId: number | null) => void;
  projectionType?: string;
}

const CLUSTER_COLORS = ['#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#f43f5e'];

export const ProjectionPlot2D: React.FC<ProjectionPlot2DProps> = ({
  points,
  personas,
  selectedCluster,
  onSelectCluster,
  projectionType = 'PCA',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredPoint, setHoveredPoint] = useState<ClusterPoint2D | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

  // Calculate centroids
  const centroids = React.useMemo(() => {
    const map = new Map<number, { sumX: number; sumY: number; count: number }>();
    points.forEach((p) => {
      const entry = map.get(p.cluster) || { sumX: 0, sumY: 0, count: 0 };
      entry.sumX += p.x;
      entry.sumY += p.y;
      entry.count += 1;
      map.set(p.cluster, entry);
    });
    return Array.from(map.entries()).map(([cluster, val]) => ({
      cluster,
      x: val.sumX / val.count,
      y: val.sumY / val.count,
    }));
  }, [points]);

  const handleZoom = (delta: number) => {
    setScale((prev) => Math.max(0.6, Math.min(3.5, prev + delta)));
  };

  const handleReset = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Background Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    const gridSize = 40 * scale;
    const startX = (width / 2 + offset.x) % gridSize;
    const startY = (height / 2 + offset.y) % gridSize;

    for (let x = startX; x < width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = startY; y < height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Coordinate Axes
    const originX = width / 2 + offset.x;
    const originY = height / 2 + offset.y;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, originY);
    ctx.lineTo(width, originY);
    ctx.moveTo(originX, 0);
    ctx.lineTo(originX, height);
    ctx.stroke();

    const spreadFactor = 70 * scale;

    // Draw Points
    points.forEach((p) => {
      const isSelected = selectedCluster === null || selectedCluster === p.cluster;
      const alpha = isSelected ? (hoveredPoint?.id === p.id ? 1.0 : 0.75) : 0.12;
      const baseRadius = (hoveredPoint?.id === p.id ? 6 : 3.5) * Math.min(1.5, Math.max(0.8, scale));

      const px = originX + p.x * spreadFactor;
      const py = originY - p.y * spreadFactor;

      const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length] || '#10b981';

      ctx.beginPath();
      ctx.arc(px, py, baseRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();

      if (hoveredPoint?.id === p.id) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // Draw Centroids
    centroids.forEach((c) => {
      if (selectedCluster !== null && selectedCluster !== c.cluster) return;
      const cx = originX + c.x * spreadFactor;
      const cy = originY - c.y * spreadFactor;
      const color = CLUSTER_COLORS[c.cluster % CLUSTER_COLORS.length];

      // Glow halo
      ctx.beginPath();
      ctx.arc(cx, cy, 14 * scale, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.25;
      ctx.fill();

      // Core centroid star/point
      ctx.beginPath();
      ctx.arc(cx, cy, 7 * scale, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = 1.0;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label badge
      const persona = personas.find((pr) => pr.clusterId === c.cluster);
      const label = persona ? `C${c.cluster}: ${persona.name.split('/')[0].trim()}` : `Cluster ${c.cluster}`;
      ctx.font = 'bold 11px system-ui, sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, cx + 12 * scale, cy - 8 * scale);
    });

    ctx.globalAlpha = 1.0;
  }, [points, personas, selectedCluster, scale, offset, hoveredPoint, centroids]);

  // Mouse Interaction
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
      return;
    }

    // Hit test points
    const originX = canvas.width / 2 + offset.x;
    const originY = canvas.height / 2 + offset.y;
    const spreadFactor = 70 * scale;
    let found: ClusterPoint2D | null = null;

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (selectedCluster !== null && selectedCluster !== p.cluster) continue;
      const px = originX + p.x * spreadFactor;
      const py = originY - p.y * spreadFactor;
      const dist = Math.hypot(mouseX - px, mouseY - py);
      if (dist < 10) {
        found = p;
        break;
      }
    }

    setHoveredPoint(found);
    if (found) {
      setTooltipPos({ x: mouseX + 16, y: mouseY + 16 });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950/80">
      {/* Control bar */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-slate-900/90 p-1.5 rounded-lg border border-slate-800 shadow-lg backdrop-blur-md">
        <Button variant="ghost" size="sm" onClick={() => handleZoom(0.25)} title="Zoom In">
          <ZoomIn className="w-4 h-4 text-slate-300" />
        </Button>
        <Button variant="ghost" size="sm" onClick={() => handleZoom(-0.25)} title="Zoom Out">
          <ZoomOut className="w-4 h-4 text-slate-300" />
        </Button>
        <Button variant="ghost" size="sm" onClick={handleReset} title="Reset View">
          <RotateCcw className="w-4 h-4 text-slate-300" />
        </Button>
      </div>

      <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
        <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-md bg-slate-900/90 border border-slate-800 text-emerald-400">
          {projectionType.toUpperCase()} 2D Embedding
        </span>
        <span className="text-[11px] text-slate-400 bg-slate-900/80 px-2 py-1 rounded border border-slate-800">
          N = {points.length} samples
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={750}
        height={480}
        className="w-full h-[480px] cursor-crosshair block"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setIsDragging(false);
          setHoveredPoint(null);
        }}
      />

      {/* Hover Tooltip Card */}
      {hoveredPoint && (
        <div
          className="absolute z-30 pointer-events-none p-3 rounded-xl bg-slate-900/95 border border-slate-700 shadow-2xl backdrop-blur-md text-xs space-y-1 w-56"
          style={{ left: Math.min(520, tooltipPos.x), top: Math.min(340, tooltipPos.y) }}
        >
          <div className="flex items-center justify-between border-b border-slate-800 pb-1.5 font-semibold">
            <span className="text-white font-mono">{hoveredPoint.id}</span>
            <span
              className="px-1.5 py-0.2 rounded text-[10px] font-bold"
              style={{
                backgroundColor: `${CLUSTER_COLORS[hoveredPoint.cluster % CLUSTER_COLORS.length]}33`,
                color: CLUSTER_COLORS[hoveredPoint.cluster % CLUSTER_COLORS.length],
              }}
            >
              Cluster {hoveredPoint.cluster}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-1 pt-1 text-slate-300">
            <div>Balance: <span className="text-white font-medium">{formatCurrency(hoveredPoint.balance)}</span></div>
            <div>Purchases: <span className="text-white font-medium">{formatCurrency(hoveredPoint.purchases)}</span></div>
            <div>Limit: <span className="text-white font-medium">{formatCurrency(hoveredPoint.creditLimit)}</span></div>
            <div>Cash Adv: <span className="text-white font-medium">{formatCurrency(hoveredPoint.cashAdvance)}</span></div>
          </div>
          <div className="text-[11px] text-emerald-400 pt-1 font-medium">
            {personas.find((p) => p.clusterId === hoveredPoint.cluster)?.name || ''}
          </div>
        </div>
      )}
    </div>
  );
};
