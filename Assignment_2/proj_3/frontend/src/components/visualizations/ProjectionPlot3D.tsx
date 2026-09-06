'use client';

import React, { useRef, useEffect, useState } from 'react';
import { ClusterPoint3D, ClusterPersona } from '@/types/api';
import { Button } from '@/components/ui/Button';
import { Play, Pause, RotateCcw } from 'lucide-react';

interface ProjectionPlot3DProps {
  points: ClusterPoint3D[];
  personas: ClusterPersona[];
  selectedCluster: number | null;
}

const CLUSTER_COLORS = ['#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#f43f5e'];

export const ProjectionPlot3D: React.FC<ProjectionPlot3DProps> = ({
  points,
  personas,
  selectedCluster,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [rotX, setRotX] = useState(0.4);
  const [rotY, setRotY] = useState(0.6);
  const [autoRotate, setAutoRotate] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // 3D Orbital Projection Math
  useEffect(() => {
    let animId: number;

    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const fov = 400;

      // Project all points with 3D rotation matrix
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);

      // Draw bounding box wireframe
      const boxSize = 2.5;
      const boxVerts = [
        [-boxSize, -boxSize, -boxSize], [boxSize, -boxSize, -boxSize],
        [boxSize, boxSize, -boxSize], [-boxSize, boxSize, -boxSize],
        [-boxSize, -boxSize, boxSize], [boxSize, -boxSize, boxSize],
        [boxSize, boxSize, boxSize], [-boxSize, boxSize, boxSize],
      ];

      const projectedVerts = boxVerts.map(([vx, vy, vz]) => {
        // Rotate Y
        const x1 = vx * cosY + vz * sinY;
        const z1 = -vx * sinY + vz * cosY;
        // Rotate X
        const y2 = vy * cosX - z1 * sinX;
        const z2 = vy * sinX + z1 * cosX + 6.0; // Camera distance
        const pScale = fov / z2;
        return { x: cx + x1 * pScale, y: cy + y2 * pScale, z: z2 };
      });

      const edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
      ];

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      edges.forEach(([i, j]) => {
        ctx.beginPath();
        ctx.moveTo(projectedVerts[i].x, projectedVerts[i].y);
        ctx.lineTo(projectedVerts[j].x, projectedVerts[j].y);
        ctx.stroke();
      });

      // Project data points and depth sort (painter's algorithm)
      const projectedPoints = points
        .map((p) => {
          // Normalize coordinates
          const px = p.x * 0.8;
          const py = p.y * 0.8;
          const pz = p.z * 0.8;

          // Rotate Y
          const x1 = px * cosY + pz * sinY;
          const z1 = -px * sinY + pz * cosY;
          // Rotate X
          const y2 = py * cosX - z1 * sinX;
          const z2 = py * sinX + z1 * cosX + 6.0;

          const pScale = fov / z2;
          return {
            ...p,
            projX: cx + x1 * pScale,
            projY: cy + y2 * pScale,
            depth: z2,
            scale: pScale,
          };
        })
        .sort((a, b) => b.depth - a.depth); // Far to near

      projectedPoints.forEach((p) => {
        const isSelected = selectedCluster === null || selectedCluster === p.cluster;
        if (!isSelected) return;

        const alpha = Math.max(0.15, Math.min(1.0, 1.2 - p.depth / 8));
        const radius = Math.max(1.5, 4.0 * (fov / (p.depth * 100)));
        const color = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];

        ctx.beginPath();
        ctx.arc(p.projX, p.projY, radius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = alpha;
        ctx.fill();
      });

      ctx.globalAlpha = 1.0;

      if (autoRotate && !isDragging) {
        setRotY((prev) => prev + 0.008);
      }
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [points, personas, selectedCluster, rotX, rotY, autoRotate, isDragging]);

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setRotY((prev) => prev + dx * 0.008);
    setRotX((prev) => Math.max(-1.2, Math.min(1.2, prev + dy * 0.008)));
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <div className="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950/80">
      <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 bg-slate-900/90 p-1.5 rounded-lg border border-slate-800 shadow-lg backdrop-blur-md">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setAutoRotate(!autoRotate)}
          title={autoRotate ? 'Pause Rotation' : 'Auto Rotate'}
        >
          {autoRotate ? <Pause className="w-4 h-4 text-emerald-400" /> : <Play className="w-4 h-4 text-slate-300" />}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setRotX(0.4);
            setRotY(0.6);
          }}
          title="Reset Orbit"
        >
          <RotateCcw className="w-4 h-4 text-slate-300" />
        </Button>
      </div>

      <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
        <span className="text-xs font-mono font-semibold px-2.5 py-1 rounded-md bg-slate-900/90 border border-slate-800 text-cyan-400">
          3D PCA ORBITAL EMBEDDING (PC1, PC2, PC3)
        </span>
        <span className="text-[11px] text-slate-400 bg-slate-900/80 px-2 py-1 rounded border border-slate-800">
          Drag to rotate orbit
        </span>
      </div>

      <canvas
        ref={canvasRef}
        width={750}
        height={480}
        className="w-full h-[480px] cursor-grab active:cursor-grabbing block"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
    </div>
  );
};
