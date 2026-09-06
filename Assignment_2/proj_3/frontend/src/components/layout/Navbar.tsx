'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Activity,
  Layers,
  Cpu,
  TableProperties,
  Sparkles,
  Server,
  RefreshCw,
  SlidersHorizontal,
} from 'lucide-react';
import { api } from '@/lib/api';
import { SystemHealthStatus } from '@/types/api';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [health, setHealth] = useState<SystemHealthStatus>({
    online: false,
    latencyMs: 0,
    version: '1.0.0',
    backendUrl: 'http://localhost:8000/api/v1',
    mode: 'mock_fallback',
    modelCount: 6,
    datasetLoaded: true,
  });
  const [isChecking, setIsChecking] = useState(false);

  const checkConnection = async () => {
    setIsChecking(true);
    const status = await api.checkHealth();
    setHealth(status);
    setIsChecking(false);
  };

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    {
      label: 'CRISP-DM Flow',
      href: '/overview',
      icon: <Activity className="w-4 h-4" />,
      badge: '6 Phases',
    },
    {
      label: 'Cluster Explorer',
      href: '/clusters',
      icon: <Layers className="w-4 h-4" />,
      badge: '2D/3D',
    },
    {
      label: 'Autoresearch Studio',
      href: '/autoresearch',
      icon: <Cpu className="w-4 h-4" />,
      badge: 'Live',
    },
    {
      label: 'Benchmark Matrix',
      href: '/benchmarks',
      icon: <TableProperties className="w-4 h-4" />,
      badge: 'LaTeX',
    },
    {
      label: 'Inference Playground',
      href: '/playground',
      icon: <SlidersHorizontal className="w-4 h-4" />,
      badge: 'Real-Time',
    },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-slate-950/85 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo & Title */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-950/50 group-hover:scale-105 transition-transform">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-white tracking-wide">CRISP-DM CLUSTERING</span>
                  <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60 font-semibold">
                    Autoresearch AI
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">Credit Card Customer Segmentation Admin</p>
              </div>
            </Link>
          </div>

          {/* Nav Links */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive =
                pathname === item.href || (item.href === '/overview' && pathname === '/');
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    'flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-all duration-150',
                    isActive
                      ? 'bg-emerald-950/70 text-emerald-300 border border-emerald-700/50 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent'
                  )}
                >
                  <span className={isActive ? 'text-emerald-400' : 'text-slate-400'}>{item.icon}</span>
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* System Status Badge & Actions */}
          <div className="flex items-center gap-3">
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
              <span
                className={cn(
                  'w-2.5 h-2.5 rounded-full',
                  health.online ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                )}
              />
              <div className="flex items-center gap-1.5 font-mono">
                <span className="text-slate-400">Backend:</span>
                <span className={health.online ? 'text-emerald-400 font-medium' : 'text-amber-400 font-medium'}>
                  {health.online ? `Online (${health.latencyMs}ms)` : 'Mock Engine (Offline)'}
                </span>
              </div>
              <button
                onClick={checkConnection}
                className="text-slate-500 hover:text-slate-300 transition-colors p-0.5 ml-1"
                title="Re-check Backend Connection"
              >
                <RefreshCw className={cn('w-3 h-3', isChecking && 'animate-spin text-emerald-400')} />
              </button>
            </div>

            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="hidden lg:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-300 border border-slate-800 text-xs transition-colors"
            >
              <Server className="w-3.5 h-3.5 text-cyan-400" />
              <span>Swagger API</span>
            </a>
          </div>
        </div>
      </div>

      {/* Fallback Banner if Offline */}
      {!health.online && (
        <div className="w-full bg-amber-950/40 border-t border-b border-amber-800/40 px-4 py-1 text-center text-xs text-amber-300 flex items-center justify-center gap-2">
          <span>⚡ <strong>Offline Resilience Active:</strong> Dashboard is operating in high-fidelity mock simulation mode with deterministic 2D/3D projections and client-side inference.</span>
        </div>
      )}
    </header>
  );
};
