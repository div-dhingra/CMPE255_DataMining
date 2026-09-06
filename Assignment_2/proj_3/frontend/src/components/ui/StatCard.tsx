import React from 'react';
import { Card } from './Card';
import { Badge } from './Badge';
import { cn } from '@/lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badge?: string;
  badgeVariant?: 'emerald' | 'cyan' | 'violet' | 'amber' | 'rose' | 'slate';
  icon?: React.ReactNode;
  trend?: string;
  className?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  badge,
  badgeVariant = 'emerald',
  icon,
  trend,
  className,
}) => {
  return (
    <Card className={cn('flex flex-col justify-between', className)}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        {icon && <div className="text-slate-400 bg-slate-800/80 p-1.5 rounded-lg border border-slate-700/50">{icon}</div>}
      </div>

      <div className="mt-3">
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold tracking-tight text-white">{value}</span>
          {badge && <Badge variant={badgeVariant}>{badge}</Badge>}
        </div>
        {(subtitle || trend) && (
          <p className="mt-1 text-xs text-slate-400 flex items-center gap-1.5">
            {trend && <span className="text-emerald-400 font-medium">{trend}</span>}
            {subtitle}
          </p>
        )}
      </div>
    </Card>
  );
};
