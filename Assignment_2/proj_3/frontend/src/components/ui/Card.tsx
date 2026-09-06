import React from 'react';
import { cn } from '@/lib/utils';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  glass?: boolean;
  glow?: 'emerald' | 'cyan' | 'violet' | 'amber' | 'none';
}

export const Card: React.FC<CardProps> = ({
  className,
  children,
  glass = true,
  glow = 'none',
  ...props
}) => {
  const glowClasses: Record<string, string> = {
    emerald: 'glow-emerald',
    cyan: 'glow-cyan',
    violet: 'glow-violet',
    amber: 'shadow-[0_0_20px_-3px_rgba(245,158,11,0.35)]',
    none: '',
  };

  return (
    <div
      className={cn(
        'rounded-xl p-5 border border-slate-800 transition-all duration-200',
        glass ? 'glass-card' : 'bg-slate-900',
        glowClasses[glow] || '',
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
