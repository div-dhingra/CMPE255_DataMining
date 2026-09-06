import React from 'react';
import { cn } from '@/lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'emerald' | 'cyan' | 'violet' | 'amber' | 'rose' | 'slate' | 'outline';
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'emerald',
  size = 'sm',
  children,
  ...props
}) => {
  const variantStyles: Record<string, string> = {
    emerald: 'bg-emerald-950/70 text-emerald-300 border-emerald-700/50',
    cyan: 'bg-cyan-950/70 text-cyan-300 border-cyan-700/50',
    violet: 'bg-violet-950/70 text-violet-300 border-violet-700/50',
    amber: 'bg-amber-950/70 text-amber-300 border-amber-700/50',
    rose: 'bg-rose-950/70 text-rose-300 border-rose-700/50',
    slate: 'bg-slate-800/80 text-slate-300 border-slate-700/50',
    outline: 'bg-transparent text-slate-300 border-slate-700',
  };

  const sizeStyles: Record<string, string> = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-xs font-medium',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border font-medium tracking-wide shadow-sm',
        variantStyles[variant] || variantStyles.emerald,
        sizeStyles[size] || sizeStyles.sm,
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
};
