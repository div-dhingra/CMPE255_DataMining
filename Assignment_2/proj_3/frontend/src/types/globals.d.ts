declare namespace React {
  export type ReactNode = any;
  export type ReactElement<P = any, T = any> = any;
  export type Key = string | number;
  export type FC<P = {}> = (props: P) => any;
  export type FunctionComponent<P = {}> = (props: P) => any;
  export class Component<P = {}, S = {}> {
    props: P;
    state: S;
    constructor(props: P);
    render(): any;
  }
  export function useState<T>(initialState: T | (() => T)): [T, (value: T | ((prev: T) => T)) => void];
  export function useEffect(effect: () => any, deps?: readonly any[]): void;
  export function useRef<T>(initialValue: T): { current: T };
  export function useMemo<T>(factory: () => T, deps: readonly any[] | undefined): T;
  export function useCallback<T extends (...args: any[]) => any>(callback: T, deps: readonly any[]): T;

  export interface HTMLAttributes<T = any> {
    className?: string;
    style?: any;
    title?: string;
    id?: string;
    children?: any;
    onClick?: (event: any) => void;
    onMouseDown?: (event: any) => void;
    onMouseMove?: (event: any) => void;
    onMouseUp?: (event: any) => void;
    onMouseLeave?: (event: any) => void;
    [key: string]: any;
  }
  export interface ButtonHTMLAttributes<T = any> extends HTMLAttributes<T> {
    disabled?: boolean;
    type?: 'submit' | 'reset' | 'button';
  }
  export interface MouseEvent<T = any> {
    clientX: number;
    clientY: number;
    preventDefault(): void;
    stopPropagation(): void;
    currentTarget: T;
    target: any;
  }
  export interface KeyboardEvent<T = any> {
    key: string;
    keyCode: number;
    preventDefault(): void;
    stopPropagation(): void;
  }
  export interface ChangeEvent<T = any> {
    target: T & { value: string };
  }
  export const Fragment: any;
}

declare module 'react' {
  export = React;
}

declare namespace JSX {
  interface Element extends any {}
  interface ElementClass extends any {}
  interface ElementAttributesProperty { props: {}; }
  interface ElementChildrenAttribute { children: {}; }
  interface IntrinsicElements {
    [elemName: string]: any;
  }
}

declare module 'next' {
  export type Metadata = {
    title?: string;
    description?: string;
    [key: string]: any;
  };
}

declare module 'next/link' {
  const Link: any;
  export default Link;
}

declare module 'next/navigation' {
  export function usePathname(): string;
  export function useRouter(): any;
}

declare module 'lucide-react' {
  export const Activity: any;
  export const Layers: any;
  export const Cpu: any;
  export const TableProperties: any;
  export const Sparkles: any;
  export const Server: any;
  export const RefreshCw: any;
  export const SlidersHorizontal: any;
  export const CheckCircle2: any;
  export const Database: any;
  export const ArrowRight: any;
  export const TrendingUp: any;
  export const AlertTriangle: any;
  export const FileSpreadsheet: any;
  export const Search: any;
  export const BarChart2: any;
  export const ZoomIn: any;
  export const ZoomOut: any;
  export const RotateCcw: any;
  export const Play: any;
  export const Pause: any;
  export const FastForward: any;
  export const Flame: any;
  export const XCircle: any;
  export const Award: any;
  export const Zap: any;
  export const Sliders: any;
  export const ShieldCheck: any;
  export const BookOpen: any;
  export const Copy: any;
  export const Check: any;
  export const Download: any;
  export const FileCode: any;
  export const Lightbulb: any;
  export const Info: any;
  export const Maximize2: any;
  export const PieChart: any;
  export const UploadCloud: any;
  export const AlertCircle: any;
  export const X: any;
  export const GitBranch: any;
}

declare module 'recharts' {
  export const ResponsiveContainer: any;
  export const RadarChart: any;
  export const PolarGrid: any;
  export const PolarAngleAxis: any;
  export const PolarRadiusAxis: any;
  export const Radar: any;
  export const Legend: any;
  export const Tooltip: any;
  export const ComposedChart: any;
  export const Line: any;
  export const Area: any;
  export const XAxis: any;
  export const YAxis: any;
  export const CartesianGrid: any;
  export const ReferenceDot: any;
}

declare module 'tailwindcss' {
  export type Config = any;
}
