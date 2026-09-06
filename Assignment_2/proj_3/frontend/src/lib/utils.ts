export type ClassValue =
  | string
  | number
  | boolean
  | undefined
  | null
  | { [key: string]: any }
  | ClassValue[];

function toVal(mix: ClassValue): string {
  let str = '';
  if (typeof mix === 'string' || typeof mix === 'number') {
    str += mix;
  } else if (typeof mix === 'object' && mix !== null) {
    if (Array.isArray(mix)) {
      for (let k = 0; k < mix.length; k++) {
        if (mix[k]) {
          const y = toVal(mix[k]);
          if (y) {
            str && (str += ' ');
            str += y;
          }
        }
      }
    } else {
      for (const k in mix) {
        if (mix[k]) {
          str && (str += ' ');
          str += k;
        }
      }
    }
  }
  return str;
}

export function cn(...inputs: ClassValue[]): string {
  let i = 0;
  let tmp: ClassValue;
  let str = '';
  while (i < inputs.length) {
    if ((tmp = inputs[i++])) {
      const x = toVal(tmp);
      if (x) {
        str && (str += ' ');
        str += x;
      }
    }
  }
  return str;
}

export function formatCurrency(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(amount);
}

export function formatNumber(num: number, decimals: number = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(num);
}

export function formatPercent(val: number, decimals: number = 1): string {
  return `${(val * 100).toFixed(decimals)}%`;
}

export function generateLatexTable(data: any[]): string {
  const headers = `\\begin{table*}[t]
\\centering
\\caption{Comparative Evaluation Across Clustering Paradigms on Kaggle Credit Card Benchmark (10-Fold CV)}
\\label{tab:clustering_benchmarks}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Algorithm} & \\textbf{Paradigm} & \\textbf{Config} & \\textbf{Silhouette} $\\uparrow$ & \\textbf{Davies-Bouldin} $\\downarrow$ & \\textbf{Calinski-Harabasz} $\\uparrow$ & \\textbf{Time (ms)} \\\\
\\hline`;

  const rows = data
    .map(
      (r) =>
        `${r.algorithm} & ${r.paradigm} & ${r.kOrEps} & ${r.silhouetteMean.toFixed(3)} $\\pm$ ${r.silhouetteStd.toFixed(3)}${r.isOptimized ? '^*' : ''} & ${r.daviesBouldinMean.toFixed(3)} $\\pm$ ${r.daviesBouldinStd.toFixed(3)} & ${r.calinskiHarabaszMean.toFixed(1)} $\\pm$ ${r.calinskiHarabaszStd.toFixed(1)} & ${r.runtimeMs.toFixed(1)} \\\\`
    )
    .join('\n');

  const footer = `\\hline
\\end{tabular}%
}
\\vspace{1mm}
\\footnotesize{$^*$ Indicates configuration optimized via autonomous hill-climbing search loop.}
\\end{table*}`;

  return `${headers}\n${rows}\n${footer}`;
}

export function generateMarkdownTable(data: any[]): string {
  const header = `| Algorithm | Paradigm | Parameters | Silhouette (↑) | Davies-Bouldin (↓) | Calinski-Harabasz (↑) | Stability ARI | Latency (ms) |\n|---|---|---|---|---|---|---|---|`;
  const rows = data
    .map(
      (r) =>
        `| **${r.algorithm}** | ${r.paradigm} | \`${r.kOrEps}\` | **${r.silhouetteMean.toFixed(3)} ± ${r.silhouetteStd.toFixed(3)}** | ${r.daviesBouldinMean.toFixed(3)} ± ${r.daviesBouldinStd.toFixed(3)} | ${r.calinskiHarabaszMean.toFixed(1)} ± ${r.calinskiHarabaszStd.toFixed(1)} | ${(r.stabilityScore * 100).toFixed(1)}% | ${r.runtimeMs.toFixed(1)}ms |`
    )
    .join('\n');
  return `${header}\n${rows}`;
}
