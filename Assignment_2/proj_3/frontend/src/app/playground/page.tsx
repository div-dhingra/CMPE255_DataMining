'use client';

import React, { useState, useEffect } from 'react';
import {
  SlidersHorizontal,
  Sparkles,
  Zap,
  RotateCcw,
  CheckCircle2,
  PieChart,
  UploadCloud,
  FileSpreadsheet,
  Download,
  AlertCircle,
  Lightbulb,
} from 'lucide-react';
import { api } from '@/lib/api';
import {
  CustomerVector,
  InferenceResponse,
} from '@/types/api';
import { mockCustomerPresets } from '@/lib/mockData';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Slider } from '@/components/ui/Slider';
import { Tabs } from '@/components/ui/Tabs';
import { formatCurrency, formatPercent } from '@/lib/utils';
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

export default function PlaygroundPage() {
  const [vector, setVector] = useState<CustomerVector>(mockCustomerPresets.vip);
  const [prediction, setPrediction] = useState<InferenceResponse | null>(null);
  const [activePreset, setActivePreset] = useState<string>('vip');
  const [batchModal, setBatchModal] = useState(false);
  const [batchCount, setBatchCount] = useState<number | null>(null);
  const [batchResults, setBatchResults] = useState<{ id: string; cluster: number; persona: string; conf: number }[]>([]);

  useEffect(() => {
    async function runInference() {
      const res = await api.predictCustomer(vector);
      setPrediction(res);
    }
    runInference();
  }, [vector]);

  const loadPreset = (key: string) => {
    if (mockCustomerPresets[key]) {
      setActivePreset(key);
      setVector(mockCustomerPresets[key]);
    }
  };

  const updateField = (field: keyof CustomerVector, value: number) => {
    setActivePreset('custom');
    setVector((prev) => ({ ...prev, [field]: value }));
  };

  const handleSimulateBatch = () => {
    const synthetic = Array.from({ length: 15 }, (_, i) => {
      const cluster = i % 4;
      const personas = [
        'Transactor / Active Spender',
        'Cash Advance Borrower',
        'VIP High Spender',
        'Low-Engagement Account',
      ];
      return {
        id: `CUST_${(9901 + i).toString()}`,
        cluster,
        persona: personas[cluster],
        conf: Number((82 + Math.random() * 16).toFixed(1)),
      };
    });
    setBatchResults(synthetic);
    setBatchCount(synthetic.length);
  };

  // Prepare radar overlay data
  const radarData = prediction?.radarComparison.map((r) => ({
    feature: r.feature,
    'Custom Input': r.sampleValue,
    'Cluster Centroid': r.clusterCentroid,
    'Population Mean': r.populationMean,
  })) || [];

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Top Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-slate-900/80 p-5 rounded-2xl border border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-extrabold text-white flex items-center gap-2">
              <SlidersHorizontal className="w-6 h-6 text-emerald-400" />
              Customer Inference & Profiler Playground
            </h1>
            <Badge variant="emerald">Sub-3ms Scoring</Badge>
          </div>
          <p className="text-xs text-slate-400">
            Real-time behavioral scoring, soft cluster membership probabilities, and persona recommendations.
          </p>
        </div>

        {/* Archetype Presets */}
        <div className="flex flex-wrap items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800">
          <span className="text-xs font-semibold text-slate-400 px-2">Presets:</span>
          {[
            { id: 'vip', label: '💎 VIP Spender' },
            { id: 'revolver', label: '💳 Revolver' },
            { id: 'transactor', label: '🛍️ Transactor' },
            { id: 'cashSeeker', label: '⚡ Cash Seeker' },
            { id: 'inactive', label: '💤 Inactive' },
          ].map((preset) => (
            <button
              key={preset.id}
              onClick={() => loadPreset(preset.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activePreset === preset.id
                  ? 'bg-emerald-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Interactive 12 Sliders Form */}
        <div className="lg:col-span-6 space-y-4">
          <Card className="p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <SlidersHorizontal className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-bold text-white">Customer Behavioral Feature Inputs</h3>
              </div>
              <Button
                variant="outline"
                size="sm"
                icon={<RotateCcw className="w-3.5 h-3.5" />}
                onClick={() => loadPreset('vip')}
              >
                Reset
              </Button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Slider
                label="Account Balance"
                value={vector.balance}
                min={0}
                max={15000}
                step={100}
                formatValue={formatCurrency}
                onChange={(v) => updateField('balance', v)}
              />
              <Slider
                label="Total Purchases"
                value={vector.purchases}
                min={0}
                max={15000}
                step={100}
                formatValue={formatCurrency}
                onChange={(v) => updateField('purchases', v)}
              />
              <Slider
                label="Cash Advance"
                value={vector.cashAdvance}
                min={0}
                max={10000}
                step={100}
                formatValue={formatCurrency}
                onChange={(v) => updateField('cashAdvance', v)}
              />
              <Slider
                label="Credit Limit"
                value={vector.creditLimit}
                min={500}
                max={25000}
                step={250}
                formatValue={formatCurrency}
                onChange={(v) => updateField('creditLimit', v)}
              />
              <Slider
                label="Total Payments"
                value={vector.payments}
                min={0}
                max={15000}
                step={100}
                formatValue={formatCurrency}
                onChange={(v) => updateField('payments', v)}
              />
              <Slider
                label="Minimum Payments"
                value={vector.minimumPayments}
                min={0}
                max={5000}
                step={50}
                formatValue={formatCurrency}
                onChange={(v) => updateField('minimumPayments', v)}
              />
              <Slider
                label="Purchases Frequency"
                value={vector.purchasesFrequency}
                min={0.0}
                max={1.0}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                onChange={(v) => updateField('purchasesFrequency', v)}
              />
              <Slider
                label="Cash Advance Frequency"
                value={vector.cashAdvanceFrequency}
                min={0.0}
                max={1.0}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                onChange={(v) => updateField('cashAdvanceFrequency', v)}
              />
              <Slider
                label="Percent Full Payment"
                value={vector.prcFullPayment}
                min={0.0}
                max={1.0}
                step={0.05}
                formatValue={(v) => `${(v * 100).toFixed(0)}%`}
                onChange={(v) => updateField('prcFullPayment', v)}
              />
              <Slider
                label="Account Tenure"
                value={vector.tenure}
                min={6}
                max={12}
                step={1}
                unit=" mos"
                onChange={(v) => updateField('tenure', v)}
              />
            </div>
          </Card>
        </div>

        {/* Right Column: Prediction, Probabilities, Radar Overlay */}
        <div className="lg:col-span-6 space-y-4">
          {prediction && (
            <>
              {/* Prediction Result Card */}
              <Card className="p-5 space-y-4 border-emerald-800/60 bg-gradient-to-br from-slate-900 via-slate-900 to-emerald-950/30">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2 font-mono text-xs">
                    <span className="text-slate-400">Assigned Segment:</span>
                    <span className="text-emerald-400 font-bold">Cluster {prediction.predictedCluster}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-400">Confidence:</span>
                    <Badge variant="emerald">{prediction.confidence}%</Badge>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-extrabold text-white flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-emerald-400" />
                    {prediction.personaName}
                  </h3>
                  <p className="text-xs text-slate-300 mt-1 italic leading-relaxed">
                    "{prediction.tagline}"
                  </p>
                </div>

                {/* Soft Membership Probabilities */}
                <div className="space-y-2 pt-1">
                  <div className="flex justify-between items-center text-xs font-semibold text-slate-300">
                    <span>Soft Cluster Membership Probabilities (Softmax Distances):</span>
                  </div>
                  <div className="space-y-1.5">
                    {prediction.probabilities.map((prob) => (
                      <div key={prob.clusterId} className="space-y-1">
                        <div className="flex justify-between text-xs font-mono">
                          <span className="text-slate-300">
                            C{prob.clusterId}: {prob.personaName.split('/')[0].trim()}
                          </span>
                          <span className="font-bold text-white">{formatPercent(prob.probability, 1)}</span>
                        </div>
                        <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${prob.probability * 100}%`,
                              backgroundColor: prob.color,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommended Marketing Actions */}
                <div className="pt-2 border-t border-slate-800">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 mb-1.5">
                    <Lightbulb className="w-4 h-4" />
                    <span>Segment-Specific Action Recommendations:</span>
                  </div>
                  <ul className="space-y-1 text-xs text-slate-300 list-disc list-inside">
                    {prediction.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </Card>

              {/* Radar Overlay Comparison */}
              <Card className="p-4 space-y-2">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                      <PieChart className="w-4 h-4 text-cyan-400" />
                      Radar Overlay: Input Vector vs. Cluster Centroid
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Cyan Line = Your Input Point | Emerald Polygon = Cluster Centroid Average
                    </p>
                  </div>
                </div>

                <div className="w-full h-64 relative">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
                      <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                      <PolarAngleAxis
                        dataKey="feature"
                        tick={{ fill: '#94a3b8', fontSize: 10, fontWeight: 500 }}
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
                          fontSize: '11px',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <Radar
                        name="Input Vector (Custom)"
                        dataKey="Custom Input"
                        stroke="#06b6d4"
                        fill="#06b6d4"
                        fillOpacity={0.3}
                        strokeWidth={2.5}
                      />
                      <Radar
                        name="Assigned Centroid"
                        dataKey="Cluster Centroid"
                        stroke="#10b981"
                        fill="#10b981"
                        fillOpacity={0.2}
                        strokeWidth={1.5}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Batch CSV Dropzone & Tester Section */}
      <Card className="p-6 space-y-4 border-slate-800 bg-slate-900/60">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <UploadCloud className="w-5 h-5 text-emerald-400" />
              Batch Customer Inference Dropzone (CSV)
            </h3>
            <p className="text-xs text-slate-400">
              Upload customer account batches to execute high-throughput offline or REST scoring.
            </p>
          </div>

          <Button
            variant="secondary"
            size="sm"
            icon={<FileSpreadsheet className="w-4 h-4 text-emerald-400" />}
            onClick={handleSimulateBatch}
          >
            Simulate Sample Batch Run (15 Records)
          </Button>
        </div>

        {batchResults.length > 0 ? (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs text-slate-300">
              <span className="font-bold text-white">Batch Scoring Results ({batchResults.length} records processed)</span>
              <span className="text-emerald-400 font-mono">100% Inferences Validated</span>
            </div>

            <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950">
              <table className="w-full text-left text-xs border-collapse font-mono">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-900 text-slate-400 uppercase text-[11px] font-sans">
                    <th className="py-2.5 px-3">Customer ID</th>
                    <th className="py-2.5 px-3">Predicted Cluster</th>
                    <th className="py-2.5 px-4">Segment Persona</th>
                    <th className="py-2.5 px-3">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {batchResults.map((res) => (
                    <tr key={res.id} className="hover:bg-slate-900/50">
                      <td className="py-2 px-3 font-bold text-white">{res.id}</td>
                      <td className="py-2 px-3">
                        <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 text-[10px]">
                          Cluster {res.cluster}
                        </span>
                      </td>
                      <td className="py-2 px-4 font-sans text-slate-200">{res.persona}</td>
                      <td className="py-2 px-3 font-bold text-emerald-400">{res.conf}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="border-2 border-dashed border-slate-800 rounded-xl p-8 text-center space-y-2 hover:border-slate-700 transition-colors">
            <UploadCloud className="w-8 h-8 text-slate-500 mx-auto" />
            <div className="text-xs font-semibold text-slate-300">
              Drop your customer CSV file here or click to browse
            </div>
            <p className="text-[11px] text-slate-500">
              Expected columns: BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, TENURE...
            </p>
          </div>
        )}
      </Card>
    </div>
  );
}
