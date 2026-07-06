import { useState } from 'react'
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts'
import { Zap, Shield, Cpu, ArrowUpRight } from 'lucide-react'

interface ModelMeta {
  format: string
  size: string
  sizeBytes: number
  tokenMatch: number
  bleu: number
  latency: number
  speedup: number
}

interface DirectionMeta {
  name: string
  scale200: ModelMeta[]
  scale1b: ModelMeta[]
}

const summaryData: DirectionMeta[] = [
  {
    name: 'English to Indic (en-indic)',
    scale200: [
      { format: 'FP32', size: '1.06 GB', sizeBytes: 1.06, tokenMatch: 100.0, bleu: 100.0, latency: 18.3, speedup: 1.0 },
      { format: 'FP16', size: '559.6 MB', sizeBytes: 0.55, tokenMatch: 99.64, bleu: 100.0, latency: 24.8, speedup: 0.74 },
      { format: 'INT8', size: '302.7 MB', sizeBytes: 0.30, tokenMatch: 74.36, bleu: 90.44, latency: 13.2, speedup: 1.59 },
      { format: 'Q4F16', size: '380.6 MB', sizeBytes: 0.37, tokenMatch: 55.18, bleu: 81.13, latency: 27.3, speedup: 0.71 }
    ],
    scale1b: [
      { format: 'FP32', size: '4.19 GB', sizeBytes: 4.19, tokenMatch: 100.0, bleu: 100.0, latency: 69.5, speedup: 1.0 },
      { format: 'FP16', size: '2.11 GB', sizeBytes: 2.11, tokenMatch: 99.73, bleu: 100.0, latency: 74.3, speedup: 0.94 },
      { format: 'INT8', size: '1.08 GB', sizeBytes: 1.08, tokenMatch: 89.55, bleu: 96.27, latency: 31.4, speedup: 2.12 },
      { format: 'Q4F16', size: '1.01 GB', sizeBytes: 1.01, tokenMatch: 82.45, bleu: 91.99, latency: 58.4, speedup: 1.19 }
    ]
  },
  {
    name: 'Indic to English (indic-en)',
    scale200: [
      { format: 'FP32', size: '907.2 MB', sizeBytes: 0.89, tokenMatch: 100.0, bleu: 100.0, latency: 12.2, speedup: 1.0 },
      { format: 'FP16', size: '471.4 MB', sizeBytes: 0.46, tokenMatch: 99.91, bleu: 99.96, latency: 14.3, speedup: 0.85 },
      { format: 'INT8', size: '257.3 MB', sizeBytes: 0.25, tokenMatch: 85.64, bleu: 94.34, latency: 10.1, speedup: 1.17 },
      { format: 'Q4F16', size: '292.4 MB', sizeBytes: 0.29, tokenMatch: 73.36, bleu: 88.31, latency: 15.8, speedup: 0.74 }
    ],
    scale1b: [
      { format: 'FP32', size: '3.85 GB', sizeBytes: 3.85, tokenMatch: 100.0, bleu: 100.0, latency: 49.0, speedup: 1.0 },
      { format: 'FP16', size: '1.94 GB', sizeBytes: 1.94, tokenMatch: 99.82, bleu: 99.96, latency: 49.7, speedup: 0.99 },
      { format: 'INT8', size: '1020.1 MB', sizeBytes: 1.00, tokenMatch: 94.45, bleu: 98.00, latency: 25.2, speedup: 1.90 },
      { format: 'Q4F16', size: '861.5 MB', sizeBytes: 0.84, tokenMatch: 88.55, bleu: 95.44, latency: 42.7, speedup: 1.08 }
    ]
  },
  {
    name: 'Indic to Indic (indic-indic)',
    scale200: [
      { format: 'FP32', size: '1.25 GB', sizeBytes: 1.25, tokenMatch: 100.0, bleu: 100.0, latency: 23.0, speedup: 1.0 },
      { format: 'FP16', size: '671.9 MB', sizeBytes: 0.66, tokenMatch: 99.82, bleu: 100.0, latency: 27.4, speedup: 0.84 },
      { format: 'INT8', size: '370.9 MB', sizeBytes: 0.36, tokenMatch: 72.18, bleu: 87.13, latency: 16.5, speedup: 1.48 },
      { format: 'Q4F16', size: '492.9 MB', sizeBytes: 0.48, tokenMatch: 45.91, bleu: 71.64, latency: 28.3, speedup: 0.83 }
    ],
    scale1b: [
      { format: 'FP32', size: '4.56 GB', sizeBytes: 4.56, tokenMatch: 100.0, bleu: 100.0, latency: 94.7, speedup: 1.0 },
      { format: 'FP16', size: '2.31 GB', sizeBytes: 2.31, tokenMatch: 99.82, bleu: 100.0, latency: 108.3, speedup: 0.87 },
      { format: 'INT8', size: '1.19 GB', sizeBytes: 1.19, tokenMatch: 83.64, bleu: 94.22, latency: 43.7, speedup: 2.24 },
      { format: 'Q4F16', size: '1.21 GB', sizeBytes: 1.21, tokenMatch: 73.18, bleu: 89.33, latency: 94.2, speedup: 1.09 }
    ]
  }
]

export function OverviewTab() {
  const [activeDirection, setActiveDirection] = useState(0)
  const [modelScale, setModelScale] = useState<'200M' | '1B'>('200M')

  const dir = summaryData[activeDirection]
  const currentDataset = modelScale === '200M' ? dir.scale200 : dir.scale1b

  // Custom tooltips for chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass border border-white/10 p-3 rounded-lg text-xs space-y-1">
          <p className="font-bold text-zinc-100">{label}</p>
          <p className="text-emerald-400">BLEU Parity: {payload[0].value}%</p>
          <p className="text-teal-400">Model Size: {payload[1].value} GB</p>
          {payload[2] && <p className="text-amber-400">Speedup: {payload[2].value}x</p>}
        </div>
      )
    }
    return null
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Title section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-zinc-100">Overview Dashboard</h2>
          <p className="text-sm text-zinc-400">Compare file size, latency, and quality tradeoffs across different quantizations.</p>
        </div>
        <div className="flex items-center gap-2 bg-zinc-900/60 p-1 border border-white/5 rounded-lg self-start">
          <button
            onClick={() => setModelScale('200M')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${modelScale === '200M' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
              }`}
          >
            Base (200M/320M)
          </button>
          <button
            onClick={() => setModelScale('1B')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-all ${modelScale === '1B' ? 'bg-white/10 text-white shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
              }`}
          >
            Large (1B)
          </button>
        </div>
      </div>

      {/* Metric summary grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Max Speedup</span>
            <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
              <Zap size={16} />
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-extrabold text-zinc-100">
              {modelScale === '200M' ? '1.59x' : '2.24x'}
            </div>
            <p className="text-xs text-zinc-400">Achieved on INT8 quantization formats.</p>
          </div>
        </div>

        <div className="glass-card p-6 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">FP16 Accuracy Parity</span>
            <div className="p-2 bg-teal-500/10 text-teal-400 rounded-lg">
              <Shield size={16} />
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-extrabold text-zinc-100">&gt; 99.6%</div>
            <p className="text-xs text-zinc-400">Exact token-match vs the FP32 PyTorch Oracle.</p>
          </div>
        </div>

        <div className="glass-card p-6 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold tracking-wider text-zinc-400 uppercase">Footprint Savings</span>
            <div className="p-2 bg-teal-500/10 text-teal-400 rounded-lg">
              <Cpu size={16} />
            </div>
          </div>
          <div className="space-y-1">
            <div className="text-3xl font-extrabold text-zinc-100">Up to 72%</div>
            <p className="text-xs text-zinc-400">File size reduced from 1.06 GB to 302.7 MB (INT8, en→indic).</p>
          </div>
        </div>
      </div>

      {/* Tabs and Graph layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left selector */}
        <div className="lg:col-span-1 flex flex-col gap-2">
          <span className="text-xs font-semibold text-zinc-500 tracking-wider uppercase mb-1">Translation Direction</span>
          {summaryData.map((d, index) => (
            <button
              key={d.name}
              onClick={() => setActiveDirection(index)}
              className={`text-left p-4 rounded-xl transition-all border ${activeDirection === index
                  ? 'bg-white/5 border-white/10 text-zinc-100 shadow'
                  : 'bg-transparent border-transparent text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
                }`}
            >
              <div className="text-sm font-bold">{d.name.split(' (')[0]}</div>
              <div className="text-xs text-zinc-500 font-mono mt-1">{d.name.split(' (')[1].replace(')', '')}</div>
            </button>
          ))}
        </div>

        {/* Right chart/table visualization */}
        <div className="lg:col-span-3 glass-card p-6 rounded-xl space-y-6">
          <div>
            <h3 className="text-lg font-bold text-zinc-100">{dir.name} Tradeoff Analysis</h3>
            <p className="text-xs text-zinc-400">Comparing SacreBLEU relative parity (Bar), Model Size in GB (Bar), and relative speedup (Line).</p>
          </div>

          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={currentDataset}
                margin={{ top: 20, right: 20, bottom: 20, left: 0 }}
              >
                <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                <XAxis dataKey="format" stroke="#a1a1aa" fontSize={11} tickLine={false} />
                <YAxis yAxisId="left" stroke="#a1a1aa" fontSize={11} label={{ value: 'Parity / Size', angle: -90, position: 'insideLeft', fill: '#a1a1aa', style: { textAnchor: 'middle' } }} />
                <YAxis yAxisId="right" orientation="right" stroke="#a1a1aa" fontSize={11} label={{ value: 'Speedup Factor', angle: 90, position: 'insideRight', fill: '#a1a1aa', style: { textAnchor: 'middle' } }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#a1a1aa' }} />

                <Bar yAxisId="left" dataKey="bleu" name="BLEU Parity (%)" fill="#0d9488" barSize={35} radius={[4, 4, 0, 0]} />
                <Bar yAxisId="left" dataKey="sizeBytes" name="Size (GB)" fill="#0f766e" barSize={35} radius={[4, 4, 0, 0]} opacity={0.6} />
                <Line yAxisId="right" type="monotone" dataKey="speedup" name="Speedup vs FP32" stroke="#f59e0b" strokeWidth={3} activeDot={{ r: 6 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Data table */}
          <div className="overflow-x-auto border border-white/5 rounded-lg">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="bg-zinc-900/60 text-zinc-400 border-b border-white/5 font-semibold">
                  <th className="p-3">Format</th>
                  <th className="p-3">Model Size</th>
                  <th className="p-3 text-right">Exact Match (Token)</th>
                  <th className="p-3 text-right">SacreBLEU Parity</th>
                  <th className="p-3 text-right">Avg Latency</th>
                  <th className="p-3 text-right">Speedup</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-zinc-300 font-mono">
                {currentDataset.map((row) => (
                  <tr key={row.format} className="hover:bg-white/5">
                    <td className="p-3 font-bold text-zinc-100">{row.format}</td>
                    <td className="p-3">{row.size}</td>
                    <td className="p-3 text-right">{row.tokenMatch.toFixed(2)}%</td>
                    <td className="p-3 text-right">{row.bleu.toFixed(2)}%</td>
                    <td className="p-3 text-right">{row.latency.toFixed(1)} ms</td>
                    <td className="p-3 text-right text-amber-400 font-bold">{row.speedup.toFixed(2)}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Model access and hugging face cards */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-zinc-100">Published ONNX Bundles</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <a
            href="https://huggingface.co/naklitechie/indictrans2-en-indic-dist-200M-ONNX"
            target="_blank"
            rel="noopener noreferrer"
            className="glass-card p-5 rounded-xl block hover:border-zinc-700 space-y-3 group"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-teal-400">Hugging Face Repo</span>
              <ArrowUpRight size={14} className="text-zinc-500 group-hover:text-zinc-300 transition-colors" />
            </div>
            <div className="space-y-1">
              <h4 className="font-bold text-zinc-100 group-hover:text-white">en-indic-dist-200M</h4>
              <p className="text-xs text-zinc-400">Highly optimized 200M distilled English-to-Indic translation bundle.</p>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-zinc-500">
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~1.1 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~303 MB INT8</span>
            </div>
          </a>

          <div className="glass-card p-5 rounded-xl block space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-teal-400">Local Cache Ready</span>
            </div>
            <div className="space-y-1">
              <h4 className="font-bold text-zinc-100">indic-en-dist-200M</h4>
              <p className="text-xs text-zinc-400">Exported locally to cache folder. Optimized cross-attention layer mapping.</p>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-zinc-500">
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~0.9 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~257 MB INT8</span>
            </div>
          </div>

          <div className="glass-card p-5 rounded-xl block space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-teal-400">External Weight Sidecars</span>
            </div>
            <div className="space-y-1">
              <h4 className="font-bold text-zinc-100">indic-indic-dist-320M</h4>
              <p className="text-xs text-zinc-400">Uses external data protobuf formats to handle decoder weight sizes &gt; 2 GB.</p>
            </div>
            <div className="flex items-center gap-2 text-[10px] text-zinc-500">
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~1.3 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~371 MB INT8</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
