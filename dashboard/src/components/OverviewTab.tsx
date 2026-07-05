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
      { format: 'FP32', size: '1.77 GB', sizeBytes: 1.77, tokenMatch: 100.0, bleu: 100.0, latency: 76.1, speedup: 1.0 },
      { format: 'FP16', size: '926.5 MB', sizeBytes: 0.90, tokenMatch: 99.64, bleu: 100.0, latency: 82.1, speedup: 0.93 },
      { format: 'INT8', size: '487.4 MB', sizeBytes: 0.48, tokenMatch: 73.73, bleu: 91.56, latency: 45.6, speedup: 1.57 },
      { format: 'Q4F16', size: '657.8 MB', sizeBytes: 0.64, tokenMatch: 55.45, bleu: 83.46, latency: 74.9, speedup: 1.01 }
    ],
    scale1b: [
      { format: 'FP32', size: '6.68 GB', sizeBytes: 6.68, tokenMatch: 100.0, bleu: 100.0, latency: 244.4, speedup: 1.0 },
      { format: 'FP16', size: '3.36 GB', sizeBytes: 3.36, tokenMatch: 99.73, bleu: 100.0, latency: 259.8, speedup: 0.94 },
      { format: 'INT8', size: '1.71 GB', sizeBytes: 1.71, tokenMatch: 89.64, bleu: 95.69, latency: 112.6, speedup: 2.23 },
      { format: 'Q4F16', size: '1.71 GB', sizeBytes: 1.71, tokenMatch: 82.27, bleu: 92.55, latency: 143.3, speedup: 1.67 }
    ]
  },
  {
    name: 'Indic to English (indic-en)',
    scale200: [
      { format: 'FP32', size: '1.26 GB', sizeBytes: 1.26, tokenMatch: 100.0, bleu: 100.0, latency: 41.4, speedup: 1.0 },
      { format: 'FP16', size: '661.6 MB', sizeBytes: 0.65, tokenMatch: 99.91, bleu: 99.98, latency: 47.6, speedup: 0.87 },
      { format: 'INT8', size: '354.0 MB', sizeBytes: 0.35, tokenMatch: 86.0, bleu: 93.87, latency: 38.9, speedup: 1.11 },
      { format: 'Q4F16', size: '392.9 MB', sizeBytes: 0.38, tokenMatch: 74.45, bleu: 89.15, latency: 47.8, speedup: 0.87 }
    ],
    scale1b: [
      { format: 'FP32', size: '5.64 GB', sizeBytes: 5.64, tokenMatch: 100.0, bleu: 100.0, latency: 171.8, speedup: 1.0 },
      { format: 'FP16', size: '2.84 GB', sizeBytes: 2.84, tokenMatch: 99.91, bleu: 99.98, latency: 180.5, speedup: 0.95 },
      { format: 'INT8', size: '1.45 GB', sizeBytes: 1.45, tokenMatch: 94.18, bleu: 97.94, latency: 76.4, speedup: 2.20 },
      { format: 'Q4F16', size: '1.19 GB', sizeBytes: 1.19, tokenMatch: 87.55, bleu: 95.17, latency: 85.5, speedup: 1.96 }
    ]
  },
  {
    name: 'Indic to Indic (indic-indic)',
    scale200: [
      { format: 'FP32', size: '1.92 GB', sizeBytes: 1.92, tokenMatch: 100.0, bleu: 100.0, latency: 98.2, speedup: 1.0 },
      { format: 'FP16', size: '1.02 GB', sizeBytes: 1.02, tokenMatch: 99.85, bleu: 100.0, latency: 102.5, speedup: 0.96 },
      { format: 'INT8', size: '535.8 MB', sizeBytes: 0.52, tokenMatch: 82.24, bleu: 94.81, latency: 54.1, speedup: 1.82 },
      { format: 'Q4F16', size: '697.1 MB', sizeBytes: 0.68, tokenMatch: 70.36, bleu: 88.52, latency: 88.4, speedup: 1.11 }
    ],
    scale1b: [
      { format: 'FP32', size: '7.85 GB', sizeBytes: 7.85, tokenMatch: 100.0, bleu: 100.0, latency: 312.4, speedup: 1.0 },
      { format: 'FP16', size: '3.98 GB', sizeBytes: 3.98, tokenMatch: 99.82, bleu: 100.0, latency: 326.8, speedup: 0.96 },
      { format: 'INT8', size: '2.04 GB', sizeBytes: 2.04, tokenMatch: 91.22, bleu: 96.88, latency: 142.1, speedup: 2.20 },
      { format: 'Q4F16', size: '2.01 GB', sizeBytes: 2.01, tokenMatch: 85.46, bleu: 93.92, latency: 182.2, speedup: 1.71 }
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
              {modelScale === '200M' ? '1.82x' : '2.23x'}
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
            <div className="text-3xl font-extrabold text-zinc-100">Up to 74%</div>
            <p className="text-xs text-zinc-400">File size reduced from 1.77 GB to 487.4 MB (INT8).</p>
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
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~1.7 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~650 MB quantized</span>
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
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~1.2 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~390 MB quantized</span>
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
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~1.9 GB FP32</span>
              <span className="bg-zinc-800 px-2 py-0.5 rounded border border-white/5">~697 MB quantized</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
