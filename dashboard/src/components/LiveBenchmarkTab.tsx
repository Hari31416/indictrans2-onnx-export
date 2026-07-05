import { useState, useEffect } from 'react'
import { Cpu, CheckCircle2, CheckCircle, XCircle } from 'lucide-react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts'

interface BenchmarkResult {
  id: string
  configId: string
  direction: string
  scale: string
  precision: string
  provider: 'webgpu' | 'wasm'
  loadTimeMs: number | null
  avgTtftMs: number | null
  avgStepLatencyMs: number | null
  tokensPerSec: number | null
  sentencesTested: number
  totalSentences: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'skipped'
  error?: string
}

export function LiveBenchmarkTab() {
  const [results, setResults] = useState<BenchmarkResult[]>([])
  const [viewMode, setViewMode] = useState<'table' | 'charts'>('table')

  // Data processing for charts
  const getAggregatedChartData = () => {
    const configs = [
      { scale: 'base', precision: 'fp32', label: 'Base (FP32)' },
      { scale: 'base', precision: 'fp16', label: 'Base (FP16)' },
      { scale: 'base', precision: 'int8', label: 'Base (INT8)' },
      { scale: 'base', precision: 'q4f16', label: 'Base (Q4F16)' },
      { scale: '1b', precision: 'fp32', label: '1B (FP32)' },
      { scale: '1b', precision: 'fp16', label: '1B (FP16)' },
      { scale: '1b', precision: 'int8', label: '1B (INT8)' },
      { scale: '1b', precision: 'q4f16', label: '1B (Q4F16)' },
    ]

    return configs.map((cfg) => {
      const webgpuItems = results.filter(
        (r) =>
          r.scale === cfg.scale &&
          r.precision === cfg.precision &&
          r.provider === 'webgpu' &&
          r.status === 'completed'
      )
      const wasmItems = results.filter(
        (r) =>
          r.scale === cfg.scale &&
          r.precision === cfg.precision &&
          r.provider === 'wasm' &&
          r.status === 'completed'
      )

      const getAverage = (items: typeof results, field: 'tokensPerSec' | 'avgTtftMs' | 'avgStepLatencyMs' | 'loadTimeMs') => {
        const valid = items.map((i) => i[field]).filter((val): val is number => val !== null && val !== undefined)
        return valid.length > 0 ? Math.round((valid.reduce((sum, v) => sum + v, 0) / valid.length) * 10) / 10 : null
      }

      const getSkipReason = (provider: 'webgpu' | 'wasm') => {
        const matches = results.filter(
          (r) =>
            r.scale === cfg.scale &&
            r.precision === cfg.precision &&
            r.provider === provider
        )
        if (matches.length > 0 && matches.every(m => m.status === 'skipped')) {
          if (provider === 'webgpu' && cfg.precision === 'int8') return 'Operator Limits'
          if (provider === 'wasm' && cfg.scale === '1b' && (cfg.precision === 'fp32' || cfg.precision === 'fp16')) return 'WASM 4GB Limit'
          if (provider === 'wasm' && cfg.scale === 'base' && cfg.precision === 'fp32') return 'Heap Frag Safety'
          if (provider === 'webgpu' && cfg.scale === '1b' && cfg.precision === 'fp32') return 'GPU 2GB Buffer'
          return 'Skipped'
        }
        return null
      }

      return {
        name: cfg.label,
        scale: cfg.scale,
        precision: cfg.precision,
        webgpuTPS: getAverage(webgpuItems, 'tokensPerSec'),
        wasmTPS: getAverage(wasmItems, 'tokensPerSec'),
        webgpuTTFT: getAverage(webgpuItems, 'avgTtftMs'),
        wasmTTFT: getAverage(wasmItems, 'avgTtftMs'),
        webgpuStep: getAverage(webgpuItems, 'avgStepLatencyMs'),
        wasmStep: getAverage(wasmItems, 'avgStepLatencyMs'),
        webgpuSkip: getSkipReason('webgpu'),
        wasmSkip: getSkipReason('wasm'),
      }
    })
  }

  // Safe fetch helper with relative fallback for local dev server
  const getUrl = (path: string): string => {
    const base = import.meta.env.BASE_URL || '/'
    return `${base}${path}`.replace(/\/+/g, '/')
  }

  useEffect(() => {
    const loadDefaultBenchmarks = async () => {
      try {
        const checkUrl = getUrl('fixtures/live-browser-benchmarks.json')
        const res = await fetch(checkUrl)
        if (res.ok) {
          const data = await res.json()
          setResults(data)
          // Default to charts view if results loaded
          if (data && data.length > 0) {
            setViewMode('charts')
          }
        }
      } catch (e) {
        // Fallback silently if not available
      }
    }
    loadDefaultBenchmarks()
  }, [])

  return (
    <div className="space-y-6 animate-in fade-in duration-300">

      {/* Top Header */}
      <div className="glass p-6 rounded-xl space-y-4">
        <div>
          <h2 className="text-lg font-extrabold text-zinc-100 flex items-center gap-2">
            <Cpu className="text-teal-400" size={20} />
            Browser Latency Benchmarks & Engine Limits
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Performance, latency profiles, and resource constraints for running IndicTrans2 ONNX models locally in client browsers (WebGPU vs WebAssembly).
          </p>
        </div>
      </div>

      <div className="w-full space-y-6 flex flex-col">
        {/* Results Table & Charts */}
        <div className="glass p-5 rounded-xl space-y-4 flex-1 flex flex-col">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-white/5 pb-3 gap-2">
            <h3 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <CheckCircle2 size={13} className="text-teal-400" />
              Benchmark Evaluation Results
            </h3>
            <div className="flex bg-zinc-950 border border-white/5 rounded-lg p-0.5 self-start sm:self-auto">
              <button
                onClick={() => setViewMode('table')}
                className={`px-3 py-1 text-[10px] font-extrabold rounded-md transition ${
                  viewMode === 'table' ? 'bg-teal-500/10 border border-teal-500/20 text-teal-300 shadow' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Table View
              </button>
              <button
                onClick={() => setViewMode('charts')}
                className={`px-3 py-1 text-[10px] font-extrabold rounded-md transition ${
                  viewMode === 'charts' ? 'bg-teal-500/10 border border-teal-500/20 text-teal-300 shadow' : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                Visual Analytics
              </button>
            </div>
          </div>

          {viewMode === 'table' ? (
            <div className="flex-1 overflow-x-auto min-h-[300px]">
              <table className="w-full text-left border-collapse text-xs font-medium">
                <thead>
                  <tr className="border-b border-white/5 text-zinc-500 text-[10px] uppercase tracking-wider">
                    <th className="py-3 px-2">Configuration</th>
                    <th className="py-3 px-2">Backend</th>
                    <th className="py-3 px-2">Load (ms)</th>
                    <th className="py-3 px-2">Avg TTFT (ms)</th>
                    <th className="py-3 px-2">Avg Step (ms)</th>
                    <th className="py-3 px-2">Speed (t/s)</th>
                    <th className="py-3 px-2">Sentences</th>
                    <th className="py-3 px-2 text-right">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-zinc-300 font-mono">
                  {results.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="py-12 text-center text-zinc-500 italic font-sans">
                        No benchmarks data found. Verify live-browser-benchmarks.json exists.
                      </td>
                    </tr>
                  ) : (
                    results.map((res) => (
                      <tr key={res.id}>
                        <td className="py-2.5 px-2">
                          <span className="font-sans font-bold text-zinc-200">{res.direction}</span>
                          <span className="text-[10px] text-zinc-400 ml-1.5">({res.scale} / {res.precision})</span>
                        </td>
                        <td className="py-2.5 px-2 uppercase font-bold text-zinc-400">{res.provider}</td>
                        <td className="py-2.5 px-2 font-bold">{res.loadTimeMs ?? '—'}</td>
                        <td className="py-2.5 px-2 text-teal-400 font-bold">{res.avgTtftMs ?? '—'}</td>
                        <td className="py-2.5 px-2 text-indigo-400 font-bold">{res.avgStepLatencyMs ?? '—'}</td>
                        <td className="py-2.5 px-2 text-emerald-400 font-bold">{res.tokensPerSec ?? '—'}</td>
                        <td className="py-2.5 px-2 text-zinc-400">{res.sentencesTested} / {res.totalSentences}</td>
                        <td className="py-2.5 px-2 text-right">
                          <span className={`inline-block px-1.5 py-0.5 rounded text-[8px] font-black uppercase tracking-wider border ${res.status === 'completed' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' :
                            res.status === 'running' ? 'bg-teal-500/10 border-teal-500/20 text-teal-400 animate-pulse' :
                              res.status === 'failed' ? 'bg-rose-500/10 border-rose-500/20 text-rose-400' :
                                res.status === 'cancelled' ? 'bg-zinc-500/10 border-zinc-500/20 text-zinc-400' :
                                  res.status === 'skipped' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
                                    'bg-zinc-900 border-white/5 text-zinc-500'
                            }`}>
                            {res.status}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="flex-1 space-y-6 overflow-y-auto max-h-[700px] pr-1.5 scrollbar-thin">
              {results.length === 0 ? (
                <div className="py-16 text-center text-zinc-500 italic">
                  No data to visualize. Verify live-browser-benchmarks.json exists.
                </div>
              ) : (
                <>
                  {/* Throughput chart */}
                  <div className="glass p-4 rounded-xl space-y-3">
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-zinc-200">Throughput Comparison (Higher is Better)</span>
                      <span className="text-[10px] text-zinc-500">Tokens generated per second during the autoregressive decode loop</span>
                    </div>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={getAggregatedChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                          <XAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 9 }} stroke="rgba(255,255,255,0.1)" />
                          <YAxis tick={{ fill: '#a1a1aa', fontSize: 9 }} stroke="rgba(255,255,255,0.1)" />
                          <Tooltip
                            cursor={{ fill: 'transparent' }}
                            contentStyle={{ background: '#09090b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '11px', color: '#f4f4f5' }}
                            formatter={(value: any, name: any, props: any) => {
                              if (value === null) {
                                const skipField = name.includes('WebGPU') ? 'webgpuSkip' : 'wasmSkip'
                                const reason = props.payload[skipField]
                                return [`Skipped: ${reason || 'N/A'}`, name]
                              }
                              return [`${value} t/s`, name]
                            }}
                          />
                          <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                          <Bar dataKey="webgpuTPS" name="WebGPU (TPS)" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="wasmTPS" name="WASM CPU (TPS)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Latency subplots side-by-side */}
                  <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                    {/* TTFT Chart */}
                    <div className="glass p-4 rounded-xl space-y-3">
                      <div className="flex flex-col">
                        <span className="text-xs font-bold text-zinc-200">Prefill Latency / TTFT (Lower is Better)</span>
                        <span className="text-[10px] text-zinc-500">Time-to-first-token in milliseconds</span>
                      </div>
                      <div className="h-52 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={getAggregatedChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 8 }} stroke="rgba(255,255,255,0.1)" />
                            <YAxis tick={{ fill: '#a1a1aa', fontSize: 8 }} stroke="rgba(255,255,255,0.1)" />
                            <Tooltip
                              cursor={{ fill: 'transparent' }}
                              contentStyle={{ background: '#09090b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '10px', color: '#f4f4f5' }}
                              formatter={(value: any, name: any) => value !== null ? [`${value} ms`, name] : ['Skipped', name]}
                            />
                            <Bar dataKey="webgpuTTFT" name="WebGPU TTFT" fill="#2dd4bf" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="wasmTTFT" name="WASM TTFT" fill="#818cf8" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    {/* Step Latency Chart */}
                    <div className="glass p-4 rounded-xl space-y-3">
                      <div className="flex flex-col">
                        <span className="text-xs font-bold text-zinc-200">Decode Step Latency (Lower is Better)</span>
                        <span className="text-[10px] text-zinc-500">Average generation latency per token in milliseconds</span>
                      </div>
                      <div className="h-52 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={getAggregatedChartData()} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                            <XAxis dataKey="name" tick={{ fill: '#a1a1aa', fontSize: 8 }} stroke="rgba(255,255,255,0.1)" />
                            <YAxis tick={{ fill: '#a1a1aa', fontSize: 8 }} stroke="rgba(255,255,255,0.1)" />
                            <Tooltip
                              cursor={{ fill: 'transparent' }}
                              contentStyle={{ background: '#09090b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '12px', fontSize: '10px', color: '#f4f4f5' }}
                              formatter={(value: any, name: any) => value !== null ? [`${value} ms/tok`, name] : ['Skipped', name]}
                            />
                            <Bar dataKey="webgpuStep" name="WebGPU Step" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                            <Bar dataKey="wasmStep" name="WASM Step" fill="#6366f1" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Compatibility Grid Heatmap */}
                  <div className="glass p-4 rounded-xl space-y-4">
                    <div className="flex flex-col">
                      <span className="text-xs font-bold text-zinc-200">Browser Compatibility & Limits Matrix</span>
                      <span className="text-[10px] text-zinc-500">Cross-reference map highlighting hardware and address space boundary failures</span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* WebGPU Column */}
                      <div className="space-y-2">
                        <div className="text-[10px] font-extrabold text-teal-400 uppercase tracking-widest px-1">
                          WebGPU (Chrome/Edge Engine)
                        </div>
                        <div className="space-y-1.5">
                          {getAggregatedChartData().map((cfg, idx) => (
                            <div
                              key={`webgpu-matrix-${idx}`}
                              className={`flex items-center justify-between px-3 py-2 rounded-xl border text-[11px] transition duration-200 ${
                                cfg.webgpuTPS !== null
                                  ? 'bg-emerald-500/5 border-emerald-500/10 text-emerald-300 hover:bg-emerald-500/10'
                                  : 'bg-rose-500/5 border-rose-500/10 text-rose-300 hover:bg-rose-500/10'
                              }`}
                            >
                              <span className="font-semibold text-zinc-300">{cfg.name}</span>
                              {cfg.webgpuTPS !== null ? (
                                <div className="flex items-center gap-1.5 font-mono font-bold">
                                  <CheckCircle size={11} className="text-emerald-400" />
                                  <span>{cfg.webgpuTPS} t/s</span>
                                </div>
                              ) : (
                                <div className="flex items-center gap-1.5 text-right select-none" title={cfg.webgpuSkip || ''}>
                                  <XCircle size={11} className="text-rose-400 shrink-0" />
                                  <span className="text-[9px] font-medium leading-tight max-w-[120px] text-rose-400/80 truncate">
                                    {cfg.webgpuSkip || 'Skipped'}
                                  </span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* WASM Column */}
                      <div className="space-y-2">
                        <div className="text-[10px] font-extrabold text-indigo-400 uppercase tracking-widest px-1">
                          WebAssembly CPU (Threaded)
                        </div>
                        <div className="space-y-1.5">
                          {getAggregatedChartData().map((cfg, idx) => (
                            <div
                              key={`wasm-matrix-${idx}`}
                              className={`flex items-center justify-between px-3 py-2 rounded-xl border text-[11px] transition duration-200 ${
                                cfg.wasmTPS !== null
                                  ? 'bg-emerald-500/5 border-emerald-500/10 text-emerald-300 hover:bg-emerald-500/10'
                                  : 'bg-rose-500/5 border-rose-500/10 text-rose-300 hover:bg-rose-500/10'
                              }`}
                            >
                              <span className="font-semibold text-zinc-300">{cfg.name}</span>
                              {cfg.wasmTPS !== null ? (
                                <div className="flex items-center gap-1.5 font-mono font-bold">
                                  <CheckCircle size={11} className="text-emerald-400" />
                                  <span>{cfg.wasmTPS} t/s</span>
                                </div>
                              ) : (
                                <div className="flex items-center gap-1.5 text-right select-none" title={cfg.wasmSkip || ''}>
                                  <XCircle size={11} className="text-rose-400 shrink-0" />
                                  <span className="text-[9px] font-medium leading-tight max-w-[120px] text-rose-400/80 truncate">
                                    {cfg.wasmSkip || 'Skipped'}
                                  </span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
