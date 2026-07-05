import { useState, useEffect } from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from 'recharts'
import { Search, Loader2 } from 'lucide-react'

interface LanguageMetric {
  total_fixtures: number
  token_exact_rate: number
  text_exact_rate: number
  token_exact_count: number
  text_exact_count: number
  sacrebleu_bleu: number
  sacrebleu_chrf: number
  fp32_avg_latency_ms: number
  cmp_avg_latency_ms: number
  speedup_vs_fp32: number
  fp32_tokens_per_sec?: number
  cmp_tokens_per_sec?: number
}

interface BenchmarkData {
  label: string
  oracle: string
  cmp_dir: string
  total_fixtures: number
  token_exact_rate: number
  text_exact_rate: number
  token_exact_count: number
  text_exact_count: number
  sacrebleu_bleu: number
  sacrebleu_chrf: number
  fp32_avg_latency_ms: number
  cmp_avg_latency_ms: number
  speedup_vs_fp32: number
  fp32_tokens_per_sec?: number
  cmp_tokens_per_sec?: number
  sacrebleu_bleu_mixed?: number
  sacrebleu_chrf_mixed?: number
  metrics_by_language: Record<string, LanguageMetric>
}

// Map language codes to human names
const langNameMap: Record<string, string> = {
  asm_Beng: 'Assamese (Bengali Script)',
  ben_Beng: 'Bengali (Bengali Script)',
  brx_Deva: 'Bodo (Devanagari Script)',
  doi_Deva: 'Dogri (Devanagari Script)',
  gom_Deva: 'Konkani (Devanagari Script)',
  guj_Gujr: 'Gujarati (Gujarati Script)',
  hin_Deva: 'Hindi (Devanagari Script)',
  kan_Knda: 'Kannada (Kannada Script)',
  kas_Arab: 'Kashmiri (Arabic Script)',
  kas_Deva: 'Kashmiri (Devanagari Script)',
  mai_Deva: 'Maithili (Devanagari Script)',
  mal_Mlym: 'Malayalam (Malayalam Script)',
  mar_Deva: 'Marathi (Devanagari Script)',
  mni_Beng: 'Manipuri (Bengali Script)',
  mni_Mtei: 'Manipuri (Meitei Script)',
  npi_Deva: 'Nepali (Devanagari Script)',
  ory_Orya: 'Odia (Oriya Script)',
  pan_Guru: 'Punjabi (Gurmukhi Script)',
  san_Deva: 'Sanskrit (Devanagari Script)',
  sat_Olck: 'Santali (Ol Chiki Script)',
  snd_Arab: 'Sindhi (Arabic Script)',
  snd_Deva: 'Sindhi (Devanagari Script)',
  tam_Taml: 'Tamil (Tamil Script)',
  tel_Telu: 'Telugu (Telugu Script)',
  urd_Arab: 'Urdu (Arabic Script)',
  eng_Latn: 'English (Latin Script)'
}

export function BenchmarkTab() {
  const [direction, setDirection] = useState<'en-indic' | 'indic-en' | 'indic-indic'>('en-indic')
  const [scale, setScale] = useState<'base' | '1b'>('base')
  const [quantization, setQuantization] = useState<'fp16' | 'int8' | 'q4f16'>('fp16')
  const [data, setData] = useState<BenchmarkData | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState<string>('')
  const [chartMetric, setChartMetric] = useState<'bleu' | 'parity'>('bleu')

  useEffect(() => {
    const fetchBenchmark = async () => {
      setLoading(true)
      setError(null)
      const scaleStr = scale === '1b' ? '1b-' : ''
      const filename = `benchmark-${direction}-${scaleStr}${quantization}.json`
      const url = `./fixtures/${filename}`

      try {
        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`Failed to load benchmark configuration: ${response.statusText}`)
        }
        const json = await response.json()
        setData(json)
      } catch (err: any) {
        setError(err.message || 'An error occurred while loading benchmark files')
      } finally {
        setLoading(false)
      }
    }

    fetchBenchmark()
  }, [direction, scale, quantization])

  // Filter languages
  const filteredLanguages = data
    ? Object.entries(data.metrics_by_language)
        .map(([code, metrics]) => ({
          code,
          name: langNameMap[code] || code,
          ...metrics
        }))
        .filter(
          (lang) =>
            lang.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
            lang.name.toLowerCase().includes(searchQuery.toLowerCase())
        )
    : []

  // Dynamic colors for metrics
  const getScoreColor = (score: number) => {
    if (score >= 98) return 'text-emerald-400'
    if (score >= 85) return 'text-indigo-400'
    if (score >= 70) return 'text-amber-400'
    return 'text-rose-400'
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Selector controls */}
      <div className="glass p-6 rounded-xl space-y-4">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Benchmark Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="space-y-2">
            <label className="text-xs font-semibold text-zinc-500">Direction</label>
            <div className="flex bg-zinc-900 border border-white/5 rounded-lg p-1">
              {(['en-indic', 'indic-en', 'indic-indic'] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDirection(d)}
                  className={`flex-1 text-center py-1.5 text-xs font-semibold rounded transition ${
                    direction === d ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold text-zinc-500">Model Scale</label>
            <div className="flex bg-zinc-900 border border-white/5 rounded-lg p-1">
              {(['base', '1b'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => setScale(s)}
                  className={`flex-1 text-center py-1.5 text-xs font-semibold rounded transition ${
                    scale === s ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {s === 'base' ? '200M/320M Base' : '1B Large'}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold text-zinc-500">Quantization / Precision</label>
            <div className="flex bg-zinc-900 border border-white/5 rounded-lg p-1">
              {(['fp16', 'int8', 'q4f16'] as const).map((q) => (
                <button
                  key={q}
                  onClick={() => setQuantization(q)}
                  className={`flex-1 text-center py-1.5 text-xs font-semibold rounded transition ${
                    quantization === q ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {q.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 space-y-4">
          <Loader2 className="animate-spin text-teal-400" size={32} />
          <p className="text-xs text-zinc-400">Fetching benchmark data from static fixtures...</p>
        </div>
      ) : error ? (
        <div className="glass-card p-8 rounded-xl text-center space-y-4 max-w-md mx-auto">
          <div className="text-rose-400 text-3xl font-extrabold">Data Unavailable</div>
          <p className="text-xs text-zinc-400">
            The benchmark file <code>benchmark-{direction}-{scale === '1b' ? '1b-' : ''}{quantization}.json</code> could not be loaded. This format may not have been exported or processed.
          </p>
        </div>
      ) : data ? (
        <div className="space-y-8">
          {/* Main cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="glass-card p-5 rounded-xl space-y-2">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">SacreBLEU Parity</span>
              <div className="flex items-baseline gap-1">
                <span className={`text-2xl font-extrabold ${getScoreColor(data.sacrebleu_bleu)}`}>
                  {data.sacrebleu_bleu.toFixed(2)}
                </span>
                <span className="text-xs text-zinc-500">/ 100</span>
              </div>
              <p className="text-[10px] text-zinc-400">
                Mixed: <span className="font-bold text-zinc-300">{(data.sacrebleu_bleu_mixed ?? data.sacrebleu_bleu).toFixed(2)}</span>
              </p>
            </div>

            <div className="glass-card p-5 rounded-xl space-y-2">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Exact Match (Token)</span>
              <div className="flex items-baseline gap-1">
                <span className={`text-2xl font-extrabold ${getScoreColor(data.token_exact_rate)}`}>
                  {data.token_exact_rate.toFixed(2)}%
                </span>
              </div>
              <p className="text-[10px] text-zinc-400">
                Count: <span className="font-bold text-zinc-300">{data.token_exact_count}</span> / {data.total_fixtures}
              </p>
            </div>

            <div className="glass-card p-5 rounded-xl space-y-2">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Generation Latency</span>
              <div className="flex items-baseline gap-1.5">
                <span className="text-2xl font-extrabold text-zinc-100">{data.cmp_avg_latency_ms.toFixed(1)}ms</span>
                <span className="text-[10px] text-zinc-500">vs {data.fp32_avg_latency_ms.toFixed(1)}ms Oracle</span>
              </div>
              <p className="text-[10px] text-zinc-400">
                Measured on single-core CPU runner.
              </p>
            </div>

            <div className="glass-card p-5 rounded-xl space-y-2">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider">Speedup Ratio</span>
              <div className="flex items-baseline gap-1">
                <span className={`text-2xl font-extrabold ${data.speedup_vs_fp32 >= 1.0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                  {data.speedup_vs_fp32.toFixed(3)}x
                </span>
              </div>
              <p className="text-[10px] text-zinc-400">
                Throughput: <span className="font-bold text-zinc-300">{data.cmp_tokens_per_sec?.toFixed(1) ?? '-'}</span> tok/s
              </p>
            </div>
          </div>

          {/* Lang breakdown visual chart */}
          {filteredLanguages.length > 1 && (
            <div className="glass-card p-6 rounded-xl space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h4 className="text-base font-bold text-zinc-100">Language Performance Distributions</h4>
                  <p className="text-xs text-zinc-400">Language-by-language quality comparison.</p>
                </div>
                <div className="flex bg-zinc-900 border border-white/5 rounded-lg p-1 self-start">
                  <button
                    onClick={() => setChartMetric('bleu')}
                    className={`px-3 py-1 text-xs font-semibold rounded transition ${
                      chartMetric === 'bleu' ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    SacreBLEU
                  </button>
                  <button
                    onClick={() => setChartMetric('parity')}
                    className={`px-3 py-1 text-xs font-semibold rounded transition ${
                      chartMetric === 'parity' ? 'bg-white/10 text-white' : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    Token Parity
                  </button>
                </div>
              </div>

              <div className="h-[250px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={filteredLanguages.slice(0, 16)} // Top 16 to avoid clattering
                    margin={{ top: 10, right: 10, bottom: 20, left: -20 }}
                  >
                    <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
                    <XAxis dataKey="code" stroke="#a1a1aa" fontSize={9} tickLine={false} />
                    <YAxis stroke="#a1a1aa" fontSize={9} domain={[0, 105]} />
                    <Tooltip
                      contentStyle={{ background: '#09090b', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px' }}
                      labelStyle={{ fontWeight: 'bold', color: '#f4f4f5', fontSize: '11px' }}
                      itemStyle={{ color: '#2dd4bf', fontSize: '11px' }}
                    />
                    <Bar
                      dataKey={chartMetric === 'bleu' ? 'sacrebleu_bleu' : 'token_exact_rate'}
                      name={chartMetric === 'bleu' ? 'SacreBLEU' : 'Exact Token Match (%)'}
                      fill="#0d9488"
                      radius={[3, 3, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Languages detailed table */}
          <div className="glass-card p-6 rounded-xl space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h4 className="text-base font-bold text-zinc-100">Language Metrics Summary</h4>
                <p className="text-xs text-zinc-400">Detailed metric grid by language.</p>
              </div>
              <div className="relative w-full sm:w-64">
                <Search className="absolute left-3 top-2.5 text-zinc-500" size={14} />
                <input
                  type="text"
                  placeholder="Search languages..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-9 pr-4 py-2 bg-zinc-900 border border-white/5 rounded-lg text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-teal-500 transition-colors"
                />
              </div>
            </div>

            <div className="overflow-x-auto border border-white/5 rounded-lg">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="bg-zinc-900/60 text-zinc-400 border-b border-white/5 font-semibold">
                    <th className="p-3">Language Code</th>
                    <th className="p-3">Language Name</th>
                    <th className="p-3 text-right">Fixtures</th>
                    <th className="p-3 text-right">Exact Token Match</th>
                    <th className="p-3 text-right">Exact Text Match</th>
                    <th className="p-3 text-right">SacreBLEU</th>
                    <th className="p-3 text-right">Avg Latency</th>
                    <th className="p-3 text-right">Speedup</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-zinc-300 font-mono">
                  {filteredLanguages.length > 0 ? (
                    filteredLanguages.map((lang) => (
                      <tr key={lang.code} className="hover:bg-white/5">
                        <td className="p-3 font-bold text-zinc-100">{lang.code}</td>
                        <td className="p-3 text-zinc-400 font-sans">{lang.name}</td>
                        <td className="p-3 text-right">{lang.total_fixtures}</td>
                        <td className="p-3 text-right">{lang.token_exact_rate.toFixed(2)}%</td>
                        <td className="p-3 text-right">{lang.text_exact_rate.toFixed(2)}%</td>
                        <td className={`p-3 text-right font-bold ${getScoreColor(lang.sacrebleu_bleu)}`}>
                          {lang.sacrebleu_bleu.toFixed(2)}
                        </td>
                        <td className="p-3 text-right">
                          {lang.cmp_avg_latency_ms.toFixed(1)}ms <span className="text-[10px] text-zinc-500">/ {lang.fp32_avg_latency_ms.toFixed(1)}ms</span>
                        </td>
                        <td className={`p-3 text-right font-bold ${lang.speedup_vs_fp32 >= 1.0 ? 'text-emerald-400' : 'text-zinc-400'}`}>
                          {lang.speedup_vs_fp32.toFixed(2)}x
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8} className="p-8 text-center text-zinc-500 font-sans">
                        No languages found matching "{searchQuery}"
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
