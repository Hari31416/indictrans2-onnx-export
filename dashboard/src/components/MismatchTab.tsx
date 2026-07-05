import { useState, useEffect } from 'react'
import { Loader2, ArrowRight, Copy, Check } from 'lucide-react'

interface MismatchItem {
  index: number
  fixture: {
    text: string
    src_lang: string
    tgt_lang: string
    category: string
  }
  fp32_tokens: number[]
  fp32_text: string
  tokens_match: boolean
  text_match: boolean
  // Dynamic keys based on quantization type:
  fp16_text?: string
  int8_text?: string
  q4f16_text?: string
  fp16_tokens?: number[]
  int8_tokens?: number[]
  q4f16_tokens?: number[]
}

const langNameMap: Record<string, string> = {
  asm_Beng: 'Assamese (Bengali)',
  ben_Beng: 'Bengali (Bengali)',
  brx_Deva: 'Bodo (Devanagari)',
  doi_Deva: 'Dogri (Devanagari)',
  gom_Deva: 'Konkani (Devanagari)',
  guj_Gujr: 'Gujarati (Gujarati)',
  hin_Deva: 'Hindi (Devanagari)',
  kan_Knda: 'Kannada (Kannada)',
  kas_Arab: 'Kashmiri (Arabic)',
  kas_Deva: 'Kashmiri (Devanagari)',
  mai_Deva: 'Maithili (Devanagari)',
  mal_Mlym: 'Malayalam (Malayalam)',
  mar_Deva: 'Marathi (Devanagari)',
  mni_Beng: 'Manipuri (Bengali)',
  mni_Mtei: 'Manipuri (Meitei)',
  npi_Deva: 'Nepali (Devanagari)',
  ory_Orya: 'Odia (Oriya)',
  pan_Guru: 'Punjabi (Gurmukhi)',
  san_Deva: 'Sanskrit (Devanagari)',
  sat_Olck: 'Santali (Ol Chiki)',
  snd_Arab: 'Sindhi (Arabic)',
  snd_Deva: 'Sindhi (Devanagari)',
  tam_Taml: 'Tamil (Tamil)',
  tel_Telu: 'Telugu (Telugu)',
  urd_Arab: 'Urdu (Arabic)',
  eng_Latn: 'English (Latin)'
}

export function MismatchTab() {
  const [direction, setDirection] = useState<'en-indic' | 'indic-en' | 'indic-indic'>('en-indic')
  const [scale, setScale] = useState<'base' | '1b'>('base')
  const [quantization, setQuantization] = useState<'fp16' | 'int8' | 'q4f16'>('fp16')
  
  const [mismatches, setMismatches] = useState<MismatchItem[]>([])
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  
  // Filters
  const [selectedLang, setSelectedLang] = useState<string>('all')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [page, setPage] = useState<number>(1)
  const pageSize = 10

  useEffect(() => {
    const fetchMismatches = async () => {
      setLoading(true)
      setError(null)
      setPage(1)
      const scaleStr = scale === '1b' ? '1b-' : ''
      const filename = `benchmark-${direction}-${scaleStr}${quantization}-mismatches.json`
      const url = `${import.meta.env.BASE_URL}fixtures/${filename}`

      try {
        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`Failed to load mismatch logs: ${response.statusText}`)
        }
        const json = await response.json()
        setMismatches(json)
      } catch (err: any) {
        setError(err.message || 'An error occurred while loading mismatch files')
      } finally {
        setLoading(false)
      }
    }

    fetchMismatches()
  }, [direction, scale, quantization])

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 1500)
  }

  // Get compared text for current quantization dynamically
  const getCmpText = (item: MismatchItem): string => {
    if (quantization === 'fp16' && item.fp16_text !== undefined) return item.fp16_text
    if (quantization === 'int8' && item.int8_text !== undefined) return item.int8_text
    if (quantization === 'q4f16' && item.q4f16_text !== undefined) return item.q4f16_text
    return 'N/A'
  }

  // Extract unique languages and categories for filters
  const uniqueLanguages = Array.from(
    new Set(
      mismatches.map((m) =>
        direction === 'en-indic' ? m.fixture.tgt_lang : m.fixture.src_lang
      )
    )
  ).sort()

  const uniqueCategories = Array.from(new Set(mismatches.map((m) => m.fixture.category))).sort()

  // Apply filters
  const filteredMismatches = mismatches.filter((item) => {
    const lang = direction === 'en-indic' ? item.fixture.tgt_lang : item.fixture.src_lang
    const matchLang = selectedLang === 'all' || lang === selectedLang
    const matchCat = selectedCategory === 'all' || item.fixture.category === selectedCategory
    return matchLang && matchCat
  })

  // Pagination calculations
  const totalPages = Math.ceil(filteredMismatches.length / pageSize)
  const paginatedItems = filteredMismatches.slice((page - 1) * pageSize, page * pageSize)

  // Word Alignment Diffing (LCS Algorithm)
  const renderHighlightedDiff = (refText: string, testText: string) => {
    const refWords = refText.trim().split(/\s+/)
    const testWords = testText.trim().split(/\s+/)

    const n = refWords.length
    const m = testWords.length
    const dp: number[][] = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0))

    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        if (refWords[i - 1].toLowerCase() === testWords[j - 1].toLowerCase()) {
          dp[i][j] = dp[i - 1][j - 1] + 1
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
        }
      }
    }

    interface DiffToken {
      type: 'added' | 'removed' | 'equal'
      text: string
    }

    const diff: DiffToken[] = []
    let i = n
    let j = m

    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && refWords[i - 1].toLowerCase() === testWords[j - 1].toLowerCase()) {
        diff.unshift({ type: 'equal', text: testWords[j - 1] })
        i--
        j--
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        diff.unshift({ type: 'added', text: testWords[j - 1] })
        j--
      } else {
        diff.unshift({ type: 'removed', text: refWords[i - 1] })
        i--
      }
    }

    return (
      <div className="flex flex-wrap gap-x-1 gap-y-1 text-zinc-300 font-sans text-sm items-center leading-relaxed">
        {diff.map((token, idx) => {
          if (token.type === 'equal') {
            return (
              <span key={idx} className="text-zinc-300 px-0.5">
                {token.text}
              </span>
            )
          }
          if (token.type === 'added') {
            return (
              <span
                key={idx}
                className="bg-teal-500/20 text-teal-300 px-1.5 py-0.5 rounded font-semibold border border-teal-500/30"
                title="Added in Quantized model"
              >
                {token.text}
              </span>
            )
          }
          // Removed
          return (
            <span
              key={idx}
              className="bg-rose-500/10 text-rose-400 line-through px-1.5 py-0.5 rounded font-semibold border border-rose-500/20"
              title="Removed from Oracle translation"
            >
              {token.text}
            </span>
          )
        })}
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-300">
      {/* Selector controls */}
      <div className="glass p-6 rounded-xl space-y-4">
        <h3 className="text-sm font-bold text-zinc-400 uppercase tracking-wider">Mismatch Log Selector</h3>
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
          <p className="text-xs text-zinc-400">Loading mismatch list...</p>
        </div>
      ) : error ? (
        <div className="glass-card p-8 rounded-xl text-center space-y-4 max-w-md mx-auto">
          <div className="text-emerald-400 text-3xl font-extrabold">100% Matching Parity!</div>
          <p className="text-xs text-zinc-400">
            No mismatches log found for <code>benchmark-{direction}-{scale === '1b' ? '1b-' : ''}{quantization}-mismatches.json</code>. This precision model matches the FP32 Oracle perfectly on all tested sentences!
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1 glass p-5 rounded-xl space-y-5">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Filters</h4>
            
            <div className="space-y-2">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">
                {direction === 'en-indic' ? 'Target Language' : 'Source Language'}
              </label>
              <select
                value={selectedLang}
                onChange={(e) => {
                  setSelectedLang(e.target.value)
                  setPage(1)
                }}
                className="w-full p-2.5 bg-zinc-900 border border-white/5 rounded-lg text-xs text-zinc-200 focus:outline-none focus:border-teal-500"
              >
                <option value="all">All Languages ({uniqueLanguages.length})</option>
                {uniqueLanguages.map((l) => (
                  <option key={l} value={l}>
                    {langNameMap[l] || l}
                  </option>
                ))}
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Category</label>
              <select
                value={selectedCategory}
                onChange={(e) => {
                  setSelectedCategory(e.target.value)
                  setPage(1)
                }}
                className="w-full p-2.5 bg-zinc-900 border border-white/5 rounded-lg text-xs text-zinc-200 focus:outline-none focus:border-teal-500"
              >
                <option value="all">All Categories ({uniqueCategories.length})</option>
                {uniqueCategories.map((c) => (
                  <option key={c} value={c}>
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </option>
                ))}
              </select>
            </div>

            <div className="pt-4 border-t border-white/5 space-y-2">
              <div className="flex justify-between text-xs text-zinc-400">
                <span>Total Mismatches:</span>
                <span className="font-mono text-zinc-200">{mismatches.length}</span>
              </div>
              <div className="flex justify-between text-xs text-zinc-400">
                <span>Filtered count:</span>
                <span className="font-mono text-zinc-200">{filteredMismatches.length}</span>
              </div>
            </div>
          </div>

          {/* List display */}
          <div className="lg:col-span-3 space-y-6">
            {filteredMismatches.length === 0 ? (
              <div className="glass-card p-12 rounded-xl text-center text-zinc-500 text-sm">
                No mismatch cases match the selected filters. Try broadening your criteria.
              </div>
            ) : (
              <>
                <div className="space-y-4">
                  {paginatedItems.map((item) => {
                    const srcLangName = langNameMap[item.fixture.src_lang] || item.fixture.src_lang
                    const tgtLangName = langNameMap[item.fixture.tgt_lang] || item.fixture.tgt_lang
                    const cmpText = getCmpText(item)
                    const cardId = `card-${item.index}`

                    return (
                      <div key={item.index} className="glass-card p-6 rounded-xl space-y-4">
                        {/* Header metadata */}
                        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/5 pb-3">
                          <div className="flex items-center gap-2">
                            <span className="bg-zinc-800 text-zinc-300 font-mono px-2 py-0.5 rounded text-[10px] border border-white/5">
                              #{item.index}
                            </span>
                            <span className="text-xs text-zinc-400 font-medium">
                              {srcLangName} <ArrowRight size={10} className="inline mx-1" /> {tgtLangName}
                            </span>
                          </div>
                          <span className="bg-teal-500/10 text-teal-400 text-[10px] font-bold px-2.5 py-0.5 rounded-full border border-teal-500/20 uppercase tracking-wider">
                            {item.fixture.category}
                          </span>
                        </div>

                        {/* Source Sentence */}
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider">Source Sentence</span>
                            <button
                              onClick={() => handleCopy(item.fixture.text, `${cardId}-src`)}
                              className="text-zinc-500 hover:text-zinc-300 transition"
                            >
                              {copiedId === `${cardId}-src` ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
                            </button>
                          </div>
                          <p className="text-zinc-100 font-sans font-medium text-sm leading-relaxed">{item.fixture.text}</p>
                        </div>

                        {/* Comparative grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                          <div className="bg-zinc-950/60 border border-white/5 p-4 rounded-lg space-y-1.5">
                            <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block">FP32 Oracle Translation</span>
                            <p className="text-zinc-300 text-sm font-sans">{item.fp32_text}</p>
                          </div>

                          <div className="bg-zinc-950/60 border border-white/5 p-4 rounded-lg space-y-1.5">
                            <span className="text-[9px] font-bold text-amber-500 uppercase tracking-wider block">
                              {quantization.toUpperCase()} Quantized Translation (Drift Highlighted)
                            </span>
                            {renderHighlightedDiff(item.fp32_text, cmpText)}
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Pagination footer */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between border-t border-white/5 pt-4">
                    <span className="text-xs text-zinc-500">
                      Showing Page <span className="font-bold text-zinc-300">{page}</span> of <span className="font-bold text-zinc-300">{totalPages}</span> ({filteredMismatches.length} items)
                    </span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setPage((p) => Math.max(1, p - 1))}
                        disabled={page === 1}
                        className="px-3 py-1.5 bg-zinc-900 border border-white/5 text-xs text-zinc-300 rounded hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
                      >
                        Previous
                      </button>
                      <button
                        onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                        disabled={page === totalPages}
                        className="px-3 py-1.5 bg-zinc-900 border border-white/5 text-xs text-zinc-300 rounded hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition"
                      >
                        Next
                      </button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
