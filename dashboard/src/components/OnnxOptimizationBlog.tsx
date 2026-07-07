import { useState } from 'react'
import {
  Zap,
  Activity,
  ChevronDown,
  ChevronUp,
  Check,
  Copy,
  Info,
  Scale
} from 'lucide-react'

type DirectionType = 'en-indic' | 'indic-en' | 'indic-indic'
type OptimizationSection = 'dedup' | 'shared-decoder' | 'graph-fusion' | 'external'

interface DirectionData {
  title: string
  modelScale: string
  before: number // MB
  after: number // MB
  savings: string
  layout: {
    file: string
    role: string
    size: string
  }[]
  breakdownBefore: { label: string; size: number; color: string }[]
  breakdownAfter: { label: string; size: number; color: string }[]
}

const directionDetails: Record<DirectionType, DirectionData> = {
  'en-indic': {
    title: 'English → Indic',
    modelScale: 'dist-200M',
    before: 1910,
    after: 1050,
    savings: '45% Size Reduction',
    layout: [
      { file: 'encoder_model.onnx', role: 'Encoder Graph Protobuf', size: '2 MB' },
      { file: 'encoder_model.onnx.data', role: 'Encoder Weight Sidecar', size: '294 MB' },
      { file: 'decoder_model.onnx', role: 'Decoder (Step 1) Graph Protobuf', size: '6 MB' },
      { file: 'decoder_with_past_model.onnx', role: 'Decoder (Step 2+) Graph Protobuf', size: '8 MB' },
      { file: 'decoder_shared.onnx.data', role: 'Shared Decoder Weight Sidecar', size: '710 MB' },
      { file: 'tokenizer_*.json, config.json', role: 'Tokenizers & Model Configs', size: '30 MB' }
    ],
    breakdownBefore: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder (Step 1)', size: 805, color: 'bg-indigo-500' },
      { label: 'Decoder (Step 2+)', size: 767, color: 'bg-violet-500' },
      { label: 'Configs/Other', size: 44, color: 'bg-zinc-600' }
    ],
    breakdownAfter: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder Shared', size: 712, color: 'bg-indigo-500' },
      { label: 'Configs/Other', size: 44, color: 'bg-zinc-600' }
    ]
  },
  'indic-en': {
    title: 'Indic → English',
    modelScale: 'dist-200M',
    before: 1220,
    after: 800,
    savings: '34% Size Reduction',
    layout: [
      { file: 'encoder_model.onnx', role: 'Encoder Graph Protobuf', size: '2 MB' },
      { file: 'encoder_model.onnx.data', role: 'Encoder Weight Sidecar', size: '294 MB' },
      { file: 'decoder_model.onnx', role: 'Decoder (Step 1) Graph Protobuf', size: '4 MB' },
      { file: 'decoder_with_past_model.onnx', role: 'Decoder (Step 2+) Graph Protobuf', size: '5 MB' },
      { file: 'decoder_shared.onnx.data', role: 'Shared Decoder Weight Sidecar', size: '460 MB' },
      { file: 'tokenizer_*.json, config.json', role: 'Tokenizers & Model Configs', size: '35 MB' }
    ],
    breakdownBefore: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder (Step 1)', size: 450, color: 'bg-indigo-500' },
      { label: 'Decoder (Step 2+)', size: 440, color: 'bg-violet-500' },
      { label: 'Configs/Other', size: 36, color: 'bg-zinc-600' }
    ],
    breakdownAfter: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder Shared', size: 470, color: 'bg-indigo-500' },
      { label: 'Configs/Other', size: 36, color: 'bg-zinc-600' }
    ]
  },
  'indic-indic': {
    title: 'Indic → Indic',
    modelScale: 'dist-320M',
    before: 1910,
    after: 1100,
    savings: '42% Size Reduction',
    layout: [
      { file: 'encoder_model.onnx', role: 'Encoder Graph Protobuf', size: '2 MB' },
      { file: 'encoder_model.onnx.data', role: 'Encoder Weight Sidecar', size: '294 MB' },
      { file: 'decoder_model.onnx', role: 'Decoder (Step 1) Graph Protobuf', size: '6 MB' },
      { file: 'decoder_with_past_model.onnx', role: 'Decoder (Step 2+) Graph Protobuf', size: '8 MB' },
      { file: 'decoder_shared.onnx.data', role: 'Shared Decoder Weight Sidecar', size: '746 MB' },
      { file: 'tokenizer_*.json, config.json', role: 'Tokenizers & Model Configs', size: '44 MB' }
    ],
    breakdownBefore: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder (Step 1)', size: 805, color: 'bg-indigo-500' },
      { label: 'Decoder (Step 2+)', size: 767, color: 'bg-violet-500' },
      { label: 'Configs/Other', size: 44, color: 'bg-zinc-600' }
    ],
    breakdownAfter: [
      { label: 'Encoder', size: 294, color: 'bg-teal-500' },
      { label: 'Decoder Shared', size: 762, color: 'bg-indigo-500' },
      { label: 'Configs/Other', size: 44, color: 'bg-zinc-600' }
    ]
  }
}

export function OnnxOptimizationBlog() {
  const [selectedDir, setSelectedDir] = useState<DirectionType>('en-indic')
  const [copiedCode, setCopiedCode] = useState<string | null>(null)
  const [expandedSection, setExpandedSection] = useState<OptimizationSection | null>('dedup')

  const currentData = directionDetails[selectedDir]

  const handleCopy = (code: string, id: string) => {
    navigator.clipboard.writeText(code)
    setCopiedCode(id)
    setTimeout(() => setCopiedCode(null), 2000)
  }

  // Get segment percentage width relative to total size
  const getSegmentPct = (value: number, total: number) => {
    return (value / total) * 100
  }

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-xl bg-gradient-to-r from-teal-500/10 via-emerald-500/5 to-zinc-950/20 border border-teal-500/10 p-6 md:p-8 space-y-4 font-sans">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-teal-500/20 rounded-lg text-teal-400 border border-teal-500/30">
            <Zap size={22} className="animate-pulse" />
          </div>
          <div>
            <h3 className="text-xl md:text-2xl font-black text-zinc-100 tracking-tight">Post-Export Size Optimizations</h3>
            <p className="text-xs text-teal-400 font-mono">Automated pipeline in onnx_bundle_optimize.py</p>
          </div>
        </div>
        <p className="text-sm text-zinc-300 leading-relaxed max-w-3xl">
          Deep learning models exported directly from PyTorch contain redundant parameters, duplicate initializers, 
          and bloated tensor schemas. To address this, we implemented an automated size optimization suite that runs 
          post-export. This cuts disk and network payloads in half, keeping translations 100% accurate.
        </p>
      </div>

      {/* Model Selection and Comparative Visualizer */}
      <div className="space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 font-sans">
          <h4 className="text-base md:text-lg font-bold text-zinc-100 flex items-center gap-2">
            <Scale size={18} className="text-teal-400" />
            Interactive Size Visualizer
          </h4>

          {/* Selector Tabs */}
          <div className="flex bg-zinc-950/40 p-1 rounded-lg border border-white/5">
            {(Object.keys(directionDetails) as DirectionType[]).map((dir) => (
              <button
                key={dir}
                onClick={() => setSelectedDir(dir)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold uppercase tracking-wider transition ${
                  selectedDir === dir
                    ? 'bg-teal-500 text-zinc-950 font-bold shadow'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                {directionDetails[dir].title}
              </button>
            ))}
          </div>
        </div>

        {/* Before and After comparative bars */}
        <div className="space-y-6 bg-zinc-950/30 border border-white/5 p-6 rounded-xl font-sans">
          <div className="flex items-center justify-between text-xs font-semibold text-zinc-400 border-b border-white/5 pb-3">
            <span>Model Scale: <strong className="text-zinc-200">{currentData.modelScale}</strong></span>
            <span className="text-teal-400 font-bold font-mono">{currentData.savings}</span>
          </div>

          {/* Graph Legend */}
          <div className="flex flex-wrap gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-teal-500"></div>
              <span className="text-zinc-400">Encoder (294 MB)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-indigo-500"></div>
              <span className="text-zinc-400">Decoder Model (Step 1)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-violet-500"></div>
              <span className="text-zinc-400">Decoder Model (Step 2+)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded bg-zinc-600"></div>
              <span className="text-zinc-400">Configs & Tokenizers</span>
            </div>
          </div>

          <div className="space-y-6 pt-2">
            {/* Before Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-zinc-400 font-bold">Before Optimization</span>
                <span className="text-zinc-300 font-bold">{(currentData.before / 1000).toFixed(2)} GB</span>
              </div>
              <div className="w-full bg-zinc-900/60 rounded-lg h-7 flex overflow-hidden border border-white/5">
                {currentData.breakdownBefore.map((seg, idx) => (
                  <div
                    key={idx}
                    style={{ width: `${getSegmentPct(seg.size, currentData.before)}%` }}
                    className={`${seg.color} hover:brightness-110 transition cursor-help relative`}
                    title={`${seg.label}: ${seg.size} MB`}
                  >
                    {getSegmentPct(seg.size, currentData.before) > 10 && (
                      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-zinc-950 font-sans truncate px-1">
                        {seg.size}M
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* After Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-teal-400 font-bold">After Optimization (Unified Layout)</span>
                <span className="text-teal-400 font-bold">{(currentData.after / 1000).toFixed(2)} GB</span>
              </div>
              <div className="w-full bg-zinc-900/60 rounded-lg h-7 flex overflow-hidden border border-teal-500/20 shadow-[0_0_10px_rgba(20,184,166,0.05)]">
                {currentData.breakdownAfter.map((seg, idx) => (
                  <div
                    key={idx}
                    style={{ width: `${getSegmentPct(seg.size, currentData.after)}%` }}
                    className={`${seg.color} hover:brightness-110 transition cursor-help relative`}
                    title={`${seg.label}: ${seg.size} MB`}
                  >
                    {getSegmentPct(seg.size, currentData.after) > 10 && (
                      <span className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-zinc-950 font-sans truncate px-1">
                        {seg.size}M
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Current Layout File Table */}
          <div className="space-y-3 pt-4 border-t border-white/5">
            <div className="text-xs font-bold text-zinc-300 uppercase tracking-wider">Optimized File Layout</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {currentData.layout.map((item, idx) => (
                <div key={idx} className="flex justify-between items-center bg-zinc-950/40 p-3 rounded-lg border border-white/5">
                  <div className="space-y-0.5 text-left">
                    <span className="text-xs font-mono font-bold text-zinc-200">{item.file}</span>
                    <span className="text-[10px] text-zinc-500 block">{item.role}</span>
                  </div>
                  <span className="text-xs font-mono font-bold text-teal-400 bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 rounded">
                    {item.size}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Optimizations Details Accordions */}
      <div className="space-y-4">
        <h4 className="text-base md:text-lg font-bold text-zinc-100 flex items-center gap-2">
          <Activity size={18} className="text-teal-400" />
          Optimizations Implemented in the Codebase
        </h4>

        <div className="space-y-3 font-sans">
          {/* 1. Tied-Weight Dedup */}
          <div className={`border rounded-xl transition-all duration-200 ${expandedSection === 'dedup' ? 'border-teal-500/20 bg-zinc-950/30' : 'border-white/5 bg-zinc-950/10 hover:bg-zinc-950/20'}`}>
            <button
              onClick={() => setExpandedSection(expandedSection === 'dedup' ? null : 'dedup')}
              className="w-full flex items-center justify-between p-5 text-left"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest bg-teal-500/10 px-1.5 py-0.5 rounded">Step 1</span>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider font-mono">Impact: ~−500 MB (Tied directions only)</span>
                </div>
                <h5 className="text-sm font-bold text-zinc-100">1. Tied-Weight Deduplication (Post-Export)</h5>
              </div>
              <span className="text-zinc-500">
                {expandedSection === 'dedup' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </span>
            </button>

            {expandedSection === 'dedup' && (
              <div className="px-5 pb-5 pt-1 border-t border-white/5 space-y-4 text-xs md:text-sm">
                <p className="text-zinc-300 leading-relaxed font-sans">
                  IndicTrans2 models tie the decoder embedding layers (`decoder.embed_tokens.weight`) and the vocabulary 
                  projection layer (`lm_head.weight`) for the <strong>en→indic</strong> and <strong>indic→en</strong> directions. 
                  During a normal export, PyTorch serializes these weights twice, resulting in two separate 251 MB matrices per decoder.
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans font-bold">
                  Our Solution:
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans">
                  The script `src/onnx_bundle_optimize.py` loads the model, verifies that these two matrices are transposes of each other within 
                  an absolute tolerance, removes the duplicate embedding weights from initializers, adds an ONNX <code>Transpose</code> layer 
                  for the vocabulary projection MatMul input, and routes both operations through a single shared <code>lm_head.weight</code> initializer.
                </p>
                <div className="p-3 bg-zinc-950/60 rounded border border-white/5 space-y-2 text-xs leading-relaxed text-zinc-400 font-sans">
                  <div className="flex items-center gap-1.5 text-zinc-200 font-bold">
                    <Info size={12} className="text-teal-400" />
                    <span>Pipeline Rule</span>
                  </div>
                  <p>
                    Tied-weight dedup is automatically skipped for the <code>indic→indic</code> direction since it has separate, 
                    untied embedding and vocab projection weights.
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-zinc-400 font-mono">
                    <span>onnx_bundle_optimize.py</span>
                    <button
                      onClick={() => handleCopy(`def dedup_tied_embed_weights(model: ModelProto, base_dir: Path) -> tuple[ModelProto, bool]:
    gather_name = _vocab_gather_weight_name(model)
    matmul_name = _vocab_projection_name(model)
    ...
    # Remove duplicate MatMul weights, replace with transpose routing
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[canonical_name],
        outputs=[transpose_output],
        name="/lm_head/weight_transpose",
        perm=[1, 0],
    )
    matmul_node.input[i] = transpose_output
    model.graph.initializer.remove(matmul_init)
    return model, True`, 'dedup_code')}
                      className="flex items-center gap-1 text-zinc-500 hover:text-teal-400 transition"
                    >
                      {copiedCode === 'dedup_code' ? (
                        <>
                          <Check size={10} className="text-teal-400" />
                          <span className="text-teal-400">Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy size={10} />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre className="bg-zinc-950/60 p-4 rounded-lg border border-white/5 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
                    <code>{`def dedup_tied_embed_weights(model: ModelProto, base_dir: Path) -> tuple[ModelProto, bool]:
    gather_name = _vocab_gather_weight_name(model)
    matmul_name = _vocab_projection_name(model)
    ...
    # Remove duplicate MatMul weights, replace with transpose routing
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[canonical_name],
        outputs=[transpose_output],
        name="/lm_head/weight_transpose",
        perm=[1, 0],
    )
    matmul_node.input[i] = transpose_output
    model.graph.initializer.remove(matmul_init)
    return model, True`}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* 2. Shared Decoder Sidecar */}
          <div className={`border rounded-xl transition-all duration-200 ${expandedSection === 'shared-decoder' ? 'border-teal-500/20 bg-zinc-950/30' : 'border-white/5 bg-zinc-950/10 hover:bg-zinc-950/20'}`}>
            <button
              onClick={() => setExpandedSection(expandedSection === 'shared-decoder' ? null : 'shared-decoder')}
              className="w-full flex items-center justify-between p-5 text-left"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest bg-teal-500/10 px-1.5 py-0.5 rounded">Step 2</span>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider font-mono">Impact: ~−550 MB (All directions)</span>
                </div>
                <h5 className="text-sm font-bold text-zinc-100">2. Shared Decoder Sidecar Weights</h5>
              </div>
              <span className="text-zinc-500">
                {expandedSection === 'shared-decoder' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </span>
            </button>

            {expandedSection === 'shared-decoder' && (
              <div className="px-5 pb-5 pt-1 border-t border-white/5 space-y-4 text-xs md:text-sm">
                <p className="text-zinc-300 leading-relaxed font-sans">
                  The autoregressive decoding pipeline requires two separate decoder ONNX graphs: <code>decoder_model.onnx</code> 
                  (first decode step) and <code>decoder_with_past_model.onnx</code> (steps 2+). Because they share over 95% of layer parameters, 
                  exporting them independently generates duplicate weight binaries.
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans font-bold">
                  Our Solution:
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans">
                  We write weights from both models to a single shared file: <code>decoder_shared.onnx.data</code>. 
                  Tensors are analyzed, deduplicated based on their SHA-256 byte hashes, and appended into this single buffer. 
                  We then rewrite initializer external storage configurations to point to this unified file using exact byte offsets and lengths.
                </p>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-widest text-zinc-400 font-mono">
                    <span>onnx_bundle_optimize.py</span>
                    <button
                      onClick={() => handleCopy(`def share_decoder_external_data(output_dir: Path) -> bool:
    models = [_load_model(p) for p in decoder_paths]
    shared_path = output_dir / "decoder_shared.onnx.data"
    
    blob_store: dict[str, tuple[int, int]] = {}
    shared_bytes = bytearray()
    
    # Content-address weights and map external locations
    for model, onnx_path in zip(models, decoder_paths):
        for tensor in model.graph.initializer:
            raw = _tensor_raw_bytes(tensor, base_dir)
            digest, offset, length = intern_bytes(raw)
            _assign_external_tensor(tensor, "decoder_shared.onnx.data", offset, length)
            
    shared_path.write_bytes(shared_bytes)
    return True`, 'shared_code')}
                      className="flex items-center gap-1 text-zinc-500 hover:text-teal-400 transition"
                    >
                      {copiedCode === 'shared_code' ? (
                        <>
                          <Check size={10} className="text-teal-400" />
                          <span className="text-teal-400">Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy size={10} />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre className="bg-zinc-950/60 p-4 rounded-lg border border-white/5 font-mono text-xs text-zinc-300 overflow-x-auto leading-relaxed">
                    <code>{`def share_decoder_external_data(output_dir: Path) -> bool:
    models = [_load_model(p) for p in decoder_paths]
    shared_path = output_dir / "decoder_shared.onnx.data"
    
    blob_store: dict[str, tuple[int, int]] = {}
    shared_bytes = bytearray()
    
    # Content-address weights and map external locations
    for model, onnx_path in zip(models, decoder_paths):
        for tensor in model.graph.initializer:
            raw = _tensor_raw_bytes(tensor, base_dir)
            digest, offset, length = intern_bytes(raw)
            _assign_external_tensor(tensor, "decoder_shared.onnx.data", offset, length)
            
    shared_path.write_bytes(shared_bytes)
    return True`}</code>
                  </pre>
                </div>
              </div>
            )}
          </div>

          {/* 3. ORT Graph Fusion */}
          <div className={`border rounded-xl transition-all duration-200 ${expandedSection === 'graph-fusion' ? 'border-teal-500/20 bg-zinc-950/30' : 'border-white/5 bg-zinc-950/10 hover:bg-zinc-950/20'}`}>
            <button
              onClick={() => setExpandedSection(expandedSection === 'graph-fusion' ? null : 'graph-fusion')}
              className="w-full flex items-center justify-between p-5 text-left"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest bg-teal-500/10 px-1.5 py-0.5 rounded">Step 3</span>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider font-mono">Impact: ~60% node reduction, faster execution</span>
                </div>
                <h5 className="text-sm font-bold text-zinc-100">3. ONNX Runtime Graph Fusion</h5>
              </div>
              <span className="text-zinc-500">
                {expandedSection === 'graph-fusion' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </span>
            </button>

            {expandedSection === 'graph-fusion' && (
              <div className="px-5 pb-5 pt-1 border-t border-white/5 space-y-4 text-xs md:text-sm">
                <p className="text-zinc-300 leading-relaxed font-sans">
                  The raw ONNX graphs compiled directly by PyTorch are full of redundant nodes, sub-optimal operators, 
                  and intermediate calculations.
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans font-bold">
                  Our Solution:
                </p>
                <p className="text-zinc-300 leading-relaxed font-sans">
                  We integrate ONNX Runtime\'s internal transformer optimizer to perform compiler passes directly on the graph. 
                  This fuses redundant nodes (such as MatMul/Add layers, Multi-Head Attention equations, and LayerNorm paths). 
                  For example, <code>decoder_model</code>\'s node count drops from <strong>3,696 → 1,662 nodes</strong>, 
                  cutting graph metadata sizes and decreasing browser initialization latencies.
                </p>
              </div>
            )}
          </div>

          {/* 4. Weight Externalization */}
          <div className={`border rounded-xl transition-all duration-200 ${expandedSection === 'external' ? 'border-teal-500/20 bg-zinc-950/30' : 'border-white/5 bg-zinc-950/10 hover:bg-zinc-950/20'}`}>
            <button
              onClick={() => setExpandedSection(expandedSection === 'external' ? null : 'external')}
              className="w-full flex items-center justify-between p-5 text-left"
            >
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-bold text-teal-400 uppercase tracking-widest bg-teal-500/10 px-1.5 py-0.5 rounded">Step 4</span>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider font-mono">Impact: Keeps graph protobufs browseable</span>
                </div>
                <h5 className="text-sm font-bold text-zinc-100">4. 100 MB Weight Externalization</h5>
              </div>
              <span className="text-zinc-500">
                {expandedSection === 'external' ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </span>
            </button>

            {expandedSection === 'external' && (
              <div className="px-5 pb-5 pt-1 border-t border-white/5 space-y-4 text-xs md:text-sm font-sans">
                <p className="text-zinc-300 leading-relaxed">
                  Protobuf limits model configurations to 2 GiB. Storing hundreds of megabytes of weight parameters inline 
                  causes loading and serialization lags when downloading or browsing graphs on platforms like Hugging Face.
                </p>
                <p className="text-zinc-300 leading-relaxed font-bold">
                  Our Solution:
                </p>
                <p className="text-zinc-300 leading-relaxed">
                  We set a threshold of 100 MB. Tensors exceeding this size (specifically the 294 MB encoder parameters) 
                  are automatically extracted from the protobuf and saved into separate <code>.onnx.data</code> sidecar files.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Integration into precision pipeline */}
      <div className="border border-white/5 rounded-xl p-6 bg-zinc-950/20 space-y-4 font-sans text-left">
        <div className="flex items-center gap-2 text-teal-400">
          <Info size={16} />
          <h4 className="text-sm md:text-base font-bold uppercase tracking-wider">Precision Pipeline Integration</h4>
        </div>
        <p className="text-xs text-zinc-400 leading-relaxed">
          The optimization suite doesn't just process full-size FP32 models. It is wired into the entire multi-precision pipeline. 
          After exports complete, the process calls <code>optimize_export_bundle()</code>. 
          When converting or quantizing weights down to Float16 (FP16), INT8, or Q4F16 formats, the scripts automatically call 
          <code>finalize_bundle_layout()</code>. This guarantees that every precision tier retains the shared decoder layout, 
          keeping browser download footprints as compact as possible.
        </p>
      </div>
    </div>
  )
}
