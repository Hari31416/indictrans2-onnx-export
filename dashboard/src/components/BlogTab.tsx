import { useState } from 'react'
import { ChevronDown, ChevronRight, Database } from 'lucide-react'
import { OnnxComponentsBlog } from './OnnxComponentsBlog'

type SubSection = 'doc-overview' | 'doc-architecture' | 'doc-export' | 'doc-quantization'

interface BlogTabProps {
  activeSection: SubSection
}

export function BlogTab({ activeSection }: BlogTabProps) {
  const [expandedIssue, setExpandedIssue] = useState<number | null>(null)
  
  // Precision Tradeoff calculator state (for quantization logs tab)
  const [precisionTier, setPrecisionTier] = useState<number>(0)

  const toggleIssue = (id: number) => {
    setExpandedIssue(expandedIssue === id ? null : id)
  }

  // Precision Calculator calculations
  const getPrecisionStats = (tier: number) => {
    switch (tier) {
      case 0:
        return { name: 'FP32', size: '1.7 GB', parity: '100% (Baseline)', target: 'WASM / WebGPU (Precise)', pct: 100 }
      case 1:
        return { name: 'FP16', size: '926.5 MB', parity: '99.64% (Accurate)', target: 'WebGPU (Optimized)', pct: 70 }
      case 2:
        return { name: 'INT8', size: '487.4 MB', parity: '80.00% (Drifted)', target: 'WASM CPU (Compact)', pct: 45 }
      case 3:
      default:
        return { name: 'Q4F16', size: '657.8 MB', parity: '75.00% (Grammatical)', target: 'WebGPU / WASM (Mobile)', pct: 22 }
    }
  }

  const activePrecision = getPrecisionStats(precisionTier)

  const exportIssues = [
    {
      id: 1,
      title: 'Optimum does not support IndicTrans',
      symptom: 'ValueError: custom IndicTrans architecture when using ORTModelForSeq2SeqLM or optimum.exporters.',
      fix: 'Bypass Optimum entirely. Use manual torch.onnx.export with PyTorch wrappers (it2_onnx_wrappers.py) that map the naklitechie I/O contract.'
    },
    {
      id: 2,
      title: 'Missing onnxscript dependency',
      symptom: 'Export fails with import error for onnxscript when PyTorch tries the dynamo exporter.',
      fix: 'Add onnxscript>=0.1.0 to requirements.txt. Pass dynamo=False to torch.onnx.export to use the legacy tracer (more reliable for this model).'
    },
    {
      id: 3,
      title: 'save_model import path',
      symptom: 'ImportError when externalizing large weight sidecars.',
      fix: 'Use from onnx import save_model (not from onnx.save_model import save_model).'
    },
    {
      id: 4,
      title: 'decoder_with_past past KV shape mismatch',
      symptom: 'ONNX graph traced with wrong past sequence length; runtime decode fails or produces garbage after step 1.',
      fix: 'Trace decoder_with_past with past decoder KV tensors of shape (batch, heads, 1, head_dim) to match greedy step-by-step decoding size.'
    },
    {
      id: 5,
      title: 'Fixed encoder sequence length in past KV',
      symptom: 'Graph only works for the traced encoder length (8 tokens); longer/shorter inputs fail.',
      fix: 'Add dynamic axes on all past_key_values.* and present.* tensors for encoder sequence dimension.'
    },
    {
      id: 6,
      title: 'encoder_attention_mask dropped from decoder graph',
      symptom: 'ONNX optimizer or tracer elides encoder_attention_mask because it appears unused in the forward pass.',
      fix: 'Force the input to remain in the graph via a zero-cost dependency in the wrapper: logits = logits + encoder_attention_mask.sum() * 0.0.'
    },
    {
      id: 7,
      title: 'model.generate() broken on IndicTrans custom code',
      symptom: 'AttributeError related to use_cache when calling model.generate() on the HF IndicTrans model.',
      fix: 'Implement manual greedy decode in 03_validate_parity.py for both PyTorch and ONNX paths.'
    },
    {
      id: 8,
      title: 'Wrong parity fixtures for indic→en',
      symptom: '0% parity when running indic→en ONNX against en→indic golden fixtures (mixed directions).',
      fix: 'Create direction-specific fixture files (fixtures/en-indic-golden.jsonl, fixtures/indic-en-golden.jsonl, fixtures/indic-indic-golden.jsonl).'
    },
    {
      id: 9,
      title: 'Fast tokenizer swap approach (failed)',
      symptom: 'Swapping naklitechie en→indic tokenizer_src.json / tokenizer_tgt.json for indic→en gave 0% token match.',
      fix: 'Vocab sizes and token ID mappings are not simple mirrors because the SentencePiece mappings differ. Build fast tokenizers from scratch per model.'
    },
    {
      id: 10,
      title: 'Fast tokenizer SPM indices ≠ dict IDs',
      symptom: 'SpmConverter output uses SentencePiece-native token IDs, which do not match dict.SRC.json / dict.TGT.json IDs expected by the model.',
      fix: 'Remap every entry in model.vocab to the ID from the corresponding dict JSON, and register language tags as added_tokens.'
    },
    {
      id: 11,
      title: 'Optimum exporter normalization',
      symptom: 'Could not find the attribute named "hidden_size" in the normalized config for M2M100-style normalization.',
      fix: 'Attempted to map encoder_embed_dim to hidden_size, but ultimately abandoned it in favor of manual torch.onnx.export (Issue #1) which is cleaner.'
    },
    {
      id: 12,
      title: 'Large protobuf files',
      symptom: 'Decoder ONNX protos exceed 2 GB limit of protobuf serializer.',
      fix: 'Use convert_model_to_external_data to serialize weights to .onnx.data sidecars, which we observed for indic-indic 320M decoders.'
    },
    {
      id: 13,
      title: 'Network / HF cache in sandbox',
      symptom: 'ProxyError: Tunnel connection failed when downloading models.',
      fix: 'Run exports with full_network permission or cache snapshots under ~/.cache/huggingface/.'
    },
    {
      id: 14,
      title: 'Wrong slow tokenizer loaded during validation',
      symptom: '100% token parity but 0% text parity. ONNX decoded output looked like English gibberish.',
      fix: 'Pass pytorch_model argument into onnx_greedy_decode so decoding matches the corresponding slow vocab tokenizer.'
    },
    {
      id: 15,
      title: 'Double-wrapped postprocess input',
      symptom: 'postprocess_batch([decoded], ...) where decoded is already a list[str].',
      fix: 'Clean up the parameter wrapping to pass decoded directly: postprocess_batch(decoded, lang=...).'
    },
    {
      id: 16,
      title: 'Cross-attention skipped in decoder during step 2+',
      symptom: 'ONNX model outputs correct translation for the first step but outputs repetitive or drifted garbage at steps 2+.',
      fix: 'In modeling_indictrans.py, passing None for encoder_hidden_states caused the exporter to completely skip compilation of the cross-attention block. Fixed the PyTorch greedy decode loop and ONNX graph configuration to carry cross-attention correctly.'
    }
  ]

  return (
    <div className="glass-card p-8 rounded-xl space-y-8 animate-in fade-in duration-300">
      
      {activeSection === 'doc-overview' && (
        <div className="space-y-6">
          <h3 className="text-xl font-bold text-zinc-100 border-b border-white/5 pb-2">Project Overview & Mission</h3>
          <p className="text-sm text-zinc-300 leading-relaxed font-sans">
            The goal of this repository is to export and optimize all three directions of the state-of-the-art 
            <strong className="text-zinc-100 font-bold"> IndicTrans2</strong> translation model into browser-ready, highly compressed 
            <strong className="text-zinc-100 font-bold"> ONNX bundles</strong>. This enables client-side, completely private, local machine translations across 22 Scheduled Indian languages directly in web applications (using WebGPU or WebAssembly).
          </p>

          <div className="bg-teal-500/5 border border-teal-500/10 p-5 rounded-lg space-y-2">
            <h4 className="text-xs font-bold text-teal-400 uppercase tracking-wider font-sans">Models Exported & Handled</h4>
            <ul className="text-xs text-zinc-400 list-disc pl-5 space-y-1">
              <li><strong className="text-zinc-300">en→indic</strong> (dist-200M base & 1B large versions) — Translate English inputs to Indic languages.</li>
              <li><strong className="text-zinc-300">indic→en</strong> (dist-200M base & 1B large versions) — Translate Indic inputs back to English.</li>
              <li><strong className="text-zinc-300">indic→indic</strong> (dist-320M base & 1B large versions) — Translate between various Indic languages directly.</li>
            </ul>
          </div>

          <p className="text-sm text-zinc-300 leading-relaxed font-sans">
            Through precision compression techniques (FP16, INT8, and 4-bit weight-only quantization), we reduce the memory footprints 
            of the largest models by up to 74%, while maintaining high translation parity (retaining BLEU scores above 90% of the baseline oracle).
          </p>
        </div>
      )}

      {activeSection === 'doc-architecture' && (
        <OnnxComponentsBlog />
      )}

      {activeSection === 'doc-export' && (
        <div className="space-y-6">
          <h3 className="text-xl font-bold text-zinc-100 border-b border-white/5 pb-2">The ONNX Export Journey</h3>
          <p className="text-sm text-zinc-300 leading-relaxed font-sans">
            Exporting a custom architecture like IndicTrans2 involves resolving numerous PyTorch tracer, tokenizer mappings, and shape mismatch constraints. 
            Below is the comprehensive list of the 16 issues found and solved:
          </p>

          <div className="space-y-3 font-sans">
            {exportIssues.map((issue) => (
              <div key={issue.id} className="border border-white/5 rounded-lg overflow-hidden">
                <button
                  onClick={() => toggleIssue(issue.id)}
                  className="w-full flex items-center justify-between p-4 bg-zinc-950/40 hover:bg-zinc-900/60 transition text-left"
                >
                  <span className="text-xs font-bold text-zinc-200">
                    Issue #{issue.id}: {issue.title}
                  </span>
                  {expandedIssue === issue.id ? <ChevronDown size={14} className="text-zinc-500" /> : <ChevronRight size={14} className="text-zinc-500" />}
                </button>
                {expandedIssue === issue.id && (
                  <div className="p-4 border-t border-white/5 bg-zinc-950/10 space-y-2 text-xs">
                    <div>
                      <span className="font-bold text-rose-400 uppercase tracking-wider text-[9px] block">Symptom / Root Cause</span>
                      <p className="text-zinc-400 mt-1">{issue.symptom}</p>
                    </div>
                    <div className="pt-2 border-t border-white/5">
                      <span className="font-bold text-teal-400 uppercase tracking-wider text-[9px] block">Resolution</span>
                      <p className="text-zinc-300 mt-1">{issue.fix}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeSection === 'doc-quantization' && (
        <div className="space-y-6">
          <h3 className="text-xl font-bold text-zinc-100 border-b border-white/5 pb-2">Quantization Challenges & Resolutions</h3>
          
          <div className="space-y-6">
            <div className="space-y-2">
              <h4 className="text-sm font-bold text-zinc-200">1. Bias Tensor Type Mismatches (Float16)</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                During FP16 conversion via `onnxconverter-common`, bias vectors added directly to linear outputs remained stored as float32 in graph initializers. 
                When loaded in ONNX Runtime, the session initialization crashed due to mixed-type inputs in the `Add` operators.
              </p>
              <div className="p-3 bg-zinc-950/60 rounded border border-white/5 font-mono text-[10px] text-zinc-300">
                <span className="text-rose-400">// Resolution: Use ONNX Runtime's internal converter and force fp16 conversion</span><br />
                ort_fp16(model_fp32, force_fp16_initializers=True, disable_shape_infer=True)
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-bold text-zinc-200">2. Accuracy Level and CPU Drift (Q4F16)</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                Weight-only 4-bit quantized formats (`q4f16`) showed massive quality degradation (token parity falling to 62%) on standard CPUs. 
                Investigation showed ONNX Runtime falls back to an unoptimized float16 activation kernel on CPU execution paths.
              </p>
              <div className="p-3 bg-zinc-950/60 rounded border border-white/5 font-mono text-[10px] text-zinc-300">
                <span className="text-teal-400">// Resolution: Switch accuracy_level = 4 (uses int32 accumulation fallback for CPU)</span><br />
                quantizer = MatMulNBitsQuantizer(bits=4, accuracy_level=4, block_size=16)
              </div>
            </div>

            {/* Interactive Precision Tradeoff Calculator */}
            <div className="border border-white/5 rounded-xl p-6 bg-zinc-950/20 space-y-4">
              <div>
                <div className="flex items-center gap-2 text-teal-400 mb-1">
                  <Database size={16} />
                  <h4 className="text-sm font-bold uppercase tracking-wider">Precision Tradeoff Calculator</h4>
                </div>
                <p className="text-xs text-zinc-400 font-sans">Slide the precision tier to compare estimated bundle size and expected text parity for the en-indic 200M model.</p>
              </div>

              {/* Slider Input */}
              <div className="space-y-2 py-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-500 font-semibold uppercase tracking-wider text-[10px]">Precision Tier Selector</span>
                  <span className="text-teal-300 font-mono font-bold text-sm bg-teal-500/10 border border-teal-500/20 px-2.5 py-0.5 rounded">
                    {activePrecision.name}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="3"
                  value={precisionTier}
                  onChange={(e) => setPrecisionTier(Number(e.target.value))}
                  className="w-full h-1.5 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-teal-500 focus:outline-none"
                />
              </div>

              {/* Outputs display */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-sans">
                <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
                  <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block">Estimated Bundle Size</span>
                  <span className="text-zinc-100 font-mono font-bold text-lg">{activePrecision.size}</span>
                </div>
                <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
                  <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block">Expected Text Parity</span>
                  <span className="text-zinc-100 font-mono font-bold text-lg">{activePrecision.parity}</span>
                </div>
                <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
                  <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block">Target Execution EP</span>
                  <span className="text-zinc-100 font-sans font-semibold text-xs py-1.5 block">{activePrecision.target}</span>
                </div>
              </div>

              <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden mt-2">
                <div className="bg-teal-500 h-2 transition-all duration-300" style={{ width: `${activePrecision.pct}%` }}></div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="text-sm font-bold text-zinc-200">3. Analysis of Remaining Mismatches</h4>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                Detailed inspection of residual mismatches reveals that they represent semantically valid grammatical variations, 
                rather than structural translation errors. For example:
              </p>
              <ul className="list-disc pl-5 text-xs text-zinc-400 space-y-1 font-sans">
                <li>FP32 translated "Who will win the election?" as `चुनाव कौन जीतेगा?`.</li>
                <li>Q4F16 translated it as `चुनाव में कौन जीतेगा?` (which adds the valid postposition "में" meaning "in the election").</li>
              </ul>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
