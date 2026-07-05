import { useState } from 'react'
import { Database, Play, ArrowLeft, ArrowRight, Check, Copy } from 'lucide-react'

// Terminology data
interface TermCard {
  id: string
  title: string
  desc: string
}

const terms: TermCard[] = [
  { id: 'opset', title: 'Opset', desc: 'The operator dialect version a graph is built against. Opset 17 is used here for onnxruntime-web compatibility.' },
  { id: 'axes', title: 'Dynamic axes', desc: 'A declaration that a specific tensor dimension (batch, sequence length) may vary at runtime instead of staying fixed at the traced value.' },
  { id: 'init', title: 'Initializer', desc: 'A trained weight tensor stored inside the ONNX protobuf, or in an external data sidecar file when the protobuf would exceed size limits.' },
  { id: 'session', title: 'Inference session', desc: 'A loaded, runnable ONNX graph. You call session.run(outputs, inputs) with named tensors and receive named output tensors back.' }
]

// Simulator steps
interface SimStep {
  title: string
  desc: string
  graph: string
  inputs: string[]
  outputs: string[]
  cache: string
}

const simSteps: SimStep[] = [
  {
    title: 'Encode the source sentence',
    desc: 'The tokenizer turns the source sentence into 5 token ids plus an attention mask of five ones.',
    graph: 'encoder_model.onnx',
    inputs: ['input_ids [1,5]', 'attention_mask [1,5]'],
    outputs: ['last_hidden_state [1,5,512]'],
    cache: 'Cache: empty. Encoder output stored for all later steps.'
  },
  {
    title: 'Decode step 1',
    desc: 'The first decoder graph runs with the start token and the fresh encoder states. It emits the first real token plus the full initial cache.',
    graph: 'decoder_model.onnx',
    inputs: ['input_ids [1,1] (start)', 'encoder_hidden_states [1,5,512]', 'encoder_attention_mask [1,5]'],
    outputs: ['logits [1,1,122672] → token 1', 'present[0..17].decoder.{key,value} [1,8,1,64]', 'present[0..17].encoder.{key,value} [1,8,5,64]'],
    cache: 'Cache: 72 tensors. Decoder half length 1. Encoder half length 5.'
  },
  {
    title: 'Decode step 2',
    desc: 'Control switches to decoder_with_past. It takes token 1, the mask, and the cache. It appends one entry to the decoder half and reuses the fixed encoder half.',
    graph: 'decoder_with_past_model.onnx',
    inputs: ['input_ids [1,1] (token 1)', 'encoder_attention_mask [1,5]', 'past_key_values[0..17].decoder.{key,value} [1,8,1,64]', 'past_key_values[0..17].encoder.{key,value} [1,8,5,64]'],
    outputs: ['logits → token 2', 'present.decoder.{key,value} [1,8,2,64]'],
    cache: 'Cache: decoder half length 2. Encoder half still length 5.'
  },
  {
    title: 'Decode step 3',
    desc: 'The same graph runs again with token 2 and the updated cache. The decoder half grows to length 3.',
    graph: 'decoder_with_past_model.onnx',
    inputs: ['input_ids [1,1] (token 2)', 'past decoder cache [1,8,2,64] per layer', 'past encoder cache [1,8,5,64] per layer'],
    outputs: ['logits → token 3', 'present decoder cache [1,8,3,64] per layer'],
    cache: 'Cache: decoder half length 3. Encoder half still length 5.'
  },
  {
    title: 'Decode step 4',
    desc: 'Token 3 is fed back. The decoder half grows to length 4. No encoder work happens. The fixed encoder cache is reused as-is.',
    graph: 'decoder_with_past_model.onnx',
    inputs: ['input_ids [1,1] (token 3)', 'past decoder cache [1,8,3,64] per layer', 'past encoder cache [1,8,5,64] per layer'],
    outputs: ['logits → token 4 (eos)', 'present decoder cache [1,8,4,64] per layer'],
    cache: 'Token 4 equals eos id 2. The loop stops.'
  },
  {
    title: 'Detokenize the output',
    desc: 'The emitted token sequence [start, 1, 2, 3, eos] is decoded back to text through the target tokenizer. The IndicProcessor postprocess step restores scripts and numerals. Translation complete.',
    graph: 'none (string decode)',
    inputs: ['output token ids'],
    outputs: ['target language text'],
    cache: 'Encoder ran once. decoder_model ran once. decoder_with_past ran three times. Total graph calls: 5.'
  }
]

// Precision calculator tiers
interface PrecisionTier {
  name: string
  ratio: number
  parity: string
  target: string
  pct: number
}

const precisionTiers: PrecisionTier[] = [
  { name: 'fp32', ratio: 1.0, parity: '100% (production)', target: 'WASM / WebGPU', pct: 100 },
  { name: 'fp16', ratio: 0.5, parity: 'projected ~99% (unmeasured)', target: 'WebGPU (Chrome 121+)', pct: 70 },
  { name: 'int8', ratio: 0.25, parity: '~80% (preview tier)', target: 'WASM CPU', pct: 45 },
  { name: 'q4f16', ratio: 0.125, parity: 'projected (unmeasured)', target: 'WebGPU / mobile', pct: 22 }
]

// Manifest files
interface ManifestFile {
  name: string
  size: string
  role: 'graph' | 'weights' | 'tokenizer' | 'config'
  desc: string
}

const manifestFiles: ManifestFile[] = [
  { name: 'encoder_model.onnx', size: '294 MB', role: 'graph', desc: 'The encoder graph. Takes source input_ids and attention_mask, returns last_hidden_state. Runs once per sentence. Weights stay internal because the proto is under the 512 MB externalization threshold.' },
  { name: 'decoder_model.onnx', size: '1.2 MB', role: 'graph', desc: 'The first step decoder graph. Takes the start token, encoder hidden states, and encoder attention mask. Returns logits plus the full initial KV cache (72 present tensors for 18 layers). Graph structure only, weights are in the sidecar.' },
  { name: 'decoder_model.onnx.data', size: '805 MB', role: 'weights', desc: 'External weight sidecar for decoder_model.onnx. Holds all decoder weight tensors moved out of the protobuf to stay under the 2 GB proto limit. Must ship next to its .onnx file.' },
  { name: 'decoder_with_past_model.onnx', size: '1.2 MB', role: 'graph', desc: 'The autoregressive decoder graph for steps 2 and onward. Takes the previous token, encoder attention mask, and the past KV cache. Returns logits and the updated cache. Uses a dummy encoder hidden states tensor and a zero-cost mask dependency to keep cross attention in the graph.' },
  { name: 'decoder_with_past_model.onnx.data', size: '767 MB', role: 'weights', desc: 'External weight sidecar for decoder_with_past_model.onnx. Inseparable from its .onnx file at load time.' },
  { name: 'tokenizer_src.json', size: '3.9 MB', role: 'tokenizer', desc: 'Fast source tokenizer in the tokenizers json format. Converted from model.SRC via SpmConverter, then remapped to dict.SRC.json ids. Used by the browser to encode source text into token ids.' },
  { name: 'tokenizer_tgt.json', size: '23.8 MB', role: 'tokenizer', desc: 'Fast target tokenizer. Converted from model.TGT and remapped to dict.TGT.json ids. Used to decode generated token ids back into target language text.' },
  { name: 'tokenizer_meta.json', size: '70 B', role: 'tokenizer', desc: 'Holds src_dict_size, tgt_dict_size, and unk_id (3). The runtime clamps any id at or above the dict size down to the unknown token to guard against untrained vocabulary entries.' },
  { name: 'model.SRC', size: '0.8 MB', role: 'tokenizer', desc: 'Original SentencePiece model for the source side. Kept so the slow Python tokenizer can still run for validation and parity checks.' },
  { name: 'model.TGT', size: '3.3 MB', role: 'tokenizer', desc: 'Original SentencePiece model for the target side. Same role as model.SRC for the target vocabulary.' },
  { name: 'dict.SRC.json', size: '0.6 MB', role: 'tokenizer', desc: 'Fairseq source dictionary mapping tokens to the ids the model was trained on. The fast tokenizer is remapped against this file so emitted ids match the training vocabulary.' },
  { name: 'dict.TGT.json', size: '3.4 MB', role: 'tokenizer', desc: 'Fairseq target dictionary. Same role as dict.SRC.json for the target vocabulary.' },
  { name: 'tokenization_indictrans.py', size: '8.0 KB', role: 'tokenizer', desc: 'The slow Python tokenizer implementation from AI4Bharat. Loaded with trust_remote_code=True during validation and postprocessing. Not used by the browser path.' },
  { name: 'tokenizer_config.json', size: '1.1 KB', role: 'tokenizer', desc: 'Hugging Face tokenizer configuration metadata. Needed to instantiate the slow tokenizer correctly in Python.' },
  { name: 'special_tokens_map.json', size: '96 B', role: 'tokenizer', desc: 'Declares the special tokens pad, bos, eos, unk and their mapping. Consumed by the slow tokenizer loader.' },
  { name: 'config.json', size: '1.4 KB', role: 'config', desc: 'Architecture constants: 18 encoder layers, 18 decoder layers, 8 heads, 512 embed dim, vocab sizes, token ids. Loaders and validation read it to recover layer count and dimensions.' },
  { name: 'generation_config.json', size: '163 B', role: 'config', desc: 'Decoding parameters. The two that matter for the greedy loop are decoder_start_token_id (2) and eos_token_id (2). Lets the bundle self-describe its generation defaults.' }
]

const roleColors: Record<string, string> = {
  graph: '#0d9488', // teal-600
  weights: '#0f766e', // teal-700
  tokenizer: '#f59e0b', // amber
  config: '#71717a' // zinc
}

// Copy Code Block helper component
function CodeBlock({ code, label }: { code: string; label: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <div className="border border-white/5 rounded-lg overflow-hidden bg-zinc-950/40 my-4 font-mono text-xs md:text-sm">
      <div className="flex items-center justify-between px-4 py-2 border-b border-white/5 bg-zinc-900/40">
        <span className="text-zinc-500 font-sans text-[10px] uppercase tracking-wider">{label}</span>
        <button onClick={handleCopy} className="text-zinc-400 hover:text-teal-400 transition flex items-center gap-1">
          {copied ? (
            <>
              <Check size={12} className="text-teal-400" />
              <span className="text-teal-400 text-[10px]">Copied!</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span className="text-[10px]">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-zinc-300 text-xs md:text-sm leading-relaxed"><code>{code}</code></pre>
    </div>
  )
}

// Interactive Quiz Component
function InteractiveQuiz({
  question,
  options,
  correctIndex,
  correctFeedback,
  incorrectFeedback,
  tip
}: {
  question: string
  options: string[]
  correctIndex: number
  correctFeedback: string
  incorrectFeedback: string
  tip: string
}) {
  const [selected, setSelected] = useState<number | null>(null)
  const [submitted, setSubmitted] = useState(false)

  const handleRetry = () => {
    setSelected(null)
    setSubmitted(false)
  }

  const isCorrect = selected === correctIndex

  return (
    <div className="border border-white/5 rounded-xl p-5 bg-zinc-900/20 space-y-4 font-sans text-sm">
      <div className="font-semibold text-zinc-200 text-sm md:text-base">{question}</div>
      <div className="flex flex-col gap-2">
        {options.map((opt, idx) => {
          let optStyle = 'border-white/5 hover:border-zinc-700 bg-zinc-900/35 text-zinc-300'
          if (submitted) {
            if (idx === correctIndex) {
              optStyle = 'border-teal-500/30 bg-teal-500/10 text-teal-300 font-semibold'
            } else if (idx === selected) {
              optStyle = 'border-rose-500/30 bg-rose-500/10 text-rose-400 line-through'
            } else {
              optStyle = 'border-white/5 opacity-55 text-zinc-500'
            }
          } else if (selected === idx) {
            optStyle = 'border-teal-500/30 bg-teal-500/5 text-teal-300'
          }

          return (
            <button
              key={idx}
              disabled={submitted}
              onClick={() => setSelected(idx)}
              className={`w-full text-left p-3 rounded-lg border text-xs md:text-sm transition-all ${optStyle}`}
            >
              {opt}
            </button>
          );
        })}
      </div>

      {!submitted ? (
        <button
          disabled={selected === null}
          onClick={() => setSubmitted(true)}
          className="px-4 py-2 bg-teal-600 hover:bg-teal-500 disabled:opacity-40 disabled:hover:bg-teal-600 text-zinc-950 font-bold text-xs md:text-sm rounded-lg transition"
        >
          Submit Answer
        </button>
      ) : (
        <div className="p-4 rounded-lg bg-zinc-950/40 border border-white/5 space-y-2 text-xs md:text-sm">
          {isCorrect ? (
            <div className="text-teal-400 font-semibold">✓ Correct! {correctFeedback}</div>
          ) : (
            <div className="space-y-3">
              <div className="text-rose-400 font-semibold">✗ Incorrect. {incorrectFeedback}</div>
              <button
                onClick={handleRetry}
                className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-[10px] rounded transition"
              >
                Try Again
              </button>
            </div>
          )}
          <div className="text-zinc-500 italic pt-1 border-t border-white/5 text-[10px]">{tip}</div>
        </div>
      )}
    </div>
  )
}

export function OnnxComponentsBlog() {
  const [flipped, setFlipped] = useState<Record<string, boolean>>({})
  const [simStep, setSimStep] = useState(0)
  const [precisionTier, setPrecisionTier] = useState(0)
  const [selectedFile, setSelectedFile] = useState<ManifestFile>(manifestFiles[0])

  const toggleCard = (id: string) => {
    setFlipped((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const activeStep = simSteps[simStep]
  const activePrecision = precisionTiers[precisionTier]
  const FP32_BASE_GB = 1.7
  const OVERHEAD_GB = 0.09
  const sizeGb = Math.max(FP32_BASE_GB * activePrecision.ratio * 0.94 + OVERHEAD_GB, OVERHEAD_GB)
  const sizeText = sizeGb >= 1 ? sizeGb.toFixed(2) + ' GB' : Math.round(sizeGb * 1024) + ' MB'

  return (
    <div className="space-y-8 font-sans text-sm md:text-base text-zinc-300 leading-relaxed max-w-4xl mx-auto">
      
      {/* 1. What ONNX actually is */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">What ONNX actually is</h2>
        <p>
          A trained PyTorch model is a graph of operations expressed in PyTorch's Python API.
          You cannot run that graph directly inside a browser or in a generic C++ runtime without
          dragging PyTorch along. <strong className="text-zinc-100 font-bold">ONNX</strong>, the Open Neural Network Exchange, is an
          intermediate representation for that graph. It stores the same math as a list of
          standardized operators plus their initial weights, serialized into a binary file.
        </p>
        <p>
          An ONNX file is a serialized <strong className="text-zinc-100 font-bold">protobuf</strong> message. It holds three things: a
          list of <strong className="text-zinc-100 font-bold">nodes</strong> (each node is one operator like MatMul, Add, or Softmax),
          a list of <strong className="text-zinc-100 font-bold">initializers</strong> (the trained weight tensors), and metadata
          describing inputs, outputs, and the <strong className="text-zinc-100 font-bold">opset</strong> version. The opset is the
          dialect of operators the graph uses. This bundle targets opset 17, which
          <code className="text-teal-400 bg-teal-500/5 px-1.5 py-0.5 rounded font-mono text-xs md:text-sm">onnxruntime-web</code> 1.21 and later supports in the browser.
        </p>
        <p>
          A separate program called a <strong className="text-zinc-100 font-bold">runtime</strong> loads the protobuf and executes the
          operator graph. <strong className="text-zinc-100 font-bold">onnxruntime</strong> is the reference runtime from Microsoft. It
          ships builds for Python, C++, and JavaScript. The JavaScript build,
          <strong className="text-zinc-100 font-bold">onnxruntime-web</strong>, can run the same file through a WASM CPU provider or a
          WebGPU provider. That portability is the whole point. You export once, then run the file
          anywhere a runtime exists.
        </p>
        <p>
          Export is the act of turning a PyTorch module into an ONNX file. PyTorch's exporter walks
          the module with example inputs, records every tensor operation into the ONNX operator
          vocabulary, and writes the protobuf. This process is called <strong className="text-zinc-100 font-bold">tracing</strong>. The
          example inputs you feed during tracing fix the operations recorded, so you must declare
          which input dimensions are allowed to vary later through <strong className="text-zinc-100 font-bold">dynamic axes</strong>.
        </p>

        <div className="bg-teal-500/5 border border-teal-500/10 p-5 rounded-lg">
          <p className="text-xs md:text-sm text-zinc-400">
            The export pipeline this article describes lives in this repository. It exports the AI4Bharat
            IndicTrans2 translation models into browser-ready ONNX bundles. Each bundle can be consumed by client-side browser translation wrappers. Every component below comes from that real pipeline.
          </p>
        </div>

        {/* Terminology Flashcards */}
        <div className="space-y-3 pt-2">
          <h4 className="text-xs md:text-sm font-bold text-zinc-400 uppercase tracking-widest">ONNX Terminology Guide</h4>
          <p className="text-xs text-zinc-500">Click any card to check its definition.</p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {terms.map((t) => {
              const isFlipped = !!flipped[t.id]
              return (
                <button
                  key={t.id}
                  onClick={() => toggleCard(t.id)}
                  className="h-28 text-center p-4 glass-card rounded-xl border border-white/5 relative flex flex-col items-center justify-center cursor-pointer transition select-none hover:border-teal-500/30 overflow-hidden"
                >
                  {!isFlipped ? (
                    <div className="space-y-1">
                      <span className="text-[9px] font-bold text-teal-400 uppercase tracking-widest block">Concept</span>
                      <span className="text-xs md:text-sm font-bold text-zinc-100">{t.title}</span>
                    </div>
                  ) : (
                    <div className="text-[10px] md:text-xs text-zinc-300 leading-tight">
                      {t.desc}
                    </div>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </section>

      {/* 2. The model you start with */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The model you start with</h2>
        <h3 className="text-base md:text-lg font-bold text-zinc-200">Encoder, decoder, and autoregression</h3>
        <p>
          IndicTrans2 is a <strong className="text-zinc-100 font-bold">sequence-to-sequence</strong> transformer. It translates a
          source sentence into a target sentence. The model has two halves. The
          <strong className="text-zinc-100 font-bold">encoder</strong> reads the source tokens and produces a contextual embedding for
          each one. The <strong className="text-zinc-100 font-bold">decoder</strong> generates target tokens one at a time, attending
          back to the encoder outputs at every step.
        </p>
        <p>
          Generation is <strong className="text-zinc-100 font-bold">autoregressive</strong>. The decoder produces token 1, then token 2
          conditioned on token 1, then token 3 conditioned on tokens 1 and 2, and so on until it
          emits an end of sentence token. Each new token depends on every previously generated
          token. This dependency is what makes the export split into multiple graphs.
        </p>

        {/* SVG Diagram 1 */}
        <div className="bg-zinc-950/40 p-5 rounded-xl border border-white/5 flex flex-col items-center">
          <svg viewBox="0 0 620 230" className="w-full max-w-xl" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="arrowB1" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
                <path d="M0,0 L0,6 L8,3 z" fill="#14b8a6" />
              </marker>
            </defs>
            <g>
              <rect x="20" y="40" width="170" height="150" rx="10" fill="rgba(13, 148, 136, 0.03)" stroke="#0d9488" strokeWidth="1.5" />
              <text x="105" y="70" textAnchor="middle" fontFamily="sans-serif" fontSize="14" fontWeight="600" fill="#f4f4f5">Encoder</text>
              <text x="105" y="115" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#e4e4e7">input_ids</text>
              <text x="105" y="135" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#e4e4e7">attention_mask</text>
              <text x="105" y="165" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#2dd4bf">→ last_hidden_state</text>
            </g>
            <line x1="190" y1="115" x2="250" y2="115" stroke="#0d9488" strokeWidth="1.5" markerEnd="url(#arrowB1)" />
            <text x="220" y="105" textAnchor="middle" fontFamily="sans-serif" fontSize="10" fill="#2dd4bf">hidden states</text>
            <g>
              <rect x="250" y="40" width="180" height="150" rx="10" fill="rgba(13, 148, 136, 0.03)" stroke="#0d9488" strokeWidth="1.5" />
              <text x="340" y="70" textAnchor="middle" fontFamily="sans-serif" fontSize="14" fontWeight="600" fill="#f4f4f5">Decoder</text>
              <text x="340" y="115" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#e4e4e7">prev token</text>
              <text x="340" y="135" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#e4e4e7">+ KV cache</text>
              <text x="340" y="165" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#2dd4bf">→ logits</text>
            </g>
            <line x1="430" y1="115" x2="490" y2="115" stroke="#0d9488" strokeWidth="1.5" markerEnd="url(#arrowB1)" />
            <g>
              <rect x="490" y="75" width="110" height="80" rx="10" fill="rgba(245, 158, 11, 0.03)" stroke="#f59e0b" strokeWidth="1.5" />
              <text x="545" y="105" textAnchor="middle" fontFamily="sans-serif" fontSize="12" fontWeight="600" fill="#f4f4f5">argmax</text>
              <text x="545" y="125" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#f59e0b">next token</text>
            </g>
            <path d="M545,155 Q545,210 340,210 Q340,180 340,190" fill="none" stroke="#64748b" strokeWidth="1.5" strokeDasharray="4 3" markerEnd="url(#arrowB1)" />
            <text x="440" y="225" textAnchor="middle" fontFamily="sans-serif" fontSize="10" fill="#a1a1aa">feed back as prev token</text>
          </svg>
          <span className="text-[10px] text-zinc-500 mt-2">Autoregressive decoding: The encoder runs once. The decoder runs repeatedly, fed its own previous output, until it emits the end of sentence token.</span>
        </div>

        <h3 className="text-base md:text-lg font-bold text-zinc-200">IndicTrans2 architecture constants</h3>
        <p>
          The export scripts hardcode a few constants that must match the model's <code className="text-teal-400 font-mono text-xs md:text-sm">config.json</code>. 
          For the en to indic 200M model, the encoder and decoder each have 18 layers. Each layer has 8 attention heads. 
          The embedding dimension is 512, so each head has dimension 64. These numbers drive the shape of every past and present KV tensor 
          in the exported graphs.
        </p>
        <p>
          The decoder starts from token id 2, the <code className="text-zinc-400 font-mono">decoder_start_token_id</code>, and stops when
          it emits token id 2, the <code className="text-zinc-400 font-mono">eos_token_id</code>. The pad token is id 1, the bos token
          is id 0, and the unknown token is id 3. The source vocabulary holds 32322 entries. The
          target vocabulary holds 122672 entries.
        </p>
      </section>

      {/* 3. Why one model becomes three graphs */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">Why one model becomes three graphs</h2>
        <h3 className="text-base md:text-lg font-bold text-zinc-200">The KV cache problem</h3>
        <p>
          Naive autoregressive decoding recomputes the full decoder forward pass for every new
          token. To generate token N, the decoder reprocesses tokens 1 through N minus 1. That is
          quadratic in the output length and wasteful, because the attention keys and values for
          earlier tokens do not change when a new token is added.
        </p>
        <p>
          The fix is the <strong className="text-zinc-100 font-bold">KV cache</strong>. At each step the decoder computes the key and
          value tensors for the current token and appends them to a running cache. The next step
          receives that cache as input and only computes attention for the new token against the
          cached history. This makes generation linear in the output length.
        </p>
        <p>
          The cache changes the decoder's input signature between the first step and every later
          step. At step 1 there is no cache yet, so the decoder takes fresh encoder outputs and
          produces a cache. At step 2 and onward the decoder takes an existing cache plus the
          previous token and produces an updated cache. ONNX graphs have fixed input and output
          lists. You cannot declare a graph that sometimes has 2 inputs and sometimes has 74. That
          is why a single PyTorch decoder becomes two ONNX graphs, and the encoder becomes a third.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200">The split</h3>
        <p>
          The export script <code className="text-teal-400 font-mono text-xs md:text-sm">01_export_encoder_decoder.py</code> produces exactly three ONNX
          files. Each has a distinct job and a distinct input and output contract.
        </p>

        <div className="overflow-x-auto border border-white/5 rounded-lg my-3">
          <table className="w-full border-collapse text-left text-sm font-mono">
            <thead>
              <tr className="bg-zinc-900/60 text-zinc-400 border-b border-white/5 font-sans font-semibold">
                <th className="p-3">File</th>
                <th className="p-3">Runs when</th>
                <th className="p-3">Past KV in</th>
                <th className="p-3">Present KV out</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-300">
              <tr>
                <td className="p-3"><code>encoder_model.onnx</code></td>
                <td className="p-3 font-sans">Once per sentence</td>
                <td className="p-3">None</td>
                <td className="p-3">None</td>
              </tr>
              <tr>
                <td className="p-3"><code>decoder_model.onnx</code></td>
                <td className="p-3 font-sans">Step 1 only</td>
                <td className="p-3">None</td>
                <td className="p-3">Full cache</td>
              </tr>
              <tr>
                <td className="p-3"><code>decoder_with_past_model.onnx</code></td>
                <td className="p-3 font-sans">Steps 2 through end</td>
                <td className="p-3">Full cache</td>
                <td className="p-3">Updated cache</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p>
          A common question is why the encoder is split at all, rather than fused into the first
          decoder step. Splitting lets the encoder run once and feed all decoder steps. It also
          matches the I/O layout of the reference naklitechie bundle, so the output works with
          existing loaders without custom glue.
        </p>

        {/* Quiz 1 */}
        <InteractiveQuiz
          question="Why does the decoder become two separate ONNX graphs instead of one?"
          options={[
            'ONNX cannot represent attention layers in a single graph.',
            "The decoder's input list changes between step 1 (no cache) and step 2+ (cache in), and ONNX graphs have a fixed input and output contract.",
            'Two graphs run faster on WebGPU than one graph.'
          ]}
          correctIndex={1}
          correctFeedback="The KV cache is absent on step 1 and present on every later step. Because an ONNX graph declares its inputs once at export time, the two signatures require two graphs."
          incorrectFeedback="The reason is the input contract, not performance or operator support. Think about what inputs the decoder needs on step 1 versus step 5."
          tip="Tip: ONNX graphs declare a fixed list of named inputs at export time. They cannot grow that list at runtime."
        />
      </section>

      {/* 4. The encoder graph */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The encoder graph</h2>
        <p>
          The encoder graph is the simplest of the three. It takes the source token ids and an
          attention mask, and returns the contextual embedding for every source token. It runs once
          per sentence. Its output, <code className="text-teal-400 font-mono text-xs md:text-sm">last_hidden_state</code>, is what the decoder attends to
          through cross attention.
        </p>
        <p>
          The export wraps the encoder in a thin PyTorch module so the exported graph exposes a
          clean two input, one output contract.
        </p>

        <CodeBlock
          label="it2_onnx_wrappers.py"
          code={`class IndicTransEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state`}
        />

        <p>
          The export call declares two dynamic axes. <code className="text-zinc-400 font-mono">batch_size</code> lets the graph accept
          any batch size at runtime. <code className="text-zinc-400 font-mono">encoder_sequence_length</code> lets it accept sentences
          of any length up to the model maximum of 256. Without these declarations the traced graph
          would lock to the dummy shape used during tracing, batch 1 and length 8, and fail on real
          inputs.
        </p>

        <CodeBlock
          label="01_export_encoder_decoder.py"
          code={`torch.onnx.export(
    wrapper,
    (input_ids, attention_mask),
    str(path),
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "encoder_sequence_length"},
        "attention_mask": {0: "batch_size", 1: "encoder_sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "encoder_sequence_length"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)`}
        />

        <p>
          <code className="text-zinc-400 font-mono">do_constant_folding=True</code> tells the exporter to precompute parts of the graph
          that depend only on constants, folding them into stored initializers. <code className="text-zinc-400 font-mono">dynamo=False</code>
          forces the legacy tracer instead of the newer dynamo exporter. The dynamo
          path requires an extra dependency called <code className="text-zinc-400 font-mono">onnxscript</code> and was less reliable for
          this custom architecture, so the pipeline pins the legacy tracer.
        </p>
      </section>

      {/* 5. The decoder (first step) graph */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The decoder (first step) graph</h2>
        <p>
          <code className="text-teal-400 font-mono text-xs md:text-sm">decoder_model.onnx</code> handles the first generation step. It receives the start
          token and the full encoder output, and it produces two things: the logits used to pick the
          first real token, and the complete initial KV cache that every later step will carry.
        </p>
        <p>
          Its inputs are <code className="text-zinc-400 font-mono">input_ids</code> (the single start token), <code className="text-zinc-400 font-mono">encoder_attention_mask</code>,
          and <code className="text-zinc-400 font-mono">encoder_hidden_states</code> (the encoder output tensor). Notice the encoder
          output is a direct input here. On step 1 the cache is empty, so cross attention must read
          the fresh encoder states.
        </p>
        <p>
          Its outputs are <code className="text-zinc-400 font-mono">logits</code> plus one set of four present KV tensors per decoder
          layer. With 18 layers that is 72 cache tensors leaving the graph. The wrapper flattens the
          nested cache into a flat tuple of named tensors so ONNX can declare each one explicitly.
        </p>

        <CodeBlock
          label="it2_onnx_wrappers.py"
          code={`class IndicTransDecoderWrapper(nn.Module):
    """First decode step: no past KV in, full present KV out + logits."""

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        logits = self.lm_head(out.last_hidden_state)
        return (logits, *_flatten_past(out.past_key_values))`}
        />

        <p>
          The output names follow a strict convention. For each layer index <code className="text-zinc-400 font-mono">i</code> the graph
          emits <code className="text-zinc-400 font-mono">present.i.decoder.key</code>, <code className="text-zinc-400 font-mono">present.i.decoder.value</code>,
          <code className="text-zinc-400 font-mono">present.i.encoder.key</code>, and <code className="text-zinc-400 font-mono">present.i.encoder.value</code>. The decoder
          self attention cache and the cross attention cache are emitted together. This naming is
          what the runtime loop uses to feed the next graph.
        </p>
      </section>

      {/* 6. The decoder_with_past graph */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The decoder_with_past graph</h2>
        <p>
          <code className="text-teal-400 font-mono text-xs md:text-sm">decoder_with_past_model.onnx</code> is the workhorse. It runs for every step after
          the first. It receives the previous token, the encoder attention mask, and the full cache
          from the prior step. It returns updated logits and an updated cache with one more entry
          appended to the decoder self attention portion.
        </p>
        <p>
          Two non-obvious tricks live in this wrapper. Both fix subtle tracing bugs that produced
          broken translations in earlier exports.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200">The dummy encoder hidden states trick</h3>
        <p>
          The AI4Bharat modeling code guards cross attention with a check: <em className="italic text-zinc-400 font-serif">if
          encoder_hidden_states is not None, run cross attention.</em> During step 2 and later, the
          real encoder information already lives in the cached cross attention keys and values. So
          you might pass <code className="text-zinc-400 font-mono">None</code> for <code className="text-zinc-400 font-mono">encoder_hidden_states</code> to signal
          "use the cache".
        </p>
        <p>
          That breaks the export. When the tracer sees <code className="text-zinc-400 font-mono">None</code>, it skips recording the
          cross attention block entirely. The exported graph then has no cross attention operators
          at all, so it produces repetitive garbage from step 2 onward. The fix is to pass a
          <strong className="text-zinc-100 font-bold">dummy</strong> zero tensor shaped like the encoder output. The presence of a real
          tensor makes the tracer compile the cross attention block. At runtime the cross attention
          reads from the cached keys and values, and the dummy states are projected away.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200">The zero-cost mask dependency</h3>
        <p>
          The ONNX optimizer noticed that <code className="text-zinc-400 font-mono">encoder_attention_mask</code> appeared unused inside
          the wrapper's forward path, because the cached cross attention already incorporated the
          mask. The optimizer then dropped the mask from the graph's input list. At runtime the
          session rejected calls that supplied the mask, or silently misused it.
        </p>
        <p>
          The fix is a single line that creates a zero-cost dependency on the mask.
        </p>

        <CodeBlock
          label="it2_onnx_wrappers.py"
          code={`# Keep encoder_attention_mask in the ONNX graph (cross-attn mask for cached KV)
logits = logits + encoder_attention_mask.sum() * 0.0`}
        />

        <p>
          Multiplying the mask sum by zero adds nothing to the logits numerically. But the tracer
          now records a data flow edge from <code className="text-zinc-400 font-mono">encoder_attention_mask</code> into the output.
          The optimizer keeps the input. The mask stays in the graph's public contract without
          changing any result.
        </p>

        <InteractiveQuiz
          question="Why is a dummy zero tensor passed as encoder_hidden_states in the decoder_with_past wrapper?"
          options={[
            'The decoder needs fresh encoder context at every step.',
            'Passing None makes the tracer skip the cross attention block, so a dummy tensor forces that block to compile while the real context is read from the cache at runtime.',
            'Zeros are required for numerical stability of attention.'
          ]}
          correctIndex={1}
          correctFeedback="The dummy exists to satisfy the tracer. The cached cross attention keys and values already hold the encoder context for steps 2 and onward."
          incorrectFeedback="The decoder does not need fresh encoder context every step. Reconsider what the tracer does when it encounters a None input."
          tip="Tip: Tracing records only the code paths actually executed with the example inputs. A None input can hide a whole branch from the exported graph."
        />

        <p>
          The dynamic axes for this graph are the most involved. The decoder cache tensors declare a
          <code className="text-zinc-400 font-mono">past_decoder_sequence_length</code> dynamic axis that grows by one each step. The
          present outputs declare a matching <code className="text-zinc-400 font-mono">past_decoder_sequence_length_plus_1</code> axis.
          The encoder cache tensors declare <code className="text-zinc-400 font-mono">encoder_sequence_length</code> dynamic, so the
          graph is not locked to the length 8 used during tracing.
        </p>
      </section>

      {/* 7. The KV cache details */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The KV cache: four tensors per layer</h2>
        <p>
          Every decoder layer contributes four cache tensors. Two belong to self attention and grow
          each step. Two belong to cross attention and stay fixed at the encoder sequence length.
          Understanding this split is the key to reading any seq2seq ONNX export.
        </p>

        <div className="overflow-x-auto border border-white/5 rounded-lg my-3">
          <table className="w-full border-collapse text-left text-sm font-mono">
            <thead>
              <tr className="bg-zinc-900/60 text-zinc-400 border-b border-white/5 font-sans font-semibold">
                <th className="p-3">Tensor</th>
                <th className="p-3">Source</th>
                <th className="p-3">Shape (batch, heads, seq, head_dim)</th>
                <th className="p-3">Grows per step?</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-300">
              <tr>
                <td className="p-3"><code>decoder.key</code></td>
                <td className="p-3 font-sans">Self attention</td>
                <td className="p-3">B, 8, T, 64</td>
                <td className="p-3 font-sans text-teal-400">Yes, T increases by 1</td>
              </tr>
              <tr>
                <td className="p-3"><code>decoder.value</code></td>
                <td className="p-3 font-sans">Self attention</td>
                <td className="p-3">B, 8, T, 64</td>
                <td className="p-3 font-sans text-teal-400">Yes, T increases by 1</td>
              </tr>
              <tr>
                <td className="p-3"><code>encoder.key</code></td>
                <td className="p-3 font-sans">Cross attention</td>
                <td className="p-3">B, 8, S, 64</td>
                <td className="p-3 font-sans text-zinc-500">No, fixed at encoder length S</td>
              </tr>
              <tr>
                <td className="p-3"><code>encoder.value</code></td>
                <td className="p-3 font-sans">Cross attention</td>
                <td className="p-3">B, 8, S, 64</td>
                <td className="p-3 font-sans text-zinc-500">No, fixed at encoder length S</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p>
          With 18 layers, each step carries 72 cache tensors in and emits 72 out. The encoder cross
          attention half of the cache is set once at step 1 and then carried unchanged through every
          later step. Only the decoder self attention half grows. This is why step 1 is expensive,
          it computes the full cross attention cache from the encoder output, and later steps are
          cheap, they only extend the self attention cache by one entry.
        </p>

        {/* SVG Diagram 3 */}
        <div className="bg-zinc-950/40 p-4 rounded-xl border border-white/5 flex flex-col items-center">
          <svg viewBox="0 0 620 260" className="w-full max-w-xl" xmlns="http://www.w3.org/2000/svg">
            {/* Layer box */}
            <rect x="20" y="20" width="580" height="220" rx="10" fill="rgba(255,255,255,0.01)" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <text x="40" y="45" fontFamily="sans-serif" fontSize="12" fontWeight="600" fill="#f4f4f5">One decoder layer's cache (repeated x18)</text>

            {/* Self attention cache */}
            <rect x="40" y="65" width="250" height="150" rx="8" fill="rgba(13, 148, 136, 0.03)" stroke="#0d9488" strokeWidth="1.2" />
            <text x="165" y="90" textAnchor="middle" fontFamily="sans-serif" fontSize="13" fontWeight="600" fill="#f4f4f5">Self attention cache</text>
            <rect x="60" y="110" width="210" height="28" rx="4" fill="rgba(13, 148, 136, 0.1)" stroke="rgba(13, 148, 136, 0.2)" />
            <text x="165" y="129" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#2dd4bf">decoder.key  [B,8,T,64]</text>
            <rect x="60" y="148" width="210" height="28" rx="4" fill="rgba(13, 148, 136, 0.1)" stroke="rgba(13, 148, 136, 0.2)" />
            <text x="165" y="167" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#2dd4bf">decoder.value  [B,8,T,64]</text>
            <text x="165" y="200" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#71717a">T grows +1 each step</text>

            {/* Cross attention cache */}
            <rect x="330" y="65" width="250" height="150" rx="8" fill="rgba(245, 158, 11, 0.03)" stroke="#f59e0b" strokeWidth="1.2" />
            <text x="455" y="90" textAnchor="middle" fontFamily="sans-serif" fontSize="13" fontWeight="600" fill="#f4f4f5">Cross attention cache</text>
            <rect x="350" y="110" width="210" height="28" rx="4" fill="rgba(245, 158, 11, 0.1)" stroke="rgba(245, 158, 11, 0.2)" />
            <text x="455" y="129" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#fbbf24">encoder.key  [B,8,S,64]</text>
            <rect x="350" y="148" width="210" height="28" rx="4" fill="rgba(245, 158, 11, 0.1)" stroke="rgba(245, 158, 11, 0.2)" />
            <text x="455" y="167" textAnchor="middle" fontFamily="monospace" fontSize="11" fill="#fbbf24">encoder.value  [B,8,S,64]</text>
            <text x="455" y="200" textAnchor="middle" fontFamily="sans-serif" fontSize="11" fill="#71717a">S fixed at encoder length</text>
          </svg>
          <span className="text-[10px] text-zinc-500 mt-2">The four cache tensors per layer. The blue self attention half grows with the output. The green cross attention half is fixed by the source sentence length.</span>
        </div>
      </section>

      {/* 8. Decode step simulator */}
      <section className="border border-white/5 rounded-xl p-6 bg-zinc-950/20 space-y-4">
        <div>
          <div className="flex items-center gap-2 text-teal-400 mb-1">
            <Play size={16} />
            <h4 className="text-sm md:text-base font-bold uppercase tracking-wider">Decode step simulator</h4>
          </div>
          <p className="text-xs text-zinc-400 font-sans">Step through the first few generation steps to see how the three graphs cooperate and how the cache grows. The example uses a 5 token source sentence.</p>
        </div>

        {/* Progress indicators */}
        <div className="flex items-center gap-2 justify-center py-1">
          {simSteps.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setSimStep(idx)}
              className={`w-2.5 h-2.5 rounded-full transition-all ${
                simStep === idx
                  ? 'bg-teal-400 ring-4 ring-teal-500/20'
                  : 'bg-zinc-700 hover:bg-zinc-500'
              }`}
              title={`Step ${idx + 1}`}
            />
          ))}
        </div>

        {/* Content Card */}
        <div className="bg-zinc-900/60 border border-white/5 rounded-lg p-5 space-y-4 font-mono text-xs md:text-sm">
          <div className="flex items-center justify-between border-b border-white/5 pb-2">
            <span className="text-zinc-400 font-sans font-bold">Step {simStep + 1}: {activeStep.title}</span>
            <span className="text-[10px] bg-teal-500/10 border border-teal-500/20 text-teal-400 px-2 py-0.5 rounded uppercase tracking-wider font-semibold font-sans">
              {activeStep.graph}
            </span>
          </div>

          <p className="text-zinc-300 font-sans text-xs md:text-sm leading-relaxed">{activeStep.desc}</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-[10px] md:text-xs">
            <div className="space-y-1 bg-zinc-950/40 p-3 rounded border border-white/5">
              <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block font-sans">Inputs (in)</span>
              <ul className="list-disc pl-4 space-y-1 text-zinc-400">
                {activeStep.inputs.map((inp, idx) => (
                  <li key={idx}>{inp}</li>
                ))}
              </ul>
            </div>

            <div className="space-y-1 bg-zinc-950/40 p-3 rounded border border-white/5">
              <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider block font-sans">Outputs (out)</span>
              <ul className="list-disc pl-4 space-y-1 text-zinc-400">
                {activeStep.outputs.map((out, idx) => (
                  <li key={idx}>{out}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="p-3 bg-teal-500/5 border border-teal-500/10 rounded text-xs text-teal-300 font-sans">
            {activeStep.cache}
          </div>
        </div>

        {/* Buttons */}
        <div className="flex justify-between items-center pt-2 font-sans">
          <button
            onClick={() => setSimStep((s) => Math.max(0, s - 1))}
            disabled={simStep === 0}
            className="flex items-center gap-1 text-xs md:text-sm text-zinc-400 hover:text-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed transition font-semibold"
          >
            <ArrowLeft size={14} /> Back
          </button>
          <button
            onClick={() => setSimStep((s) => Math.min(simSteps.length - 1, s + 1))}
            disabled={simStep === simSteps.length - 1}
            className="flex items-center gap-1 text-xs md:text-sm text-teal-400 hover:text-teal-300 disabled:opacity-30 disabled:cursor-not-allowed transition font-semibold"
          >
            Next <ArrowRight size={14} />
          </button>
        </div>
      </section>

      {/* 9. External data sidecars */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">External data sidecars</h2>
        <p>
          An ONNX protobuf has a hard practical size limit around 2 GB because of the 32 bit
          message size cap in the protobuf format. The decoder weights for the 200M model approach
          or exceed that limit. The export pipeline handles this with a sidecar file.
        </p>
        <p>
          The function <code className="text-zinc-400 font-mono">_externalize_if_large</code> checks the serialized proto size. If it
          exceeds 512 MB, it calls <code className="text-zinc-400 font-mono">convert_model_to_external_data</code> to move every weight
          tensor into a separate binary file named <code className="text-zinc-400 font-mono">encoder_model.onnx.data</code> or
          <code className="text-zinc-400 font-mono">decoder_model.onnx.data</code>. The proto then keeps only the graph structure and
          small metadata, while the bulk weights live in the sidecar.
        </p>

        <CodeBlock
          label="01_export_encoder_decoder.py"
          code={`def _externalize_if_large(onnx_path: Path, size_threshold_mb=512) -> None:
    model = onnx.load(str(onnx_path), load_external_data=True)
    proto_mb = onnx_path.stat().st_size / (1024 * 1024)
    if proto_mb < size_threshold_mb:
        return
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=1024,
    )
    save_model(model, str(onnx_path))`}
        />

        <p>
          In the real en to indic bundle this produces a striking split. The encoder proto is 294
          MB, under the threshold, so its weights stay inside <code className="text-zinc-400 font-mono">encoder_model.onnx</code>. Each
          decoder proto is about 1.2 MB of graph structure, while the actual weights live in a 805
          MB and 767 MB sidecar respectively. You must ship the <code className="text-zinc-400 font-mono">.onnx.data</code> file next to
          its <code className="text-zinc-400 font-mono">.onnx</code> file. The runtime loads both automatically as long as they sit in
          the same directory.
        </p>

        <div className="bg-teal-500/5 border border-teal-500/10 p-5 rounded-lg text-xs md:text-sm leading-relaxed text-zinc-400">
          If you upload a bundle to Hugging Face without the <code className="text-zinc-300 font-mono">.onnx.data</code> sidecars,
          the runtime will fail to load the decoders. Treat each <code className="text-zinc-300 font-mono">.onnx</code> plus its
          <code className="text-zinc-300 font-mono">.onnx.data</code> as an inseparable pair.
        </div>
      </section>

      {/* 10. Tokenizers */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">Tokenizers: the other half of the bundle</h2>
        <p>
          The ONNX graphs only know about integer token ids. Turning raw text into ids and ids back
          into text is the job of the tokenizer, which ships as a separate set of files in the same
          bundle. For IndicTrans2 this is where most of the export difficulty lives.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200" id="slow-vs-fast">Slow versus fast</h3>
        <p>
          The original model ships a <strong className="text-zinc-100 font-bold">slow tokenizer</strong> implemented in Python, the file
          <code className="text-zinc-400 font-mono">tokenization_indictrans.py</code>, loaded through <code className="text-zinc-400 font-mono">trust_remote_code=True</code>.
          It wraps two SentencePiece models, <code className="text-zinc-400 font-mono">model.SRC</code> and <code className="text-zinc-400 font-mono">model.TGT</code>, plus
          two Fairseq dictionaries, <code className="text-zinc-400 font-mono">dict.SRC.json</code> and <code className="text-zinc-400 font-mono">dict.TGT.json</code>. The
          slow tokenizer works in Python but cannot run efficiently in a browser.
        </p>
        <p>
          The browser needs a <strong className="text-zinc-100 font-bold">fast tokenizer</strong>, the single file format
          <code className="text-zinc-400 font-mono">tokenizer.json</code> understood by the Rust <code className="text-zinc-400 font-mono">tokenizers</code> library and by
          onnxruntime-web's tokenizer support. The build script
          <code className="text-teal-400 font-mono text-xs md:text-sm">02_build_fast_tokenizers.py</code> converts the SentencePiece models into
          <code className="text-zinc-400 font-mono">tokenizer_src.json</code> and <code className="text-zinc-400 font-mono">tokenizer_tgt.json</code> using the Hugging Face
          <code className="text-zinc-400 font-mono">SpmConverter</code>.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200" id="dict-remap">Why the dict remap is necessary</h3>
        <p>
          The conversion is not a one step copy. <code className="text-zinc-400 font-mono">SpmConverter</code> assigns token ids using
          SentencePiece's native indexing, which does not match the Fairseq dictionary ids the model
          was trained on. If you shipped the raw converted file, the model would receive the wrong
          id for almost every token and produce garbage.
        </p>
        <p>
          The build script remaps every vocabulary entry to the id from the matching dict json. It
          then registers the special tokens and the language tags as <strong className="text-zinc-100 font-bold">added tokens</strong>
          with their correct dict ids. The language tags like <code className="text-zinc-400 font-mono">hin_Deva</code> and
          <code className="text-zinc-400 font-mono">eng_Latn</code> are marked <code className="text-zinc-400 font-mono">single_word: true</code> and <code className="text-zinc-400 font-mono">special: false</code>,
          matching the reference bundle. Finally it installs a <code className="text-zinc-400 font-mono">TemplateProcessing</code>
          post-processor that appends the end of sentence token with id 2.
        </p>

        <CodeBlock
          label="02_build_fast_tokenizers.py"
          code={`remapped = {
    token: vocab_dict.get(token, UNK_ID)
    for token in fast["model"]["vocab"]
}
fast["model"]["vocab"] = remapped
fast["added_tokens"] = added_tokens
fast["post_processor"] = EOS_POST_PROCESSOR`}
        />

        <p>
          A small <code className="text-teal-400 font-mono text-xs md:text-sm">tokenizer_meta.json</code> file records <code className="text-zinc-400 font-mono">src_dict_size</code>,
          <code className="text-zinc-400 font-mono">tgt_dict_size</code>, and <code className="text-zinc-400 font-mono">unk_id: 3</code>. The runtime uses these to clamp
          any id greater than or equal to the dict size down to the unknown token. This guards
          against the fast tokenizer emitting ids for vocabulary entries the model never trained on.
        </p>
        <p>
          Encoding a source sentence also requires a language prefix. The runtime prepends the
          source and target language tags before the text, for example
          <code className="text-zinc-400 font-mono">hin_Deva eng_Latn यह एक परीक्षण वाक्य है।</code>. The added tokens setup ensures the
          tokenizer recognizes those tags and maps them to their dict ids.
        </p>

        <h3 className="text-base md:text-lg font-bold text-zinc-200" id="two-tokenizers">Why two tokenizers</h3>
        <p>
          Source and target use separate vocabularies, separate SentencePiece models, and separate
          dictionaries. The source side encodes input text in the source script. The target side
          decodes generated ids back to the target script. They are not interchangeable. Swapping a
          source tokenizer from one direction into another direction gives zero percent token match,
          because the dictionaries interleave language tags at different ids. Each direction builds
          its own pair from scratch.
        </p>

        {/* Quiz 2 */}
        <InteractiveQuiz
          question="After SpmConverter produces a fast tokenizer, why is an extra remap step required?"
          options={[
            'The fast format is missing the post-processor.',
            'SpmConverter uses SentencePiece-native ids that differ from the Fairseq dict ids the model was trained on, so ids must be remapped to the dict json values.',
            'The conversion drops language tags, so they must be reinserted manually.'
          ]}
          correctIndex={1}
          correctFeedback="Without the remap the model receives ids from the wrong vocabulary indexing and produces garbage output."
          incorrectFeedback="The post-processor and language tags are added separately. The remap exists because of an id numbering mismatch between two vocabulary systems."
          tip="Tip: SentencePiece and Fairseq are two different tokenization systems with their own id assignments. The model was trained against the Fairseq ids."
        />
      </section>

      {/* 11. Config files */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">Config files</h2>
        <p>
          Two json files carry model level configuration into the bundle.
          <code className="text-zinc-400 font-mono">config.json</code> holds the architecture constants, layer counts, head counts,
          embedding dimensions, vocabulary sizes, and token ids. The runtime does not always need it
          to run the graphs, but loaders and validation scripts read it to recover the layer count
          and dimensions.
        </p>
        <p>
          <code className="text-zinc-400 font-mono">generation_config.json</code> holds the decoding parameters. The two values that
          matter most for the greedy loop are <code className="text-zinc-400 font-mono">decoder_start_token_id</code> and
          <code className="text-zinc-400 font-mono">eos_token_id</code>, both 2 for these models. The runtime uses the start id to seed
          the decoder and the eos id to detect when generation is complete. Without
          <code className="text-zinc-400 font-mono">generation_config.json</code> the validation script falls back to defaults, which
          happen to match, but shipping the file makes the bundle self describing.
        </p>
        <p>
          The remaining tokenizer support files, <code className="text-zinc-400 font-mono">tokenizer_config.json</code>,
          <code className="text-zinc-400 font-mono">special_tokens_map.json</code>, and <code className="text-zinc-400 font-mono">tokenization_indictrans.py</code>, are
          copied from the original repo. They let the slow tokenizer still load in Python for
          validation and postprocessing even though the browser uses the fast json tokenizers.
        </p>
      </section>

      {/* 12. Greedy decode loop */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">Putting it together: the greedy decode loop</h2>
        <p>
          The validation script <code className="text-teal-400 font-mono text-xs md:text-sm">03_validate_parity.py</code> contains the cleanest statement
          of how all the components cooperate. It runs the same greedy loop against both the PyTorch
          model and the three ONNX sessions and compares the emitted token ids. The pass bar is 99
          percent token exact match. All three directions hit 100 percent on their fixtures.
        </p>
        <p>
          The loop is short. The encoder session runs once. The decoder session runs for step 1.
          The decoder_with_past session runs for every step after that, carrying the cache forward
          each time. Argmax over the last logits position picks the next token. When that token
          equals the eos id, the loop stops.
        </p>

        <CodeBlock
          label="03_validate_parity.py"
          code={`enc_out = enc.run(["last_hidden_state"], {
    "input_ids": input_ids,
    "attention_mask": attn_mask,
})[0]

decoder_input_ids = np.array([[decoder_start_id]], dtype=np.int64)
output_ids = [decoder_start_id]
past_outputs = None

for step in range(max_new_tokens):
    if step == 0:
        dec_out = dec.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_hidden_states": enc_out,
            "encoder_attention_mask": attn_mask,
        })
    else:
        dec_out = dec_past.run(None, {
            "input_ids": decoder_input_ids,
            "encoder_attention_mask": attn_mask,
            **_past_feed_from_outputs(past_outputs, num_layers),
        })

    logits = dec_out[0]
    past_outputs = list(dec_out[1:])
    next_id = int(np.argmax(logits[0, -1, :]))
    output_ids.append(next_id)
    if next_id == eos_id:
        break
    decoder_input_ids = np.array([[next_id]], dtype=np.int64)`}
        />

        <p>
          The helper <code className="text-zinc-400 font-mono">_past_feed_from_outputs</code> is the bridge between the two decoder
          graphs. It takes the flat list of present outputs from the previous step and renames them
          into the <code className="text-zinc-400 font-mono">past_key_values.*</code> input names the next step expects. The encoder
          cross attention half of that feed never changes after step 1. Only the decoder self
          attention half grows.
        </p>
        <p>
          Note that <code className="text-zinc-400 font-mono">model.generate()</code>, the usual Hugging Face generation entry point,
          does not work on this custom architecture. It raises an attribute error around
          <code className="text-zinc-400 font-mono">use_cache</code>. The manual loop is the workaround, and it doubles as the
          reference implementation that the browser worker later mirrors in TypeScript.
        </p>
      </section>

      {/* 13. Quantization */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">Quantization: a second tier</h2>
        <p>
          The fp32 bundles are large. The en to indic bundle is about 1.7 GB. For low bandwidth or
          mobile deployments, the pipeline offers an optional INT8 tier built by
          <code className="text-teal-400 font-mono text-xs md:text-sm">04_quantize_int8.py</code>. It runs <strong className="text-zinc-100 font-bold">dynamic quantization</strong> on the
          weights of all three graphs, leaving activations in floating point.
        </p>
        <p>
          Dynamic quantization replaces weight matrices with int8 lookups and small scale and zero
          point metadata. The script uses <code className="text-zinc-400 font-mono">QuantType.QInt8</code> and per channel scaling,
          which keeps a separate scale per output channel and usually preserves more quality than
          per tensor scaling. The tokenizer and config files are copied across unchanged from the
          fp32 bundle, because quantization only touches the graph weights.
        </p>

        <CodeBlock
          label="04_quantize_int8.py"
          code={`quantize_dynamic(
    model_input=str(src),
    model_output=str(dst),
    weight_type=QuantType.QInt8,
    per_channel=True,
)`}
        />

        <p>
          The tradeoff is real. The reference int8 benchmark hits about 80 percent exact text match,
          well below the 99 percent bar that fp32 clears. INT8 ships as a separate preview repo with
          a warning that it is not production quality. The roadmap also lists fp16 and q4f16 as
          future tiers with better size to quality tradeoffs.
        </p>

        {/* Precision Tradeoff Calculator */}
        <div className="border border-white/5 rounded-xl p-6 bg-zinc-950/20 space-y-4">
          <div>
            <div className="flex items-center gap-2 text-teal-400 mb-1">
              <Database size={16} />
              <h4 className="text-sm md:text-base font-bold uppercase tracking-wider font-sans">Precision tradeoff calculator</h4>
            </div>
            <p className="text-xs text-zinc-400">Adjust the precision tier to see the estimated bundle size and quality outcome for the en to indic 200M model.</p>
          </div>

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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-sans text-xs md:text-sm">
            <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Estimated Bundle Size</span>
              <span className="text-zinc-100 font-mono font-bold text-lg">{sizeText}</span>
            </div>
            <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Expected Text Parity</span>
              <span className="text-zinc-100 font-mono font-bold text-lg">{activePrecision.parity}</span>
            </div>
            <div className="bg-zinc-900/60 border border-white/5 p-4 rounded-lg space-y-1">
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Runtime Target</span>
              <span className="text-zinc-100 font-sans font-semibold text-xs py-1.5 block">{activePrecision.target}</span>
            </div>
          </div>

          <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden mt-2">
            <div className="bg-teal-500 h-2 transition-all duration-300" style={{ width: `${activePrecision.pct}%` }}></div>
          </div>

          {/* Calculator formula info */}
          <details className="formula-details border border-white/5 rounded-lg p-3 bg-zinc-900/10 cursor-pointer">
            <summary className="formula-summary text-xs text-zinc-400 font-semibold focus:outline-none select-none font-sans">How are these values calculated?</summary>
            <div className="mt-3 text-xs md:text-sm text-zinc-400 leading-relaxed font-sans space-y-2 cursor-auto" onClick={(e) => e.stopPropagation()}>
              <p>
                The size estimate starts from the measured fp32 en to indic bundle size of 1.7 GB and
                applies a reduction ratio per tier. fp16 halves the weight bytes, giving roughly 0.5
                times the fp32 size. INT8 reduces 32 bit weights to 8 bit, giving roughly 0.25 times
                the fp32 size. q4f16 uses 4 bit weights with 16 bit activations, giving roughly 0.125
                times the fp32 size. A fixed overhead of 90 MB is added to every tier to account for
                the two fast tokenizer json files and the config files, which are not quantized.
              </p>
              
              <CodeBlock
                label="size model"
                code={`FP32_BASE_GB = 1.7
OVERHEAD_GB   = 0.09
RATIO = {"fp32": 1.0, "fp16": 0.5, "int8": 0.25, "q4f16": 0.125}
size_gb = max(FP32_BASE_GB * RATIO[tier] * 0.94 + OVERHEAD_GB, OVERHEAD_GB)`}
              />
              
              <p>
                The 0.94 factor accounts for the roughly 6 percent of the fp32 bundle that is already
                non weight data, tokenizer json and configs, which the ratio should not shrink. Parity
                values come from the measured fp32 result of 100 percent and the documented int8
                benchmark of about 80 percent. fp16 and q4f16 parity is projected, not yet measured,
                and marked as such.
              </p>
            </div>
          </details>
        </div>
      </section>

      {/* 14. Manifest Explorer */}
      <section className="space-y-4">
        <h2 className="text-xl md:text-2xl font-bold text-zinc-100 border-b border-white/5 pb-2">The full bundle manifest</h2>
        <p>
          Click any file in the exported en to indic bundle to see what it is, why it is needed, and
          its real size on disk. This is the complete set of components that ship together.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
          {/* File Selector List */}
          <div className="md:col-span-1 border border-white/5 rounded-lg divide-y divide-white/5 overflow-hidden max-h-[400px] overflow-y-auto">
            {manifestFiles.map((file) => (
              <button
                key={file.name}
                type="button"
                onClick={() => setSelectedFile(file)}
                className={`w-full text-left p-3 text-xs md:text-sm transition-all flex flex-col gap-1 ${
                  selectedFile.name === file.name
                    ? 'bg-teal-500/10 text-teal-300'
                    : 'bg-transparent text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: roleColors[file.role] }}></span>
                  <span className="font-mono font-bold truncate">{file.name}</span>
                </div>
                <div className="text-[10px] md:text-xs text-zinc-500 flex justify-between">
                  <span>{file.role}</span>
                  <span>{file.size}</span>
                </div>
              </button>
            ))}
          </div>

          {/* Details Pane */}
          <div className="md:col-span-2 bg-zinc-900/30 border border-white/5 p-5 rounded-lg min-h-[180px] flex flex-col justify-between">
            <div className="space-y-4">
              <div className="flex items-center justify-between border-b border-white/5 pb-2">
                <span className="font-mono font-bold text-zinc-200 text-sm">{selectedFile.name}</span>
                <span
                  className="text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-wider"
                  style={{
                    backgroundColor: `${roleColors[selectedFile.role]}1a`,
                    color: roleColors[selectedFile.role],
                    borderColor: `${roleColors[selectedFile.role]}30`
                  }}
                >
                  {selectedFile.role}
                </span>
              </div>
              <p className="text-zinc-300 text-xs md:text-sm leading-relaxed">{selectedFile.desc}</p>
            </div>
            <div className="text-[10px] text-zinc-500 flex justify-between pt-3 border-t border-white/5 mt-4">
              <span>Size on disk: <strong className="text-zinc-400">{selectedFile.size}</strong></span>
              <span>Role Category: <strong className="text-zinc-400">{selectedFile.role}</strong></span>
            </div>
          </div>
        </div>

        <p className="pt-2">
          Every file above serves one of three roles. The three <code>.onnx</code> graphs plus their
          <code>.data</code> sidecars hold the model weights and operator graph. The tokenizer files
          turn text into ids and back. The config files describe the architecture and decoding
          parameters. Drop the whole directory into a Hugging Face repo and a browser worker can
          load it end to end.
        </p>

        <div className="bg-teal-500/5 border border-teal-500/10 p-5 rounded-lg text-xs md:text-sm leading-relaxed text-zinc-400 font-sans">
          The pipeline validates the bundle before it ships. The parity check runs the same
          greedy decode loop against PyTorch and ONNX, compares token ids, and requires 99
          percent exact match. All three directions pass at 100 percent. The committed parity
          reports live under <code>fixtures/parity-report-*.json</code>.
        </div>
      </section>

    </div>
  )
}
