import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2'
import { transliterate } from './transliterate.js'

env.allowLocalModels = false
env.allowRemoteModels = true
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/'

export const DIRECTIONS = {
  'en-indic': { label: 'English → Indic' },
  'indic-en': { label: 'Indic → English' },
  'indic-indic': { label: 'Indic → Indic' },
}

export const REQUIRED_FILES = [
  'encoder_model.onnx',
  'decoder_model.onnx',
  'decoder_with_past_model.onnx',
  'tokenizer_src.json',
  'tokenizer_tgt.json',
  'tokenizer_meta.json',
  'generation_config.json',
]

const ONNX_GRAPH_FILES = [
  'encoder_model.onnx',
  'decoder_model.onnx',
  'decoder_with_past_model.onnx',
]

let currentSessions = null
let srcTokenizer = null
let tgtTokenizer = null
let tokenizerMeta = null
let generationConfig = null

export function isModelLoaded() {
  return currentSessions !== null
}

export function buildFileMap(fileList) {
  const map = new Map()
  for (const file of fileList) {
    const name = file.webkitRelativePath
      ? file.webkitRelativePath.split('/').pop()
      : file.name
    map.set(name, file)
  }
  return map
}

export function validateBundleFiles(fileMap) {
  const missing = REQUIRED_FILES.filter((name) => !fileMap.has(name))
  const dataFiles = [...fileMap.keys()].filter((name) => name.endsWith('.data'))
  return { missing, dataFiles, ok: missing.length === 0 }
}

async function loadTokenizerFromJson(data, modelName) {
  const originalFetch = window.fetch

  window.fetch = async (fetchUrl, options) => {
    const urlStr = fetchUrl.toString()
    if (urlStr.endsWith('tokenizer.json')) {
      return new Response(JSON.stringify(data), {
        headers: { 'Content-Type': 'application/json' },
      })
    }
    if (urlStr.endsWith('tokenizer_config.json')) {
      return new Response(JSON.stringify({ tokenizer_class: 'PreTrainedTokenizerFast' }), {
        headers: { 'Content-Type': 'application/json' },
      })
    }
    return originalFetch(fetchUrl, options)
  }

  const tok = await AutoTokenizer.from_pretrained(modelName)
  window.fetch = originalFetch
  return tok
}

async function readJsonFile(file) {
  return JSON.parse(await file.text())
}

function getPastFeed(prevOutputs, numLayers) {
  const feed = {}
  for (let i = 0; i < numLayers; i++) {
    feed[`past_key_values.${i}.decoder.key`] = prevOutputs[`present.${i}.decoder.key`]
    feed[`past_key_values.${i}.decoder.value`] = prevOutputs[`present.${i}.decoder.value`]
    feed[`past_key_values.${i}.encoder.key`] = prevOutputs[`present.${i}.encoder.key`]
    feed[`past_key_values.${i}.encoder.value`] = prevOutputs[`present.${i}.encoder.value`]
  }
  return feed
}

async function buildExternalData(fileMap) {
  const dataFiles = [...fileMap.entries()].filter(([name]) => name.endsWith('.data'))
  if (dataFiles.length === 0) {
    return []
  }

  const externalData = []
  for (const [name, file] of dataFiles) {
    const bytes = new Uint8Array(await file.arrayBuffer())
    externalData.push({ path: name, data: bytes })
    if (!name.startsWith('./')) {
      externalData.push({ path: `./${name}`, data: bytes })
    }
  }
  return externalData
}

async function createSessionFromBuffer(modelBuffer, ortOptions, externalData) {
  const options = { ...ortOptions }
  if (externalData.length > 0) {
    options.externalData = externalData
    ort.env.wasm.numThreads = 1
  }
  return ort.InferenceSession.create(modelBuffer, options)
}

async function createSessionFromUrl(modelUrl, ortOptions, externalData) {
  const options = { ...ortOptions }
  if (externalData && externalData.length > 0) {
    options.externalData = externalData
  }
  return ort.InferenceSession.create(modelUrl, options)
}

async function probeExternalDataUrls(baseUrl) {
  const root = baseUrl.replace(/\/$/, '')
  const candidates = [
    'encoder_model.onnx.data',
    'decoder_shared.onnx.data',
    'decoder_model.onnx.data',
    'decoder_with_past_model.onnx.data',
  ]
  const found = []
  for (const name of candidates) {
    try {
      const res = await fetch(`${root}/${name}`, { method: 'HEAD' })
      if (res.ok) {
        found.push(name)
      }
    } catch {
      // ignore unreachable sidecars
    }
  }
  return found
}

async function loadOnnxSessions({ fileMap, baseUrl, provider, onProgress }) {
  const ortOptions = {
    executionProviders: [provider, 'wasm'],
  }

  let externalData = []
  if (fileMap) {
    externalData = await buildExternalData(fileMap)
    if (externalData.length > 0) {
      ort.env.wasm.numThreads = 1
    }
  } else {
    const sidecars = await probeExternalDataUrls(baseUrl)
    if (sidecars.length > 0) {
      ort.env.wasm.numThreads = 1
      const root = baseUrl.replace(/\/$/, '')
      for (const name of sidecars) {
        onProgress('sidecar-fetch', `Downloading sidecar ${name}`, 50)
        const res = await fetch(`${root}/${name}`)
        if (!res.ok) {
          throw new Error(`Failed to fetch external data file: ${name}`)
        }
        const bytes = new Uint8Array(await res.arrayBuffer())
        externalData.push({ path: name, data: bytes })
        if (!name.startsWith('./')) {
          externalData.push({ path: `./${name}`, data: bytes })
        }
      }
    }
  }

  const sessions = {}

  for (const graphName of ONNX_GRAPH_FILES) {
    const progressId = graphName.replace('.onnx', '').replace(/_/g, '-')
    const label = graphName.replace(/_/g, ' ')
    onProgress(progressId, label, 5)

    if (fileMap) {
      const file = fileMap.get(graphName)
      if (!file) {
        throw new Error(`Missing ${graphName}`)
      }
      const buffer = await file.arrayBuffer()
      onProgress(progressId, label, 40)
      sessions[graphName] = await createSessionFromBuffer(buffer, ortOptions, externalData)
    } else {
      const url = `${baseUrl.replace(/\/$/, '')}/${graphName}`
      onProgress(progressId, label, 40)
      sessions[graphName] = await createSessionFromUrl(url, ortOptions, externalData)
    }

    onProgress(progressId, label, 100)
  }

  const decSession = sessions['decoder_model.onnx']
  const numLayers = (decSession.outputNames.length - 1) / 4

  // Inspect which execution provider ORT actually selected. With
  // executionProviders: [provider, 'wasm'], ORT silently falls back to wasm
  // if the requested EP fails to init — without this check a wasm fallback
  // would be mislabeled as 'webgpu' by the benchmark.
  let activeEps = []
  try {
    const eps = await sessions['encoder_model.onnx'].fetchexecutionProviders?.()
    if (Array.isArray(eps)) {
      activeEps = eps
    }
  } catch {
    // Older ORT builds don't expose fetchexecutionProviders; leave empty so
    // the benchmark can't falsely reject a load it cannot introspect.
  }

  return {
    enc: sessions['encoder_model.onnx'],
    dec: decSession,
    decPast: sessions['decoder_with_past_model.onnx'],
    numLayers,
    activeEps,
  }
}

async function loadTokenizersAndConfig({ fileMap, baseUrl, onProgress }) {
  onProgress('meta', 'Tokenizer meta & generation config', 10)

  if (fileMap) {
    tokenizerMeta = await readJsonFile(fileMap.get('tokenizer_meta.json'))
    onProgress('meta', 'Tokenizer meta & generation config', 50)
    generationConfig = await readJsonFile(fileMap.get('generation_config.json'))
  } else {
    const metaRes = await fetch(`${baseUrl}/tokenizer_meta.json`)
    tokenizerMeta = await metaRes.json()
    onProgress('meta', 'Tokenizer meta & generation config', 50)
    const genRes = await fetch(`${baseUrl}/generation_config.json`)
    generationConfig = await genRes.json()
  }

  onProgress('meta', 'Tokenizer meta & generation config', 100)

  onProgress('tok-src', 'Source tokenizer', 10)
  if (fileMap) {
    const srcData = await readJsonFile(fileMap.get('tokenizer_src.json'))
    srcTokenizer = await loadTokenizerFromJson(srcData, 'local-bundle-src')
  } else {
    const srcRes = await fetch(`${baseUrl}/tokenizer_src.json`)
    const srcData = await srcRes.json()
    srcTokenizer = await loadTokenizerFromJson(srcData, 'url-bundle-src')
  }
  onProgress('tok-src', 'Source tokenizer', 100)

  onProgress('tok-tgt', 'Target tokenizer', 10)
  if (fileMap) {
    const tgtData = await readJsonFile(fileMap.get('tokenizer_tgt.json'))
    tgtTokenizer = await loadTokenizerFromJson(tgtData, 'local-bundle-tgt')
  } else {
    const tgtRes = await fetch(`${baseUrl}/tokenizer_tgt.json`)
    const tgtData = await tgtRes.json()
    tgtTokenizer = await loadTokenizerFromJson(tgtData, 'url-bundle-tgt')
  }
  onProgress('tok-tgt', 'Target tokenizer', 100)
}

export async function loadModelFromFiles(fileMap, provider, onProgress) {
  const validation = validateBundleFiles(fileMap)
  if (!validation.ok) {
    throw new Error(`Missing required files: ${validation.missing.join(', ')}`)
  }

  await loadTokenizersAndConfig({ fileMap, onProgress })
  currentSessions = await loadOnnxSessions({ fileMap, provider, onProgress })
  return currentSessions.activeEps
}

export async function loadModelFromUrl(baseUrl, provider, onProgress) {
  await loadTokenizersAndConfig({ baseUrl, onProgress })
  currentSessions = await loadOnnxSessions({ baseUrl, provider, onProgress })
  return currentSessions.activeEps
}

export function unloadModel() {
  currentSessions = null
  srcTokenizer = null
  tgtTokenizer = null
  tokenizerMeta = null
  generationConfig = null
  ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1)
}

export async function translate(text, srcLang, tgtLang, onStep) {
  if (!currentSessions) {
    throw new Error('No model loaded')
  }

  const startTime = performance.now()

  let processedText = text
  if (srcLang !== 'eng_Latn') {
    processedText = transliterate(text, srcLang, 'hin_Deva')
  }

  const srcLangRes = await srcTokenizer(srcLang, { add_special_tokens: false })
  const srcLangId = Number(srcLangRes.input_ids.data[0])

  const tgtLangRes = await srcTokenizer(tgtLang, { add_special_tokens: false })
  const tgtLangId = Number(tgtLangRes.input_ids.data[0])

  const preparedText = processedText.startsWith(' ') ? processedText : ` ${processedText}`
  const textRes = await srcTokenizer(preparedText)

  const textIds = Array.from(textRes.input_ids.data).map(Number)
  const safeInputIds = [srcLangId, tgtLangId, ...textIds].map((id) =>
    id < tokenizerMeta.src_dict_size ? id : Number(tokenizerMeta.unk_id),
  )

  const inputIdsTensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(safeInputIds.map(BigInt)),
    [1, safeInputIds.length],
  )

  const textMaskArray = Array.from(textRes.attention_mask.data).map(Number)
  const attnMaskArray = [1, 1, ...textMaskArray]
  const attnMaskTensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(attnMaskArray.map(BigInt)),
    [1, attnMaskArray.length],
  )

  const encOut = await currentSessions.enc.run({
    input_ids: inputIdsTensor,
    attention_mask: attnMaskTensor,
  })
  const encHiddenState = encOut.last_hidden_state

  const decoderStartId = BigInt(generationConfig.decoder_start_token_id || 2)
  const eosId = BigInt(generationConfig.eos_token_id || 2)

  let decoderInputIds = new ort.Tensor('int64', BigInt64Array.from([decoderStartId]), [1, 1])
  const outputIds = [Number(decoderStartId)]
  let pastOutputs = null

  const maxNewTokens = 128
  let totalTokens = 0
  let ttftTime = null

  for (let step = 0; step < maxNewTokens; step++) {
    let decOut
    if (step === 0) {
      decOut = await currentSessions.dec.run({
        input_ids: decoderInputIds,
        encoder_hidden_states: encHiddenState,
        encoder_attention_mask: attnMaskTensor,
      })
    } else {
      decOut = await currentSessions.decPast.run({
        input_ids: decoderInputIds,
        encoder_attention_mask: attnMaskTensor,
        ...getPastFeed(pastOutputs, currentSessions.numLayers),
      })
    }

    if (step === 0) {
      ttftTime = performance.now() - startTime
    }

    const logits = decOut.logits
    pastOutputs = decOut

    const dims = logits.dims
    const seqLen = dims[1]
    const vocabSize = dims[2]
    const offset = (seqLen - 1) * vocabSize
    const logitsData = logits.data

    let maxVal = -Infinity
    let nextId = 0
    for (let v = 0; v < vocabSize; v++) {
      const val = logitsData[offset + v]
      if (val > maxVal) {
        maxVal = val
        nextId = v
      }
    }

    outputIds.push(nextId)
    totalTokens++

    if (onStep) {
      onStep(step, totalTokens, ttftTime)
    }

    if (BigInt(nextId) === eosId) {
      break
    }

    decoderInputIds = new ort.Tensor('int64', BigInt64Array.from([BigInt(nextId)]), [1, 1])
  }

  const tgtDictSize = tokenizerMeta.tgt_dict_size
  const safeOutputIds = outputIds.map((id) =>
    id < tgtDictSize ? id : Number(tokenizerMeta.unk_id),
  )
  const decodedText = await tgtTokenizer.decode(safeOutputIds, { skip_special_tokens: true })

  let finalOutput = decodedText
  if (tgtLang !== 'eng_Latn') {
    finalOutput = transliterate(decodedText, 'hin_Deva', tgtLang)
  }

  const totalTimeMs = performance.now() - startTime

  return {
    translation: finalOutput,
    totalTokens,
    ttftTime,
    totalTimeMs,
  }
}
