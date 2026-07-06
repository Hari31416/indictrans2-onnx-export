import { LANGUAGES } from './transliterate.js'
import {
  REQUIRED_FILES,
  buildFileMap,
  isModelLoaded,
  loadModelFromFiles,
  loadModelFromUrl,
  translate,
  unloadModel,
  validateBundleFiles,
} from './translator.js'

const selectDirection = document.getElementById('model-direction')
const selectProvider = document.getElementById('ort-provider')
const inputFolder = document.getElementById('bundle-folder')
const inputUrl = document.getElementById('bundle-url')
const btnLoadFolder = document.getElementById('btn-load-folder')
const btnLoadUrl = document.getElementById('btn-load-url')
const btnUnload = document.getElementById('btn-unload')

const fileInventory = document.getElementById('file-inventory')
const folderSummary = document.getElementById('folder-summary')

const loadingCard = document.getElementById('loading-card')
const loadingHeader = document.getElementById('loading-header')
const loadingBody = document.getElementById('loading-body')
const loadingNotes = document.getElementById('loading-notes')
const loadingSummaryBadge = document.getElementById('loading-summary-badge')
const chevronIcon = document.getElementById('chevron-icon')
const progressContainer = document.getElementById('progress-container')

const badgeStatus = document.getElementById('badge-status')
const selectSrcLang = document.getElementById('src-lang')
const selectTgtLang = document.getElementById('tgt-lang')
const textareaSrc = document.getElementById('src-text')
const textareaTgt = document.getElementById('tgt-text')
const btnTranslate = document.getElementById('btn-translate')
const translationLoader = document.getElementById('translation-loader')

const metricsContainer = document.getElementById('metrics')
const metricLoad = document.getElementById('metric-load')
const metricSpeed = document.getElementById('metric-speed')
const metricTotal = document.getElementById('metric-total')
const warningBanner = document.getElementById('quantization-warning')

let pendingFileMap = null

loadingHeader.addEventListener('click', () => {
  const isHidden = loadingBody.classList.contains('hidden')
  if (isHidden) {
    loadingBody.classList.remove('hidden')
    chevronIcon.classList.add('rotate-180')
  } else {
    loadingBody.classList.add('hidden')
    chevronIcon.classList.remove('rotate-180')
  }
})

function updateLanguageSelectors() {
  const direction = selectDirection.value
  selectSrcLang.innerHTML = ''
  selectTgtLang.innerHTML = ''

  const indicCodes = Object.keys(LANGUAGES).filter((code) => code !== 'eng_Latn')

  if (direction === 'en-indic') {
    selectSrcLang.innerHTML = '<option value="eng_Latn">English</option>'
    indicCodes.forEach((code) => {
      selectTgtLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
  } else if (direction === 'indic-en') {
    indicCodes.forEach((code) => {
      selectSrcLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
    selectTgtLang.innerHTML = '<option value="eng_Latn">English</option>'
  } else {
    indicCodes.forEach((code) => {
      selectSrcLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
      selectTgtLang.innerHTML += `<option value="${code}">${LANGUAGES[code]}</option>`
    })
  }
}

function renderFileInventory(fileMap) {
  if (!fileMap || fileMap.size === 0) {
    fileInventory.innerHTML = '<p class="text-xs text-zinc-500">No folder selected yet.</p>'
    return
  }

  const validation = validateBundleFiles(fileMap)
  const rows = []

  for (const name of REQUIRED_FILES) {
    const found = fileMap.has(name)
    rows.push(
      `<div class="file-chip ${found ? 'ok' : 'missing'} text-xs">${found ? '✓' : '✗'} ${name}</div>`,
    )
  }

  for (const name of validation.dataFiles) {
    rows.push(`<div class="file-chip optional text-xs">• ${name} (weight sidecar)</div>`)
  }

  fileInventory.innerHTML = rows.join('')
  folderSummary.textContent = validation.ok
    ? `Ready — ${fileMap.size} files (${validation.dataFiles.length} weight sidecar(s))`
    : `Incomplete — missing ${validation.missing.length} required file(s)`
  folderSummary.className = validation.ok ? 'text-xs text-emerald-400' : 'text-xs text-red-400'

  btnLoadFolder.disabled = !validation.ok || isModelLoaded()
}

function checkQuantizationWarning() {
  const provider = selectProvider.value
  const hasInt8 = pendingFileMap && [...pendingFileMap.keys()].some((n) => n.includes('int8'))
  const hasQ4 = pendingFileMap && [...pendingFileMap.keys()].some((n) => n.includes('q4'))
  if ((hasInt8 || hasQ4) && provider === 'webgpu') {
    warningBanner.classList.remove('hidden')
  } else {
    warningBanner.classList.add('hidden')
  }
}

selectDirection.addEventListener('change', updateLanguageSelectors)
selectProvider.addEventListener('change', checkQuantizationWarning)
updateLanguageSelectors()

inputFolder.addEventListener('change', () => {
  pendingFileMap = buildFileMap(inputFolder.files)
  renderFileInventory(pendingFileMap)
  checkQuantizationWarning()
})

function createProgressBar(id, name) {
  const div = document.createElement('div')
  div.id = `progress-item-${id}`
  div.className = 'space-y-1.5'
  div.innerHTML = `
    <div class="flex justify-between text-xs font-semibold text-zinc-300">
      <span>${name}</span>
      <span id="progress-val-${id}">0%</span>
    </div>
    <div class="w-full bg-zinc-950 h-2 rounded-full overflow-hidden border border-zinc-800">
      <div id="progress-bar-${id}" class="bg-teal-500 h-full w-0 transition-all duration-150"></div>
    </div>
  `
  progressContainer.appendChild(div)
}

function updateProgressBar(id, progress) {
  const bar = document.getElementById(`progress-bar-${id}`)
  const text = document.getElementById(`progress-val-${id}`)
  if (bar && text) {
    bar.style.width = `${progress}%`
    text.innerText = `${progress}%`
  }
}

function setLoadingUi(active) {
  btnLoadFolder.disabled = active || !pendingFileMap || validateBundleFiles(pendingFileMap).missing.length > 0
  btnLoadUrl.disabled = active
  inputFolder.disabled = active
  inputUrl.disabled = active
  selectDirection.disabled = active
  selectProvider.disabled = active
}

function setReadyUi(loadTimeSec) {
  badgeStatus.innerText = 'Ready'
  badgeStatus.className =
    'px-2 py-0.5 text-xs font-semibold bg-emerald-950 text-emerald-400 border border-emerald-900 rounded-full'

  selectSrcLang.disabled = false
  selectTgtLang.disabled = false
  textareaSrc.disabled = false
  btnTranslate.disabled = false
  btnUnload.disabled = false

  metricsContainer.classList.remove('hidden')
  metricLoad.innerText = `${loadTimeSec}s`
  metricSpeed.innerText = '-'
  metricTotal.innerText = '-'
}

function resetDisconnectedUi() {
  badgeStatus.innerText = 'Disconnected'
  badgeStatus.className =
    'px-2 py-0.5 text-xs font-semibold bg-red-950 text-red-400 border border-red-900 rounded-full'

  selectSrcLang.disabled = true
  selectTgtLang.disabled = true
  textareaSrc.disabled = true
  btnTranslate.disabled = true
  btnUnload.disabled = true

  metricsContainer.classList.add('hidden')
  textareaSrc.value = ''
  textareaTgt.value = ''
}

async function runLoad(loaderFn, note) {
  progressContainer.innerHTML = ''
  loadingCard.classList.remove('hidden')
  loadingBody.classList.remove('hidden')
  chevronIcon.classList.add('rotate-180')
  loadingSummaryBadge.classList.add('hidden')
  loadingNotes.innerText = note
  setLoadingUi(true)

  const startTime = performance.now()

  try {
    await loaderFn((id, label, percent) => {
      if (!document.getElementById(`progress-item-${id}`)) {
        createProgressBar(id, label)
      }
      updateProgressBar(id, percent)
    })

    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2)
    setReadyUi(loadTime)
    loadingNotes.innerText = 'Model loaded. Run a translation in the playground below.'
    setTimeout(() => {
      loadingBody.classList.add('hidden')
      chevronIcon.classList.remove('rotate-180')
      loadingSummaryBadge.classList.remove('hidden')
    }, 800)
  } catch (err) {
    console.error('Failed to load model:', err)
    loadingNotes.innerText = `Error: ${err.message}`
    resetDisconnectedUi()
  } finally {
    setLoadingUi(false)
    btnLoadFolder.disabled = !pendingFileMap || validateBundleFiles(pendingFileMap).missing.length > 0
  }
}

btnLoadFolder.addEventListener('click', async () => {
  if (!pendingFileMap) return
  await runLoad(
    (onProgress) => loadModelFromFiles(pendingFileMap, selectProvider.value, onProgress),
    'Loading ONNX bundle from local folder into browser memory…',
  )
})

btnLoadUrl.addEventListener('click', async () => {
  const baseUrl = inputUrl.value.trim()
  if (!baseUrl) {
    loadingCard.classList.remove('hidden')
    loadingNotes.innerText = 'Enter a base URL, e.g. http://127.0.0.1:8000/scratch/en-indic-onnx'
    return
  }
  await runLoad(
    (onProgress) => loadModelFromUrl(baseUrl, selectProvider.value, onProgress),
    `Fetching bundle from ${baseUrl} …`,
  )
})

btnUnload.addEventListener('click', () => {
  unloadModel()
  resetDisconnectedUi()
  loadingCard.classList.add('hidden')
  renderFileInventory(pendingFileMap)
})

btnTranslate.addEventListener('click', async () => {
  const text = textareaSrc.value.trim()
  if (!text) return

  translationLoader.classList.remove('hidden')
  btnTranslate.disabled = true

  try {
    const result = await translate(text, selectSrcLang.value, selectTgtLang.value)
    textareaTgt.value = result.translation

    const genMs = result.totalTimeMs - (result.ttftTime || 0)
    const speed = genMs > 0 ? (result.totalTokens / (genMs / 1000)).toFixed(1) : '-'
    metricSpeed.innerText = `${speed} tok/s`
    metricTotal.innerText = `${(result.totalTimeMs / 1000).toFixed(2)}s`
  } catch (err) {
    console.error('Translation failed:', err)
    textareaTgt.value = `Translation failed: ${err.message}`
  } finally {
    translationLoader.classList.add('hidden')
    btnTranslate.disabled = false
  }
})

renderFileInventory(null)

// Show default local server hint from current origin
if (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') {
  inputUrl.placeholder = `${window.location.origin}/../scratch/en-indic-onnx`
}
