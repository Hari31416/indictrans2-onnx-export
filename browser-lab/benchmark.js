import {
  loadModelFromUrl,
  unloadModel,
  translate
} from './translator.js';

// Setup custom logging helper
const logsContainer = document.getElementById('console-logs');
function log(msg, type = 'info') {
  console.log(`[Benchmark] ${msg}`);
  const div = document.createElement('div');
  if (type === 'error') {
    div.className = 'text-red-400 font-semibold';
  } else if (type === 'success') {
    div.className = 'text-emerald-400 font-semibold';
  } else if (type === 'skip') {
    div.className = 'text-zinc-500 italic';
  } else {
    div.className = 'text-zinc-300';
  }
  div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logsContainer.appendChild(div);
  logsContainer.scrollTop = logsContainer.scrollHeight;
}

// 10 Seed sentences (English)
const EN_SENTENCES = [
  "The quick brown fox jumps over the lazy dog.",
  "What is the capital of India?",
  "Can you please help me find the nearest hospital?",
  "I would like to book a flight ticket to Delhi.",
  "The train is delayed by two hours.",
  "Today is Friday, 3rd July, 2026.",
  "My phone number is +91-9876543210.",
  "The price of petrol has increased by five rupees.",
  "The constitution guarantees freedom of speech and expression.",
  "The prime minister addressed the nation on television."
];

// 10 Seed sentences (Hindi)
const HINDI_SENTENCES = [
  "तेज भूरा लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है।",
  "भारत की राजधानी क्या है?",
  "क्या आप कृपया निकटतम hospital खोजने में मेरी मदद कर सकते हैं?",
  "मैं दिल्ली के लिए उड़ान टिकट बुक करना चाहूंगा।",
  "ट्रेन दो घंटे देरी से चलती है।",
  "आज शुक्रवार, 3 जुलाई, 2026 है।",
  "मेरा फ़ोन नंबर + 91-9876543210 है।",
  "पेट्रोल के दाम में पांच रुपये की बढ़ोतरी हुई है।",
  "संविधान बोलने और अभिव्यक्ति की स्वतंत्रता की गारंटी देता है।",
  "प्रधानमंत्री ने टेलीविजन पर राष्ट्र को संबोधित किया।"
];

// Directions, Scales, Precisions, Providers
const DIRECTIONS = ['en-indic', 'indic-en', 'indic-indic'];
const SCALES = ['base', '1b'];
const PRECISIONS = ['fp32', 'fp16', 'int8', 'q4f16'];
const PROVIDERS = ['webgpu', 'wasm'];

// Generate target list
const TARGETS = [];
for (const direction of DIRECTIONS) {
  for (const scale of SCALES) {
    for (const precision of PRECISIONS) {
      for (const provider of PROVIDERS) {
        TARGETS.push({ direction, scale, precision, provider });
      }
    }
  }
}

function getFolderName(direction, scale, precision) {
  let folder = direction;
  if (scale === '1b') {
    folder += '-1b';
  }
  folder += '-onnx';
  if (precision !== 'fp32') {
    folder += '-' + precision;
  }
  return folder;
}

async function fetchExistingResults() {
  try {
    const res = await fetch('/fixtures/live-browser-benchmarks.json');
    if (res.ok) {
      return await res.json();
    }
  } catch (e) {
    // File doesn't exist yet, return empty list
  }
  return [];
}

async function saveResultsToServer(results) {
  try {
    const res = await fetch('/save-benchmarks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(results)
    });
    const respData = await res.json();
    if (respData.status !== 'success') {
      log(`Server failed to save progress: ${respData.message}`, "error");
    }
  } catch (e) {
    log(`Failed to save progress to server: ${e.message}`, "error");
  }
}

async function run() {
  const results = await fetchExistingResults();
  const currentIndex = results.length;

  document.getElementById('progress-badge').innerText = `${currentIndex} / ${TARGETS.length}`;

  if (currentIndex >= TARGETS.length) {
    log("Benchmarking complete! All results saved to server.", "success");
    document.getElementById('current-target-id').innerText = 'Finished';
    document.getElementById('current-status').innerText = 'All benchmarks completed.';
    document.getElementById('completed-panel').classList.remove('hidden');
    document.getElementById('results-json').innerText = JSON.stringify(results, null, 2);

    // Clear localStorage crash flag
    localStorage.removeItem('running_target');
    window.benchmarkStatus = 'done';
    window.benchmarkResults = results;
    return;
  }

  const target = TARGETS[currentIndex];
  const targetId = `${target.direction}-${target.scale}-${target.precision}-${target.provider}`;
  const configId = `${target.direction}-${target.scale}-${target.precision}`;

  document.getElementById('current-target-id').innerText = targetId;

  // Check if we crashed on this target previously
  if (localStorage.getItem('running_target') === targetId) {
    log(`Warning: Target ${targetId} crashed the tab in the previous run. Skipping to avoid infinite crash loops.`, "error");
    results.push({
      id: targetId,
      configId: configId,
      direction: target.direction,
      scale: target.scale,
      precision: target.precision,
      provider: target.provider,
      loadTimeMs: null,
      avgTtftMs: null,
      avgStepLatencyMs: null,
      tokensPerSec: null,
      sentencesTested: 0,
      totalSentences: 10,
      status: "skipped",
      error: "Tab crashed during execution"
    });

    localStorage.removeItem('running_target');
    await saveResultsToServer(results);

    log(`Proceeding to next model...`);
    setTimeout(() => {
      window.location.reload();
    }, 1000);
    return;
  }

  // Mark that we are running this target
  localStorage.setItem('running_target', targetId);

  // Clean state before loading
  unloadModel();

  const folderName = getFolderName(target.direction, target.scale, target.precision);
  log(`Loading model from folder ${folderName} on ${target.provider}...`);
  document.getElementById('current-status').innerText = `Loading model ${folderName}...`;

  const loadStart = performance.now();
  try {
    await loadModelFromUrl(`../scratch/${folderName}`, target.provider, (id, label, percent) => {
      if (percent === 100) {
        log(`Loaded graph segment: ${label}`);
      }
    });

    const loadTimeMs = Math.round(performance.now() - loadStart);
    log(`Model loaded successfully in ${loadTimeMs}ms.`, "success");

    // Select sentences
    const sentences = target.direction === 'en-indic' ? EN_SENTENCES : HINDI_SENTENCES;
    const srcLang = target.direction === 'en-indic' ? 'eng_Latn' : 'hin_Deva';
    const tgtLang = target.direction === 'en-indic' ? 'hin_Deva' : (target.direction === 'indic-en' ? 'eng_Latn' : 'guj_Gujr');

    log(`Running translations for 10 sentences...`);
    const sentenceResults = [];

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i];
      document.getElementById('current-status').innerText = `Translating sentence ${i + 1}/10...`;

      const res = await translate(sentence, srcLang, tgtLang);

      const ttft = res.ttftTime;
      const genMs = res.totalTimeMs - (res.ttftTime || 0);
      const stepLatency = res.totalTokens > 1 ? (genMs / (res.totalTokens - 1)) : 0;
      const tokensPerSec = genMs > 0 ? (res.totalTokens / (genMs / 1000)) : 0;

      log(`Sentence ${i + 1} - TTFT: ${Math.round(ttft)}ms, Step Latency: ${Math.round(stepLatency)}ms, Speed: ${tokensPerSec.toFixed(1)} t/s`);

      sentenceResults.push({ ttft, stepLatency, tokensPerSec });
    }

    // Compute averages
    const avgTtft = sentenceResults.reduce((sum, r) => sum + r.ttft, 0) / sentenceResults.length;
    const avgStep = sentenceResults.reduce((sum, r) => sum + r.stepLatency, 0) / sentenceResults.length;
    const avgTps = sentenceResults.reduce((sum, r) => sum + r.tokensPerSec, 0) / sentenceResults.length;

    log(`Averages for ${targetId} - Load: ${loadTimeMs}ms, TTFT: ${Math.round(avgTtft)}ms, Step: ${Math.round(avgStep)}ms, Speed: ${avgTps.toFixed(1)} t/s`, "success");

    results.push({
      id: targetId,
      configId: configId,
      direction: target.direction,
      scale: target.scale,
      precision: target.precision,
      provider: target.provider,
      loadTimeMs: loadTimeMs,
      avgTtftMs: Math.round(avgTtft),
      avgStepLatencyMs: Math.round(avgStep),
      tokensPerSec: parseFloat(avgTps.toFixed(1)),
      sentencesTested: sentences.length,
      totalSentences: sentences.length,
      status: "completed"
    });
  } catch (err) {
    log(`Failed executing ${targetId}: ${err.message}`, "error");
    results.push({
      id: targetId,
      configId: configId,
      direction: target.direction,
      scale: target.scale,
      precision: target.precision,
      provider: target.provider,
      loadTimeMs: null,
      avgTtftMs: null,
      avgStepLatencyMs: null,
      tokensPerSec: null,
      sentencesTested: 0,
      totalSentences: 10,
      status: "skipped"
    });
  }

  // Clear running target flag since it completed or failed gracefully
  localStorage.removeItem('running_target');

  // Save progress directly to server
  await saveResultsToServer(results);

  log(`Reloading page in 1 second to start next model...`);
  setTimeout(() => {
    window.location.reload();
  }, 1000);
}

// Start execution
run();
