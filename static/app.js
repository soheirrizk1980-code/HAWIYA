/**
 * Iris Lab interface: sample selection, real backend jobs, stage visualizations,
 * interactive geometry experiments, and scientific readouts.
 * Real biometric inference runs in Python; browser experiments are educational.
 */
'use strict';
const $ = id => document.getElementById(id);
const STAGES = [
  { k: 'THE PHOTOGRAPH', title: 'A photograph is made of pixels.', plain: 'A camera does not send an eye to a computer. It sends a grid of tiny color values. The first challenge is getting an image that is sharp enough to show the iris texture.', deep: 'The photo is decoded into an RGB image. The backend checks orientation, dimensions, brightness, contrast and focus statistics before YOLO begins. Each pixel has red, green and blue channel values.' },
  { k: 'THE DETECTOR', title: 'Detect the iris.', plain: 'YOLO looks for two regions: the colored iris and the dark pupil. It marks the pixels that belong to each region. It is finding a shape—not learning a person’s name.', deep: 'YOLO Iris V2 is a trained instance-segmentation model. It predicts iris (class 0) and pupil (class 1) masks. Our preprocessing chooses a valid pair; detection confidence describes region detection, not identity.' },
  { k: 'THE GEOMETRY', title: 'Measure the boundaries.', plain: 'The computer uses the detected shapes to calculate two circles. One follows the pupil edge and the other follows the outer edge of the iris. Now it knows exactly which ring to read.', deep: 'Robust geometric circle fitting estimates the center coordinates (cx,cy) and radius r for each boundary using the predicted masks. These circles define the sampling region for normalization.' },
  { k: 'THE TRANSFORMATION', title: 'Normalize the iris.', plain: 'Imagine opening a ring and laying it flat. HAWIYA samples the iris all the way around, from the pupil to the outer boundary, and arranges the pixels in a rectangle.', deep: 'Rubber-sheet normalization maps angle θ and relative radius ρ into the source image: P(ρ,θ)=(1−ρ)·PupilEdge(θ)+ρ·IrisEdge(θ). The resulting array has 64 rows and 512 columns.' },
  { k: 'THE TWO VIEWS', title: 'Prepare the texture.', plain: 'We prepare two views of the same flat iris: one in grayscale, and another with brightness adjusted. They reveal patterns using different treatments of light and dark.', deep: 'The original preprocessing produces grayscale replicated across three channels and a gamma-adjusted view (γ≈0.75). Both derive from the same normalized source pixels.' },
  { k: 'THE FEATURE VECTOR', title: 'Extract 512 features.', plain: 'ArcIris studies the flat texture and creates a list of 512 numbers. These numbers are a learned description of visual patterns—not a name, a photograph, or a personal ID.', deep: 'ArcIris uses a trained ResNet100 backbone to create L2-normalized 512-dimensional embeddings. We calculate two embeddings, for the grayscale and gamma inputs. The chart groups adjacent feature values for display.' },
  { k: 'THE COMPARISON', title: 'Iris comparison results.', plain: 'ArcIris comparison of eye A and eye B.', deep: 'Grayscale and gamma cosine scores. Match categories use experimental development-test cutoffs, not a verified identity threshold.' },
];
const PHASE_STAGE = { quality: 0, segmentation: 1, circles: 2, unwrap: 3, encoded: 5, done: 5 };
let eyeA = null, eyeB = null, analysis = null, comparison = null, stage = 0, deep = false, analysisPhase = '', inspectionTouched = false;
let canvasImage = null, sourceGeom = null, manualCenter = { x: 280, y: 210 };
let scanBusy = false, compareBusy = false;
let preparedCache = null, booting = true;
let operationSerial = 0;
let pendingLiveAction = null;
// If the user changes eyes during a scan, run the newest requested calculation
// after the current model job finishes. Never display a result for an old eye.
// Queue a fresh request when the user changes an image mid-analysis.
function scheduleLiveAction(kind) {
  pendingLiveAction = kind;
  setTimeout(drainLiveAction, 0);
}
function drainLiveAction() {
  if (scanBusy || compareBusy || !pendingLiveAction) return;
  const kind = pendingLiveAction;
  pendingLiveAction = null;
  if (kind === 'analyze' && eyeA) analyzeEye();
  else if (kind === 'compare' && eyeA && eyeB) compareEyes();
}


function e(tag, cls, text) { const el = document.createElement(tag); if (cls) el.className = cls; if (text !== undefined) el.textContent = String(text); return el }
function img(src, alt, cls) { const el = document.createElement('img'); el.src = src; el.alt = alt || ''; if (cls) el.className = cls; return el }
function text(id, value) { $(id).textContent = String(value) }
function say(message, isError = false) { const el = $('jobMessage'); el.textContent = message; el.classList.toggle('error', Boolean(isError)) }
function previewUrl(eye) { return eye?.url || '' }
function nameFor(id) { return { original: 'HAWIYA blue iris', green: 'Green iris', brown: 'Brown iris', amber: 'Amber iris', hazel: 'Hazel iris' }[id] || id }

function applyPreparedSelection() {
  // Render only real, previously computed outputs. No simulated progress indicators.
  if (!preparedCache || !eyeA?.sample) return false;
  const found = preparedCache.analyses?.[eyeA.sample];
  if (!found) return false;
  analysis = found;
  analysisPhase = 'done';
  // The reference is precomputed for first load only. Comparisons run live.
  comparison = null;
  stage = stage === 6 && comparison ? 6 : 1;
  inspectionTouched = false;
  renderInspector(); renderSteps();
  setBrowserImage(found.crop, true, found.geometry);
  say('Blue reference loaded. Choose another eye to analyze it, or select eye B for a live comparison.');
  return true;
}

// Manage reference/gallery uploads and the selected eye photos.
async function chooseSample(id, which) {
  const chosen = document.querySelector(`button[data-which="${which}"][data-sample="${id}"]`);
  if (chosen) { chosen.disabled = true; chosen.textContent = '…' }
  try {
    const r = await fetch('/api/samples/' + encodeURIComponent(id)); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'The photo could not be loaded.');
    const blob = await r.blob(); if (!blob.type.startsWith('image/')) throw new Error('This source did not return an image.');
    const file = new File([blob], `${id}.jpg`, { type: 'image/jpeg' });
    assignEye(which, { file, url: URL.createObjectURL(blob), name: nameFor(id), sample: id });
    if (!booting) {
      // A different eye A gets its own fresh model analysis.
      // Any explicit eye B selection triggers a new model comparison, even blue.
      if (which === 'a' && !preparedCache?.analyses?.[id]) scheduleLiveAction('analyze');
      else if (which === 'b' && eyeA) scheduleLiveAction('compare');
    }
  } catch (err) { say(err.message, true) } finally { if (chosen) { chosen.disabled = false; chosen.textContent = which.toUpperCase() } }
}
function assignEye(which, next) {
  operationSerial++;
  const old = which === 'a' ? eyeA : eyeB;
  if (old?.url?.startsWith('blob:')) URL.revokeObjectURL(old.url);
  if (which === 'a') {
    eyeA = next; analysis = null; comparison = null; analysisPhase = ''; inspectionTouched = false; stage = 0;
    sourceGeom = null; setBrowserImage(next.url); renderInspector();
  } else { eyeB = next; comparison = null; if (stage === 6) renderInspector() }
  document.querySelectorAll(`button[data-which="${which}"]`).forEach(b => b.classList.toggle('picked', b.dataset.sample === next.sample));
  document.querySelectorAll('.specimen').forEach(el => { el.classList.toggle('selected-a', el.dataset.sample === eyeA?.sample); el.classList.toggle('selected-b', el.dataset.sample === eyeB?.sample) });
  const box = which === 'a' ? 'selectedAImg' : 'selectedBImg', label = which === 'a' ? 'selectedAName' : 'selectedBName';
  $(box).src = next.url; text(label, next.name); refreshButtons(); renderSteps();
  if (!booting && applyPreparedSelection()) return;
  say(which === 'a' ? `${next.name} selected as eye A. Select Analyze if new results are needed.` : `${next.name} selected as eye B. Select Compare to calculate new results.`);
}
function chooseFile(file, which) {  
if (!file) return;
  if (!file.type.startsWith('image/')) { say('Please select a JPEG or PNG eye photograph.', true); return }
  if (file.size > 9 * 1024 * 1024) { say('Please choose an image smaller than 9 MB.', true); return }
  assignEye(which, { file, url: URL.createObjectURL(file), name: file.name, sample: null });
  if (!booting) { if (which === 'a') scheduleLiveAction('analyze'); else if (eyeA) scheduleLiveAction('compare') }
}
function refreshButtons() { $('analyzeBtn').disabled = !eyeA || scanBusy || compareBusy; $('compareBtn').disabled = !(eyeA && eyeB) || scanBusy || compareBusy }
// Start and poll backend processing jobs.
async function request(endpoint, form) { const res = await fetch(endpoint, { method: 'POST', body: form }); let data; try { data = await res.json() } catch { throw new Error('The server did not return a valid response. Check the terminal.') } if (!res.ok) throw new Error(data.detail || data.error || 'Could not start analysis.'); return data }
function pollSleep(ms) { return new Promise(done => setTimeout(done, ms)) }

async function pollJob(jobid, kind) {
  for (let attempt = 0; attempt < 540; attempt++) {
    const r = await fetch('/api/jobs/' + encodeURIComponent(jobid), { cache: 'no-store' });
    if (!r.ok) throw new Error('The analysis session could not be found.');
    const data = await r.json();
    if (kind === 'analyze') {
      if (data.outputs) { analysis = { ...(analysis || {}), ...data.outputs }; }
      if (data.phase !== analysisPhase) {        
analysisPhase = data.phase; const messages = { quality: 'Image checked. YOLO is locating the pupil and iris.', segmentation: 'YOLO has identified the regions. Measuring their boundaries…', circles: 'Fitted circles are ready. Unwrapping the iris texture…', unwrap: 'The normalized iris strip is ready. ArcIris is extracting its features…', encoded: 'ArcIris has produced the feature vector.', queued: 'Reading your image…' };
        if (messages[data.phase]) say(messages[data.phase]);
        if (PHASE_STAGE[data.phase] !== undefined && !inspectionTouched) stage = PHASE_STAGE[data.phase];
      }
      renderSteps(); renderInspector();
    }
    if (kind === 'compare' && data.phase.startsWith('eye_')) {
      const human = data.phase.startsWith('eye_a_') ? 'Eye A' : 'Eye B';
      const phaseName = data.phase.slice(6);
      const labels = { quality: 'photo quality checked', segmentation: 'YOLO regions found', circles: 'iris and pupil circles fitted', unwrap: 'iris texture unwrapped', encoded: '512 ArcIris features extracted' };
      say(`${human}: ${labels[phaseName] || 'processing'}…`);
    }
    if (data.status === 'done') return data.result;
    if (data.status === 'error') throw new Error(data.error || 'The analysis could not complete.');
    await pollSleep(750);
  }
  throw new Error('The analysis session has expired. Check the server window.');
}
async function analyzeEye() {  
if (!eyeA || scanBusy || compareBusy) return; const serial = ++operationSerial; scanBusy = true; refreshButtons(); analysis = null; comparison = null; analysisPhase = 'queued'; inspectionTouched = false; stage = 0; renderInspector(); renderSteps();
  try {
    const form = new FormData(); form.append('file', eyeA.file);
    say('Reading your eye photograph…');
    const { id } = await request('/api/jobs/analyze', form);
    const result = await pollJob(id, 'analyze');
    if (serial !== operationSerial) return;
    analysis = result; analysisPhase = 'done'; renderSteps(); renderInspector(); setBrowserImage(result.crop, true, result.geometry);
    say('Live AI analysis complete. Open the stages, then change the iris edge in the browser-based lab.');
    setQuickProgress('Live AI analysis complete. Now change the iris edge to see the geometry recalculate.');
  } catch (err) { if (serial === operationSerial) { analysisPhase = 'error'; say(err.message, true); renderSteps(); renderInspector() } }
  finally { scanBusy = false; refreshButtons(); drainLiveAction() }
}
async function compareEyes() {  
if (!eyeA || !eyeB || compareBusy || scanBusy) return; const serial = ++operationSerial; compareBusy = true; refreshButtons(); comparison = null; inspectionTouched = false; stage = 6; renderSteps(); renderInspector();
  try {
    say('Computing ArcIris features for both eye photographs. The result will appear after both scans finish.');
    const form = new FormData(); form.append('first', eyeA.file); form.append('second', eyeB.file);
    const { id } = await request('/api/jobs/compare', form);
    const result = await pollJob(id, 'compare');
    if (serial !== operationSerial) return;
    comparison = result; analysis = result.first; analysisPhase = 'done';
    setBrowserImage(result.first.crop, true, result.first.geometry);
    say('Live ArcIris comparison complete. The two displayed scores were computed from the selected photographs.');
    setQuickProgress('Live comparison complete. Inspect the scores and try uploading another eye.');
    renderSteps(); renderInspector();
  } catch (err) { if (serial === operationSerial) say(err.message, true) } finally { compareBusy = false; refreshButtons(); drainLiveAction() }
}

// Show experimental similarity categories without asserting identity.
function researchCategory(scores) {
  if (!scores || !Number.isFinite(Number(scores.grayscale)) || !Number.isFinite(Number(scores.gamma))) return 'INCONCLUSIVE';
  if (scores.grayscale >= 0.30780 && scores.gamma >= 0.31653) return 'HIGH_SIMILARITY_IN_DEVELOPMENT_TEST';
  if (scores.grayscale < 0.218898 && scores.gamma < 0.22556365) return 'LOW_SIMILARITY_IN_DEVELOPMENT_TEST';
  return 'INCONCLUSIVE';
}
function renderMatchResult(result) {
  const score = result.similarity || {};
  const category = result.research_category || researchCategory(score);
  const high = category === 'HIGH_SIMILARITY_IN_DEVELOPMENT_TEST';
  const low = category === 'LOW_SIMILARITY_IN_DEVELOPMENT_TEST';
  const kind = high ? 'match' : low ? 'nomatch' : 'uncertain';
  const title = high ? 'POSSIBLE MATCH' : low ? 'LIKELY NO MATCH' : 'INCONCLUSIVE';
  const root = e('div', 'match-report ' + kind);
  const header = e('div', 'match-report-head');
  header.append(e('span', 'match-dot', high ? '✓' : low ? '×' : '?'), e('div', 'match-report-title', title));
  root.append(header);
  const scores = e('div', 'match-scores');
  for (const [label, value] of [['GRAYSCALE COSINE', score.grayscale], ['GAMMA COSINE', score.gamma]]) {
    const metric = e('div', 'match-score');
    metric.append(e('span', '', label), e('strong', '', Number.isFinite(Number(value)) ? Number(value).toFixed(5) : '—'));
    scores.append(metric);
  }
  root.append(scores);
  root.append(e('div', 'match-disclaimer', 'EXPERIMENTAL RESULT · Not verified identity'));
  return root;
}
function nextStatus(i) {  
if (i === 6) return comparison ? 'complete' : compareBusy ? 'working' : 'waiting';
  if (analysis?.status === 'detected') return 'complete';
  if (!analysisPhase || analysisPhase === 'error') return 'waiting';
  const value = PHASE_STAGE[analysisPhase] ?? -1;
  return i <= value ? 'complete' : i === value + 1 ? 'working' : 'waiting';
}
// Render the seven educational pipeline stages.
function renderSteps() {  
document.querySelectorAll('.process-step').forEach((node, i) => {
    node.classList.toggle('active', i === stage); node.classList.toggle('complete', nextStatus(i) === 'complete'); node.classList.toggle('working', nextStatus(i) === 'working'); node.setAttribute('aria-current', i === stage ? 'step' : 'false');
  })
}
function figureImage(url, label, mode = 'normal') {
  const fig = e('div', 'media-main' + (mode === 'wide' ? ' wide' : ''));
  if (!url) return fig;
  fig.append(img(url, label)); return fig;
}
function outputPlaceholder(title, explanation) { const div = e('div', 'empty-stage'); div.append(e('div', 'camera-box', '◎'), e('p', '', title + ' ' + explanation)); return div }
function factLine(lines) { const parent = $('inspectFacts'); parent.replaceChildren(); parent.textContent = lines.join('\n') }
function renderInspector() {
  const st = STAGES[stage], box = $('inspectMedia'); text('inspectKicker', `STEP ${String(stage + 1).padStart(2, '0')} / ${st.k}`); text('inspectTitle', st.title); text('inspectExplanation', deep ? st.deep : st.plain); text('inspectCount', `${String(stage + 1).padStart(2, '0')} / 07`);
  $('prevStep').disabled = stage === 0; $('nextStep').disabled = stage === 6;
  box.replaceChildren(); let facts = [];
  if (stage === 0) {    
if (!eyeA) { box.append(outputPlaceholder('Start with the eye library.', 'Choose an eye A and continue.')) } else {
      box.append(figureImage(eyeA.url, 'Selected eye photograph'));
      if (analysis?.quality) { const q = analysis.quality; facts = [`SOURCE ${q.width} × ${q.height} pixels`, `SHARPNESS ${q.sharpness}`, `BRIGHTNESS ${q.brightness}`, `CONTRAST ${q.contrast}`]; if (q.warnings?.length) facts.push('PHOTO ADVICE: ' + q.warnings.join(' ')); }
      else facts = ['STEP 01 · YOUR SELECTED PHOTOGRAPH', 'The pixel values enter the recognition pipeline.'];
    }  
}
  if (stage === 1) {    
if (!analysis?.segmentation) { box.append(outputPlaceholder('YOLO will mark the iris and pupil.', 'Start the analysis above to see the predicted regions.')); } else {
      box.append(figureImage(analysis.segmentation, 'Model-generated iris and pupil segmentation masks'));
      const conf = analysis.confidence; if (conf) facts = [`IRIS REGION CONFIDENCE ${(conf.iris * 100).toFixed(1)}%`, `PUPIL REGION CONFIDENCE ${(conf.pupil * 100).toFixed(1)}%`, 'GREEN = IRIS · AMBER = PUPIL'];
    }  
}
  if (stage === 2) {    
if (!analysis?.detected) box.append(outputPlaceholder('Circle fitting follows detection.', 'These boundaries come from the predicted masks.'));
    else { box.append(figureImage(analysis.detected, 'Fitted inner and outer iris circle boundaries')); const g = analysis.geometry; if (g) facts = [`IRIS RADIUS ${g.iris[2].toFixed(1)} pixels`, `PUPIL RADIUS ${g.pupil[2].toFixed(1)} pixels`, 'CIRCLES FITTED FROM ACTUAL MASKS']; }  
}
  if (stage === 3) {    
if (!analysis?.normalized_gray) { box.append(outputPlaceholder('Watch the ring become a strip.', 'The flat texture appears when the boundaries are measured.')); } else {
      const v = e('div', 'view-grid'); const one = e('div'); one.append(e('div', 'media-label', 'SOURCE: FITTED IRIS')); one.append(img(analysis.detected, 'Iris boundary circles')); const two = e('div'); two.append(e('div', 'media-label', 'OUTPUT: 64 × 512 STRIP')); two.append(img(analysis.normalized_gray, 'Unwrapped iris strip')); v.append(one, two); box.append(v); facts = ['512 columns: angle around the iris', '64 rows: distance across the iris', 'Move a point in the interactive lab below ↓'];    
}  
}
  if (stage === 4) {    
if (!analysis?.normalized_gray) { box.append(outputPlaceholder('Two views of the same texture.', 'They are calculated after normalization.')); } else {
      const v = e('div', 'view-grid'); for (const [label, url] of [['GRAYSCALE', analysis.normalized_gray], ['GAMMA 0.75', analysis.normalized_gamma]]) { const item = e('div'); item.append(e('div', 'media-label', label), img(url, `${label} iris strip`)); v.append(item) } box.append(v); facts = ['Both views use the same normalized pixels.', 'Brightness transformations, not invented patterns.'];    
}  
}
  if (stage === 5) {    
if (!analysis?.feature_profile) { box.append(outputPlaceholder('ArcIris will calculate a learned description.', 'The 512 features appear as a summary chart.')); } else {
      const frame = e('div', 'feature-card'); frame.append(e('div', 'media-label', '512 FEATURES · 16 GROUPED SUMMARIES')); const chart = e('div', 'feature-bars'); const values = analysis.feature_profile; const peak = Math.max(...values.map(x => Math.abs(x)), .00001); values.forEach((value, i) => { const b = e('i'); b.style.height = `${Math.max(3, Math.abs(value) / peak * 100)}%`; b.title = `Group ${i + 1}: ${value}`; chart.append(b) }); frame.append(chart); box.append(frame); facts = ['ResNet100 neural network', '512-dimensional vector', '16 shown groups × 32 dimensions', 'Plot summarizes computed vector values'];    
}  
}
  if (stage === 6) {    
if (!comparison) { box.append(outputPlaceholder('Choose a second photo.', 'Press “Compare A + B” to calculate a score with our trained ArcIris model.')); } else {
      const c = e('div', 'comparison-result'); const v = e('div', 'comparison-big'); v.append(img(comparison.first?.detected, 'Processed eye A'), e('span', '', '↔'), img(comparison.second?.detected, 'Processed eye B')); c.append(v, renderMatchResult(comparison)); box.append(c); facts = ['SIMILARITY SCORES: −1 TO +1', 'YOLO + ARCIRIS RESNET100', 'EXPERIMENTAL THRESHOLDS'];    
}  
}
  factLine(facts);
  if (typeof window.refreshScience === 'function') window.refreshScience();
}
function stepTo(index) { stage = Math.max(0, Math.min(6, index)); inspectionTouched = true; renderInspector(); renderSteps() }

// --- The browser's own polar-to-rectangular mapping, recalculated on every change. ---
const srcCanvas = $('sourceCanvas'), dstCanvas = $('stripCanvas'), srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true }), dstCtx = dstCanvas.getContext('2d');
const off = document.createElement('canvas'); off.width = 560; off.height = 440; const ctx = off.getContext('2d', { willReadFrequently: true });
const raw = new Image(); let imageReady = false; let useFitted = false;
function setBrowserImage(url, model = false, geom = null) {
  imageReady = false; useFitted = Boolean(model && geom); sourceGeom = geom || null;
  raw.onload = () => {
    canvasImage = raw; imageReady = true; manualCenter = { x: off.width * .5, y: off.height * .5 };
    if (sourceGeom) { const scale = Math.min(off.width / raw.naturalWidth, off.height / raw.naturalHeight); manualCenter = { x: off.width / 2, y: off.height / 2 }; }
    redrawMechanism();
  };
  raw.onerror = () => { imageReady = false; redrawMechanism() };
  raw.src = url;
}

// Interactive circle/normalization preview; not a model inference.
function getGeometry() {
  if (!useFitted || !sourceGeom || !canvasImage) return null;

  const w = off.width, h = off.height;
  const imageW = canvasImage.naturalWidth;
  const imageH = canvasImage.naturalHeight;

  const sc = Math.min(w / imageW, h / imageH);
  const ox = (w - imageW * sc) / 2;
  const oy = (h - imageH * sc) / 2;

  const box = sourceGeom.crop_box;
  let sx = 1, sy = 1;

  if (box && box.length === 4) {
    const originalW = box[2] - box[0];
    const originalH = box[3] - box[1];

    if (originalW > 0 && originalH > 0) {
      sx = imageW / originalW;
      sy = imageH / originalH;
    }
  }

  const innerOffset = Number($('inner').value);
  const outerOffset = Number($('outer').value);

  const p = sourceGeom.pupil;
  const i = sourceGeom.iris;

  const pupil = {
    x: ox + p[0] * sx * sc,
    y: oy + p[1] * sy * sc,
    r: Math.max(2, p[2] * ((sx + sy) / 2) * sc + innerOffset)
  };

  const iris = {
    x: ox + i[0] * sx * sc,
    y: oy + i[1] * sy * sc,
    r: Math.max(3, i[2] * ((sx + sy) / 2) * sc + outerOffset)
  };

  if (pupil.r >= iris.r - 4) {
    pupil.r = iris.r - 4;
  }

  return { iris, pupil };
}
function plotCircle(c, g, color) { c.beginPath(); c.arc(g.x, g.y, g.r, 0, Math.PI * 2); c.strokeStyle = color; c.lineWidth = 2.5; c.setLineDash([7, 6]); c.stroke(); c.setLineDash([]) }
function drawPoint(ctx, x, y, r, color) { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#1a2927'; ctx.stroke() }
function sampleRGB(data, x, y) {
  const w = off.width, h = off.height; if (x < 0 || x >= w - 1 || y < 0 || y >= h - 1) return [23, 35, 33, 255];
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const p = (ix, iy, c) => data[(iy * w + ix) * 4 + c];
  const out = [0, 0, 0, 255]; for (let c = 0; c < 3; c++)out[c] = Math.round((1 - fy) * ((1 - fx) * p(x0, y0, c) + fx * p(x0 + 1, y0, c)) + fy * ((1 - fx) * p(x0, y0 + 1, c) + fx * p(x0 + 1, y0 + 1, c))); return out;
}
function redrawMechanism() {
  srcCtx.clearRect(0, 0, srcCanvas.width, srcCanvas.height); dstCtx.clearRect(0, 0, dstCanvas.width, dstCanvas.height);
  if (!imageReady) { srcCtx.fillStyle = '#30453f'; srcCtx.fillRect(0, 0, 560, 440); srcCtx.fillStyle = '#f0d6b8'; srcCtx.font = '22px Georgia'; srcCtx.fillText('Select an eye above', 133, 222); dstCtx.fillStyle = '#253c38'; dstCtx.fillRect(0, 0, 512, 128); return; }
  const iw = canvasImage.naturalWidth, ih = canvasImage.naturalHeight; const scale = Math.min(off.width / iw, off.height / ih); const dx = (off.width - iw * scale) / 2, dy = (off.height - ih * scale) / 2;
  ctx.fillStyle = '#192b2a'; ctx.fillRect(0, 0, off.width, off.height); ctx.drawImage(canvasImage, dx, dy, iw * scale, ih * scale);
  const pix = ctx.getImageData(0, 0, off.width, off.height), geo = getGeometry(), theta = Number($('angle').value) * Math.PI / 180, depth = Number($('depth').value) / 100;
  if (!geo) {    
srcCtx.putImageData(pix, 0, 0); dstCtx.fillStyle = '#253c38'; dstCtx.fillRect(0, 0, 512, 128);
    $('mechStatus').textContent = 'Analyze the image to display the measured iris and pupil boundaries.';
    $('coordReadout').textContent = 'Waiting for the measured iris and pupil boundaries.'; return;  
}
  const inner = geo.pupil, outer = geo.iris;
  const pixelPoint = (t, r) => ({ x: (1 - r) * (inner.x + inner.r * Math.cos(t)) + r * (outer.x + outer.r * Math.cos(t)), y: (1 - r) * (inner.y + inner.r * Math.sin(t)) + r * (outer.y + outer.r * Math.sin(t)) });
  const target = dstCtx.createImageData(512, 128);
  for (let y = 0; y < 128; y++) { const r = y / 127; for (let x = 0; x < 512; x++) { const pt = pixelPoint((x / 512) * 2 * Math.PI, r), col = sampleRGB(pix.data, pt.x, pt.y), n = (y * 512 + x) * 4; target.data[n] = col[0]; target.data[n + 1] = col[1]; target.data[n + 2] = col[2]; target.data[n + 3] = 255; } }
  dstCtx.putImageData(target, 0, 0);
  srcCtx.putImageData(pix, 0, 0); plotCircle(srcCtx, inner, '#f8ab7f'); plotCircle(srcCtx, outer, '#c8f3df'); const p = pixelPoint(theta, depth);
  srcCtx.beginPath(); srcCtx.moveTo(inner.x + inner.r * Math.cos(theta), inner.y + inner.r * Math.sin(theta)); srcCtx.lineTo(outer.x + outer.r * Math.cos(theta), outer.y + outer.r * Math.sin(theta)); srcCtx.strokeStyle = '#ffffffba'; srcCtx.lineWidth = 2; srcCtx.stroke(); drawPoint(srcCtx, p.x, p.y, 8, '#f17851');
  const markerX = Number($('angle').value) / 360 * 512, markerY = depth * 127;
  dstCtx.fillStyle = '#fff'; dstCtx.fillRect(markerX, 0, 1.8, 128); drawPoint(dstCtx, markerX, markerY, 5, '#f17851');
  $('coordReadout').textContent = `${$('angle').value}° around the iris · ${$('depth').value}% between the pupil and the iris edge · source pixel (${p.x.toFixed(0)}, ${p.y.toFixed(0)}) → strip (${Math.floor(markerX)}, ${Math.floor(markerY)}).`;
  $('mechStatus').textContent = 'At 0 px, the circles show the fitted boundaries. Adjust a boundary to see the effect on the strip.';
  if (typeof window.refreshScience === 'function') window.refreshScience();
}
function updateSliders() {
  text('angleOut', $('angle').value + '°'); text('depthOut', $('depth').value + '%');
  text('innerOut', String($('inner').value) + ' px'); text('outerOut', String($('outer').value) + ' px');
  if (imageReady) redrawMechanism();
  if (useFitted && sourceGeom && (Number($('inner').value) !== 0 || Number($('outer').value) !== 0)) {
    setQuickProgress('The browser recomputed the iris strip from your changed boundaries. ArcIris scores have NOT changed; rerun model processing for a new AI score.');
  }
}
['angle', 'depth', 'inner', 'outer'].forEach(id => $(id).addEventListener('input', updateSliders));
// Clicking the eye must NOT erase YOLO's measured geometry. Keep source data intact.
// Visitors can experiment by changing boundary offsets instead.

function setQuickProgress(message) {
  const node = $('quickProgress'); if (node) node.textContent = message;
}
function quickGo(id) { const node = $(id); if (node) node.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
function bindQuickStart() {
  const analyze = $('quickAnalyze'), unwrap = $('quickUnwrap'), compare = $('quickCompare');
  if (analyze) analyze.addEventListener('click', () => {
    quickGo('analysis');
    if (!eyeA) { setQuickProgress('Select eye A from the library first.'); return; }
    setQuickProgress('Running real YOLO and ArcIris analysis on eye A…'); analyzeEye();
  });
  if (unwrap) unwrap.addEventListener('click', () => {
    quickGo('mechanism');
    if (!useFitted) { setQuickProgress('First analyze eye A to load its measured iris boundaries.'); return; }
    const slider = $('outer'); slider.value = slider.value === '0' ? '8' : '0';
    updateSliders(); slider.focus({ preventScroll: true });
  });
  if (compare) compare.addEventListener('click', () => {
    quickGo('analysis');
    if (!eyeA || !eyeB) { setQuickProgress('Select both eye A and eye B first.'); return; }
    setQuickProgress('Running real YOLO and ArcIris comparison for both eyes…'); compareEyes();
  });
}
// Bind page controls and initialize the interface.
function bindEvents() {
  document.querySelectorAll('button[data-which][data-sample]').forEach(b => b.addEventListener('click', () => chooseSample(b.dataset.sample, b.dataset.which)));
  $('uploadA').addEventListener('change', ev => chooseFile(ev.target.files[0], 'a'));
  $('uploadB').addEventListener('change', ev => chooseFile(ev.target.files[0], 'b'));
  $('phoneA').addEventListener('change', ev => chooseFile(ev.target.files[0], 'a'));
  $('analyzeBtn').addEventListener('click', analyzeEye); $('compareBtn').addEventListener('click', compareEyes);
  document.querySelectorAll('.process-step').forEach((node, i) => node.addEventListener('click', () => stepTo(i)));
  $('prevStep').addEventListener('click', () => stepTo(stage - 1)); $('nextStep').addEventListener('click', () => stepTo(stage + 1));
  $('modePlain').addEventListener('click', () => { deep = false; $('modePlain').classList.add('active'); $('modeDeep').classList.remove('active'); renderInspector() });
  $('modeDeep').addEventListener('click', () => { deep = true; $('modeDeep').classList.add('active'); $('modePlain').classList.remove('active'); renderInspector() });
  document.querySelectorAll('.specimen-img img').forEach(im => im.addEventListener('error', () => {
    im.alt = 'This sample photograph is unavailable.';
    im.closest('.specimen')?.classList.add('photo-unavailable');
  }));
}
async function statusEngine() {
  try { const r = await fetch('/api/status', { cache: 'no-store' }); const d = await r.json(); $('readyChip').classList.toggle('is-ready', Boolean(d.ready)); text('readyText', d.ready ? 'AI model files found' : 'Check model files'); text('serverState', d.ready ? 'YOLO + ArcIris available' : 'Models not connected'); }
  catch { say('Could not reach the AI server. Launch START_WINDOWS.bat.', true); text('readyText', 'AI offline'); text('serverState', 'AI server offline') }
}
async function bootstrap() {
  renderInspector(); renderSteps(); redrawMechanism();
  await statusEngine();
  try { const r = await fetch('/api/prepared', { cache: 'no-store' }); if (r.ok) { const json = await r.json(); if (json.ready) preparedCache = json; } }
  catch (err) { console.warn('Prepared sample cache unavailable', err) }
  try {
    await chooseSample('original', 'a');
    await chooseSample(preparedCache?.defaultB || 'green', 'b');
  } finally { booting = false }
  if (applyPreparedSelection()) {
    text('readyText', 'Blue reference ready'); text('serverState', 'Blue reference loaded · all other eyes processed live');
  } else {
    say('The blue reference is not ready. Please check the server setup.', true);
    renderInspector(); renderSteps();
  }
}
bindEvents(); bindQuickStart(); bootstrap();

// Scientific explanations and browser-only visualization tools.
'use strict';
// This script adds scientific interpretations to HAWIYA without modifying AI inference.
(function () {
  const node = id => document.getElementById(id);
  const set = (id, value) => { const el = node(id); if (el) el.textContent = value; };
  const numeric = value => Number.isFinite(Number(value)) ? Number(value) : null;
  const fmt = (value, digits = 2) => numeric(value) === null ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
  const ctxOf = id => node(id).getContext('2d', { willReadFrequently: true });
  let previewImage = new Image(), stripImage = new Image(), loadedImage = '', loadedStrip = '';
  let lightReady = false, stripReady = false;
  function loadImage(url, kind) {
    if (!url) return;
    if (kind === 'photo' && loadedImage === url) return;
    if (kind === 'strip' && loadedStrip === url) return;
    const im = kind === 'photo' ? previewImage : stripImage;
    if (kind === 'photo') { loadedImage = url; lightReady = false; } else { loadedStrip = url; stripReady = false; }
    im.onload = () => { if (kind === 'photo') { lightReady = true; drawLight(); } else { stripReady = true; drawGamma(); } };
    im.onerror = () => { if (kind === 'photo') lightReady = false; else stripReady = false; };
    im.src = url;
  }
  function drawLight() {
    if (!lightReady) return;
    const c = node('lightPreview'), x = c.getContext('2d', { willReadFrequently: true }), shift = Number(node('lightOffset').value);
    x.clearRect(0, 0, c.width, c.height); x.drawImage(previewImage, 0, 0, c.width, c.height);
    const frame = x.getImageData(0, 0, c.width, c.height), a = frame.data, hist = Array(256).fill(0); let sum = 0;
    for (let i = 0; i < a.length; i += 4) {
      const r = Math.max(0, Math.min(255, a[i] + shift)), g = Math.max(0, Math.min(255, a[i + 1] + shift)), b = Math.max(0, Math.min(255, a[i + 2] + shift));
      a[i] = r; a[i + 1] = g; a[i + 2] = b;
      const y = Math.max(0, Math.min(255, Math.round(.299 * r + .587 * g + .114 * b))); hist[y]++; sum += y;
    }
    x.putImageData(frame, 0, 0);
    const h = node('histogramCanvas'), hc = h.getContext('2d'), max = Math.max(...hist, 1);
    hc.clearRect(0, 0, h.width, h.height); hc.fillStyle = '#152a28'; hc.fillRect(0, 0, h.width, h.height);
    hc.fillStyle = '#7bd0ab'; for (let b = 0; b < 256; b++)hc.fillRect(b * h.width / 256, h.height - 8 - (hist[b] / max) * (h.height - 18), Math.max(1, h.width / 256), hist[b] / max * (h.height - 18));
    set('lightOut', (shift > 0 ? '+' : '') + shift); set('lightMetrics', 'Preview mean: ' + fmt(sum / (a.length / 4), 1) + ' / 255 · live 256-bin histogram');
  }
  function drawGamma() {
    if (!stripReady) return;
    const c = node('gammaPreview'), ctx = c.getContext('2d', { willReadFrequently: true }); ctx.clearRect(0, 0, c.width, c.height); ctx.drawImage(stripImage, 0, 0, c.width, c.height);
    const frame = ctx.getImageData(0, 0, c.width, c.height), a = frame.data, gamma = Number(node('gammaLab').value);
    for (let i = 0; i < a.length; i += 4)for (let j = 0; j < 3; j++)a[i + j] = Math.round(255 * Math.pow(a[i + j] / 255, gamma));
    ctx.putImageData(frame, 0, 0); set('gammaOut', gamma.toFixed(2)); set('gammaMetrics', 'Browser-only gamma preview · model inputs and embeddings are unchanged');
  }
  function updateModelNumbers() {
    if (typeof analysis === 'undefined' || !analysis || analysis.status !== 'detected') {
      set('dataOrigin', 'Select an eye and press Analyze to populate these measurements. Sample preview values are shown only when cached model output has loaded.'); return;
    }
    const q = analysis.quality || {}, g = analysis.geometry || {}, ir = g.iris, pu = g.pupil, confidence = analysis.confidence || {};
    const cache = !!(eyeA && eyeA.sample && preparedCache?.analyses?.[eyeA.sample] === analysis);
    set('dataOrigin', cache ? 'SOURCE: Precomputed results from the trained YOLO + ArcIris models. Press Analyze to recompute live.' : 'SOURCE: Backend model analysis. Browser experiments below are calculated independently.');
    set('mResolution', q.width && q.height ? `${q.width} × ${q.height}` : '—');
    set('mBrightness', fmt(q.brightness, 1)); set('mFocus', fmt(q.sharpness, 1));
    set('mRatio', ir && pu ? fmt(pu[2] / ir[2], 3) : '—');
    set('mArea', ir && pu ? fmt(Math.PI * (ir[2] ** 2 - pu[2] ** 2), 0) : '—');
    set('mFeatures', analysis.embedding_dimensions === 512 ? '512 values' : '—');
    set('mConf', numeric(confidence.iris) !== null && numeric(confidence.pupil) !== null ? `${fmt(confidence.iris * 100, 1)}% / ${fmt(confidence.pupil * 100, 1)}%` : '—');
    set('mIrisCenter', ir ? `${fmt(ir[0], 1)}, ${fmt(ir[1], 1)} px` : '—');
    set('mPupilCenter', pu ? `${fmt(pu[0], 1)}, ${fmt(pu[1], 1)} px` : '—');
    set('mRadii', ir && pu ? `${fmt(ir[2], 1)} / ${fmt(pu[2], 1)} px` : '—');
    if (analysis.crop) loadImage(analysis.crop, 'photo');
    if (analysis.normalized_gray) loadImage(analysis.normalized_gray, 'strip');
    const bars = node('vectorBars'); bars.replaceChildren();
    const features = analysis.feature_profile;
    if (Array.isArray(features) && features.length) {
      const max = Math.max(.00001, ...features.map(v => Math.abs(v)));
      features.forEach((v, i) => { const bar = document.createElement('i'); bar.style.height = Math.max(3, Math.abs(v) / max * 100) + '%'; bar.className = v < 0 ? 'negative' : ''; bar.title = `Group ${i + 1} mean: ${fmt(v, 6)}`; bars.append(bar); });
    }
    const similarity = typeof comparison !== 'undefined' && comparison?.similarity;
    set('mGrayScore', similarity ? fmt(similarity.grayscale, 5) : '—'); set('mGammaScore', similarity ? fmt(similarity.gamma, 5) : '—');
  }
  function updateExperimentNumbers() {
    if (typeof getGeometry !== 'function' || typeof useFitted === 'undefined' || !useFitted || !sourceGeom || !imageReady) {
      set('mAreaDelta', '—'); set('mStripDifference', '—'); return;
    }
    const current = getGeometry(); if (!current) return;
    const a = node('inner'), b = node('outer'), va = a.value, vb = b.value;
    // Obtain zero-offset geometry without recomputing backend data.
    let reference; try { a.value = '0'; b.value = '0'; reference = getGeometry(); } finally { a.value = va; b.value = vb; }
    if (!reference) return;
    const area = g => Math.PI * Math.max(0, g.iris.r * g.iris.r - g.pupil.r * g.pupil.r);
    set('mAreaDelta', `${fmt((area(current) / Math.max(1, area(reference)) - 1) * 100, 2)}% vs 0 px`);
    const original = ctx.getImageData(0, 0, off.width, off.height).data;
    function point(g, t, r) { return { x: (1 - r) * (g.pupil.x + g.pupil.r * Math.cos(t)) + r * (g.iris.x + g.iris.r * Math.cos(t)), y: (1 - r) * (g.pupil.y + g.pupil.r * Math.sin(t)) + r * (g.iris.y + g.iris.r * Math.sin(t)) }; }
    let sq = 0, n = 0;
    for (let row = 0; row < 16; row++)for (let col = 0; col < 64; col++) {
      const radius = (row + .5) / 16, angle = col * Math.PI * 2 / 64;
      const p = point(current, angle, radius), r = point(reference, angle, radius);
      const c1 = sampleRGB(original, p.x, p.y), c2 = sampleRGB(original, r.x, r.y);
      for (let ch = 0; ch < 3; ch++) { sq += (c1[ch] - c2[ch]) ** 2; n++; }
    }
    set('mStripDifference', fmt(Math.sqrt(sq / n), 2) + ' / 255 sampled RMS');
  }
  function refresh() { updateModelNumbers(); updateExperimentNumbers(); }
  window.refreshScience = refresh;
  node('lightOffset').addEventListener('input', drawLight);
  node('gammaLab').addEventListener('input', drawGamma);
  refresh();
})();
