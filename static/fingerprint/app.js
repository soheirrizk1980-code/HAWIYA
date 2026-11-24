/**
 * Fingerprint Journey: an educational, browser-only simulation using synthetic
 * ridge features. No biometric photographs are captured or stored here.
 */
'use strict';
const $ = id => document.getElementById(id);
const pages = [...document.querySelectorAll('.page')];
const navItems = [...document.querySelectorAll('.nav-item')];
let currentPage = 0;
// Navigate between the sections of the fingerprint journey.
function showPage(i) {
  currentPage = Math.max(0, Math.min(pages.length - 1, i));
  pages.forEach((p, idx) => p.classList.toggle('active', idx === currentPage));
  navItems.forEach((n, idx) => n.classList.toggle('active', idx === currentPage));
  $('progressNumber').textContent = `0${currentPage + 1} / 06`.replace('010', '10');
  $('progressFill').style.width = ((currentPage + 1) / pages.length * 100) + '%';
  $('sidebar').classList.remove('open');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  if (currentPage === 3) { updateExperiment(); drawScanCanvas() }
  if (currentPage === 4) { runMatch() }
}
document.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => showPage(Number(b.dataset.go))));
navItems.forEach(n => n.addEventListener('click', () => showPage(Number(n.dataset.page))));
$('logo').addEventListener('click', e => { e.preventDefault(); showPage(0) });
$('menuButton').addEventListener('click', () => $('sidebar').classList.toggle('open'));
$('bigIdeaButton').addEventListener('click', () => { $('bigIdea').hidden = !$('bigIdea').hidden });

// Draw illustrative fingerprint and scanning canvases.
function addGrid(ctx, w, h) {
  ctx.fillStyle = '#f4f1e9'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(100,150,210,.12)'; ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 24) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke() }
  for (let y = 0; y < h; y += 24) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke() }
}
function ridgePattern(ctx, w, h, id = 0, alpha = .9) {
  ctx.save();
  ctx.translate(w / 2, h / 2);
  const s = Math.min(w, h) * .76; ctx.scale(s, s);
  ctx.lineWidth = 1.2 / s; ctx.strokeStyle = `rgba(15,115,232,${alpha})`;
  for (let k = 0; k < 22; k++) {
    ctx.beginPath(); let first = true;
    for (let j = 0; j <= 90; j++) {
      const t = j / 90, y = (t - .5) * 1.08, z = (k - 10.5) / 10.5;
      const shape = id === 0 ? .15 * Math.sin((t - .15) * Math.PI * 1.6) : id === 1 ? .1 * Math.sin(t * Math.PI * 2 + 1.2) : .06 * Math.sin(t * Math.PI);
      const x = z * .38 + shape * (.18 + Math.abs(z) * .6) + .03 * Math.sin(10 * y + id * 1.8) * (.7 - Math.abs(z) * .45);
      if (first) { ctx.moveTo(x, y); first = false } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}
function point(ctx, x, y, type, r = 4) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = type === 'ending' ? '#547c71' : '#c58562'; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, r + 4, 0, Math.PI * 2); ctx.strokeStyle = type === 'ending' ? 'rgba(15,115,232,.28)' : 'rgba(240,165,78,.3)'; ctx.stroke();
}
function rng(seed) { let s = seed >>> 0; return () => { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296 } }
function makeTemplate(id) {
  const r = rng(4200 + id * 777); const pts = [];
  for (let i = 0; i < 28; i++) {
    const row = Math.floor(i / 6), col = i % 6;
    let x = .18 + col * .126 + (r() - .5) * .045, y = .13 + row * .18 + (r() - .5) * .065;
    if (y > .9) continue;
    pts.push({ x, y, type: r() > .45 ? 'ending' : 'split' });
  }
  return pts;
}
const templates = [makeTemplate(0), makeTemplate(1), makeTemplate(2)];
const names = ['Alex', 'Melissa', 'Jack'];

function drawHero() { const c = $('heroCanvas'), ctx = c.getContext('2d'); addGrid(ctx, c.width, c.height); ridgePattern(ctx, c.width, c.height, 0, .8); templates[0].slice(0, 8).forEach(p => point(ctx, c.width * (.19 + p.x * .62), c.height * (.11 + p.y * .76), p.type, 3)); }
function drawPatternPreview() { const c = $('patternPreview'), ctx = c.getContext('2d'); addGrid(ctx, c.width, c.height); ridgePattern(ctx, c.width, c.height, 1, .65) }
function drawSensorPhoto() {  
const c = $('sensorPhotoCanvas'), ctx = c.getContext('2d'); const w = c.width, h = c.height; addGrid(ctx, w, h); ridgePattern(ctx, w, h, 0, .18); ctx.save(); ctx.globalAlpha = .6; ridgePattern(ctx, w, h, 0, .6); ctx.restore();
  ctx.fillStyle = 'rgba(255,255,255,.38)'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = 'rgba(15,115,232,.08)'; for (let i = 0; i < 13; i++) { ctx.fillRect(24 + i * 32, 0, 1, h) }
}

const sensorInfo = {
  capacitive: { title: 'Capacitive sensor', text: 'It measures tiny electrical differences between the ridges and valleys that touch the surface. This is common in many phones.' },
  optical: { title: 'Optical sensor', text: 'It uses light to capture an image of the fingerprint pattern, then software analyzes that image.' },
  ultrasonic: { title: 'Ultrasonic sensor', text: 'It sends sound waves and reads their reflections to build a detailed 3D-like map of the fingerprint.' }
};
// Explain the different physical sensor technologies.
function setSensor(type) {
  document.querySelectorAll('.sensor-tab').forEach(b => b.classList.toggle('active', b.dataset.sensor === type));
  $('sensorDescription').innerHTML = `<h3>${sensorInfo[type].title}</h3><p>${sensorInfo[type].text}</p>`;
}
document.querySelectorAll('.sensor-tab').forEach(b => b.addEventListener('click', () => setSensor(b.dataset.sensor)));
setSensor('capacitive');

let logicStep = 0;
function drawLogic() {
  const c = $('logicCanvas'), ctx = c.getContext('2d'), w = c.width, h = c.height; addGrid(ctx, w, h); const map = ['Fingerprint image', 'Minutiae points', 'Saved template'];
  const text = ['The software starts with a fingerprint pattern from the sensor.', 'Then it finds special points such as ridge endings and splits.', 'Finally it saves a small template made from those important points.'];
  $('logicTitle').textContent = map[logicStep]; $('logicText').textContent = text[logicStep];
  document.querySelectorAll('.logic-box').forEach((b, idx) => b.classList.toggle('active', idx === logicStep));
  if (logicStep === 0) { ridgePattern(ctx, w, h, 0, .8); $('templateBox').textContent = 'Template preview will appear here.' }
  if (logicStep === 1) { ridgePattern(ctx, w, h, 0, .38); templates[0].forEach(p => point(ctx, w * (.19 + p.x * .62), h * (.11 + p.y * .76), p.type, 4)); $('templateBox').textContent = 'Detected points: endings + splits'; }
  if (logicStep === 2) { templates[0].slice(0, 10).forEach((p, idx) => point(ctx, 70 + (idx % 5) * 80, 80 + Math.floor(idx / 5) * 80, p.type, 5)); $('templateBox').textContent = `{ id: 'A-001', features: ${templates[0].length}, types: ['ending','split'] }`; }
}
document.querySelectorAll('.logic-box').forEach(b => b.addEventListener('click', () => { logicStep = Number(b.dataset.logic); drawLogic() }));
drawLogic();

// Experiment page
const scannerGrid = $('scannerGrid');
for (let i = 0; i < 160; i++) { const d = document.createElement('div'); scannerGrid.appendChild(d) }
let showCells = false;
// Demonstrate how contact and image quality affect an illustrative scan.
function updateExperiment() {
  const pos = Number($('fingerSlider').value), quality = Number($('qualitySlider').value) / 100;
  $('fingerOutput').textContent = pos + '%'; $('qualityOutput').textContent = Math.round(quality * 100) + '%';
  $('contactValue').textContent = pos + '% contact';
  $('contactText').textContent = pos < 18 ? 'Finger above sensor' : pos < 60 ? 'Finger touching sensor' : 'Finger pressed on sensor';
  $('stageFinger').style.top = (18 + pos * 1.36) + 'px';
  $('bigScanner').classList.remove('scanning'); void $('bigScanner').offsetWidth; if (pos > 18) $('bigScanner').classList.add('scanning');
  [...scannerGrid.children].forEach((cell, i) => {
    const x = i % 16, y = Math.floor(i / 16), ridge = Math.sin((x - 7.5) * .9 + Math.sin(y * .62) * 1.6) > 0.05, amount = (pos / 100) * quality, lit = (ridge && amount > .18 && rng(700 + i + Math.floor(amount * 100))() < amount);
    cell.classList.toggle('lit', lit && showCells);
    cell.style.opacity = showCells ? (.48 + .52 * amount) : 0.9;
  });
}
function drawScanCanvas() {
  const c = $('scanCanvas'), ctx = c.getContext('2d'), w = c.width, h = c.height; addGrid(ctx, w, h); const pos = Number($('fingerSlider').value) / 100, q = Number($('qualitySlider').value) / 100; const clarity = Math.max(.2, (pos * .55 + q * .45));
  ctx.save(); ctx.globalAlpha = .28 + .58 * clarity; ridgePattern(ctx, w, h, 0, .8 * clarity); ctx.restore();
  if (showCells) { ctx.strokeStyle = 'rgba(99,150,210,.18)'; for (let x = 0; x < w; x += 22) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke() } for (let y = 0; y < h; y += 22) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke() } }
  $('scanSummary').textContent = clarity > .72 ? 'Clear scan: many ridge details can be read.' : clarity > .42 ? 'Medium scan: some useful details are visible.' : 'Weak scan: the software may struggle to read enough detail.';
}
$('fingerSlider').addEventListener('input', () => { updateExperiment(); drawScanCanvas() });
$('qualitySlider').addEventListener('input', () => { updateExperiment(); drawScanCanvas() });
$('togglePixels').addEventListener('click', () => { showCells = !showCells; $('togglePixels').textContent = showCells ? 'Hide cells' : 'Show cells'; updateExperiment(); drawScanCanvas(); });
$('autoMove').addEventListener('click', () => {
  let value = 0; $('fingerSlider').value = 0; updateExperiment(); drawScanCanvas();
  const timer = setInterval(() => { value += 5; $('fingerSlider').value = value; updateExperiment(); drawScanCanvas(); if (value >= 100) clearInterval(timer) }, 50);
});
updateExperiment(); drawScanCanvas();

// Match lab
let savedPerson = 0, scanPerson = 0, scanNonce = 0;
// Generate simulated feature points and show an educational match score.
function newScan(id, rotDeg, nonce) {
  const rad = rotDeg * Math.PI / 180, r = rng(5000 + id * 100 + nonce * 31 + rotDeg * 4); const pts = [];
  for (const p of templates[id]) {
    if (r() < .12) continue;
    const dx = p.x - .5, dy = p.y - .5;
    let x = .5 + dx * Math.cos(rad) - dy * Math.sin(rad), y = .5 + dx * Math.sin(rad) + dy * Math.cos(rad);
    x += (r() - .5) * .03; y += (r() - .5) * .03;
    pts.push({ x, y, type: p.type });
  }
  return pts;
}
function alignPoint(p, degrees) { let rad = degrees * Math.PI / 180, dx = p.x - .5, dy = p.y - .5; return { x: .5 + dx * Math.cos(rad) - dy * Math.sin(rad), y: .5 + dx * Math.sin(rad) + dy * Math.cos(rad), type: p.type }; }
function scoreMatch(ref, scan) { let best = 0, bestCount = 0; for (let deg = -36; deg <= 36; deg += 3) { let used = new Set(), count = 0; for (const p of scan) { let q = alignPoint(p, deg), near = -1, dist = .072; for (let j = 0; j < ref.length; j++) { if (used.has(j) || ref[j].type !== q.type) continue; const d = Math.hypot(ref[j].x - q.x, ref[j].y - q.y); if (d < dist) { dist = d; near = j; } } if (near >= 0) { used.add(near); count++; } } const score = 2 * count / (ref.length + scan.length || 1); if (score > best) { best = score; bestCount = count; } } return { score: Math.round(best * 100), matches: bestCount }; }
function drawTemplateMap(canvasId, pts, label) {
  const c = $(canvasId), ctx = c.getContext('2d'), w = c.width, h = c.height; addGrid(ctx, w, h); ridgePattern(ctx, w, h, 0, .14); ctx.fillStyle = '#66796c'; ctx.font = '700 11px Inter'; ctx.fillText(label, 14, 18); pts.forEach(p => point(ctx, w * (.19 + p.x * .62), h * (.13 + p.y * .72), p.type, 4));
}
function runMatch() {
  const rotation = Number($('rotationSlider').value), threshold = Number($('thresholdSlider').value), scanPts = newScan(scanPerson, rotation, scanNonce), result = scoreMatch(templates[savedPerson], scanPts);
  document.querySelectorAll('#savedPeople .person-card').forEach((b, idx) => b.classList.toggle('active', idx === savedPerson));
  document.querySelectorAll('#candidateList .candidate').forEach((b, idx) => b.classList.toggle('active', idx === scanPerson));
  $('savedName').textContent = names[savedPerson]; $('scanName').textContent = names[scanPerson];
  $('rotationOutput').textContent = rotation + '°'; $('thresholdOutput').textContent = threshold + '%';
  drawTemplateMap('savedCanvas', templates[savedPerson], 'SAVED TEMPLATE');
  drawTemplateMap('matchCanvas', scanPts, 'NEW SCAN');
  $('scoreValue').textContent = result.score + '%'; $('scoreFill').style.width = result.score + '%';
  const ok = result.score >= threshold; $('decision').textContent = ok ? 'MATCH' : 'NO MATCH'; $('decision').className = 'decision ' + (ok ? 'yes' : 'no');
  $('decisionText').textContent = ok ? `The new scan is close enough to identify ${names[savedPerson]}.` : 'The score is below the threshold, so the device rejects the scan.';
}
document.querySelectorAll('#savedPeople .person-card').forEach(b => b.addEventListener('click', () => { savedPerson = Number(b.dataset.enroll); runMatch(); }));
document.querySelectorAll('#candidateList .candidate').forEach(b => b.addEventListener('click', () => { scanPerson = Number(b.dataset.scan); runMatch(); }));
$('rotationSlider').addEventListener('input', runMatch); $('thresholdSlider').addEventListener('input', runMatch); $('rescanButton').addEventListener('click', () => { scanNonce++; runMatch() });
runMatch();

// Uses
const useInfo = {
  phone: ['Phone unlock', 'The device compares a new scan with templates already enrolled on the phone. If one matches, the phone unlocks.'],
  door: ['Secure doors', 'A fingerprint device can control physical access by checking whether the scan matches an authorized record.'],
  school: ['Attendance', 'The system links a matching fingerprint template to a stored ID and logs that the person checked in.']
};
document.querySelectorAll('.use-card').forEach(card => card.addEventListener('click', () => {
  document.querySelectorAll('.use-card').forEach(c => c.classList.toggle('active', c === card));
  const key = card.dataset.use; $('useDetail').innerHTML = `<h3>${useInfo[key][0]}</h3><p>${useInfo[key][1]}</p>`;
}));

drawHero(); drawPatternPreview(); drawSensorPhoto();
