import './style.css';
import { ACTIONS, SIZE, LIMIT, World, generate, route, cloneGarden, probabilities, type Policy } from './engine';
import { teach, type Lesson } from './learn';

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id)! as T;
const canvas = el<HTMLCanvasElement>('garden'), ctx = canvas.getContext('2d')!;
const portrait = el<HTMLCanvasElement>('portrait'), portraitCtx = portrait.getContext('2d')!;
const brain = el<HTMLSelectElement>('brain'), timeline = el<HTMLInputElement>('timeline');
const runButton = el<HTMLButtonElement>('run');
let garden = generate(42), world = new World(cloneGarden(garden));
let mode = 'watch', tile = 1, running = false, frame = 0;
let policies: Record<string, Policy> = {};
let history = [{ pos: world.pos, steps: 0, outcome: 'running' }];
let episodeActions: number[] = [], demonstration: Lesson[] = [], lessons: Lesson[] = [];
let animationTimer: ReturnType<typeof setInterval> | undefined;
const notes: Record<string, string> = {
  'bc-ppo': 'A head start from 256 expert routes, followed by 65,536 reward-learning steps.',
  'ppo': '65,536 steps of exploring and learning from rewards. No demonstrations.',
  'bc': 'Learned to imitate 256 expert routes. No reward-based fine-tuning.',
  'untrained': 'The same network before training. Curiosity without experience.',
  'personal': 'Updated neural weights from your routes. Session only. A familiar route can overfit.'
};

function critter(c: CanvasRenderingContext2D, x: number, y: number, scale: number, bounce = 0, success = false) {
  c.save(); c.translate(x, y + bounce); c.scale(scale, scale);
  c.fillStyle = '#49634625'; c.beginPath(); c.ellipse(0, 24, 24, 7, 0, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#709b7e'; c.beginPath(); c.ellipse(-12, -15, 8, 14, -.3, 0, Math.PI * 2); c.fill();
  c.beginPath(); c.ellipse(12, -15, 8, 14, .3, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#a9c4a1'; c.beginPath(); c.roundRect(-26, -20, 52, 46, 19); c.fill();
  c.fillStyle = '#bad2ad'; c.beginPath(); c.ellipse(0, 12, 17, 13, 0, 0, Math.PI * 2); c.fill();
  c.fillStyle = '#2e453a';
  if (success) { c.lineWidth = 2; c.strokeStyle = '#2e453a'; for (const ex of [-9, 9]) { c.beginPath(); c.arc(ex, -2, 4, Math.PI, 0); c.stroke(); } }
  else { c.beginPath(); c.ellipse(-9, -3, 3, 4, 0, 0, Math.PI * 2); c.ellipse(9, -3, 3, 4, 0, 0, Math.PI * 2); c.fill(); c.fillStyle = '#fffdeb'; c.fillRect(-10, -5, 1.5, 1.5); c.fillRect(8, -5, 1.5, 1.5); }
  c.strokeStyle = '#547460'; c.lineWidth = 1.7; c.beginPath(); c.arc(0, 5, 4, 0, Math.PI); c.stroke();
  c.fillStyle = '#cfb39a'; c.globalAlpha = .6; c.beginPath(); c.ellipse(-17, 5, 4, 2, 0, 0, Math.PI * 2); c.ellipse(17, 5, 4, 2, 0, 0, Math.PI * 2); c.fill(); c.globalAlpha = 1;
  c.fillStyle = '#73997b'; c.beginPath(); c.ellipse(-14, 25, 7, 4, 0, 0, Math.PI * 2); c.ellipse(14, 25, 7, 4, 0, 0, Math.PI * 2); c.fill();
  c.restore();
}

function draw(time = 0) {
  const shown = history[frame], pos = shown.pos;
  ctx.clearRect(0, 0, 700, 700);
  const unit = 76, ox = 84, oy = 70;
  ctx.fillStyle = '#bbc69c'; ctx.beginPath(); ctx.roundRect(ox - 8, oy + 13, unit * 7 + 16, unit * 7 + 8, 15); ctx.fill();
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const index = y * SIZE + x, terrain = garden.cells[index], px = ox + x * unit, py = oy + y * unit;
    ctx.fillStyle = (x + y) % 2 ? '#d9e2bc' : '#d4dfb5';
    ctx.beginPath(); ctx.roundRect(px + 1, py + 1, unit - 2, unit - 2, 5); ctx.fill();
    ctx.fillStyle = '#a8bf8f';
    for (let k = 0; k < 3; k++) { const sx = px + 12 + ((index * 17 + k * 23) % 50), sy = py + 16 + ((index * 11 + k * 19) % 45); ctx.fillRect(sx, sy, 2, 4); ctx.fillRect(sx + 3, sy + 2, 2, 3); }
    if (terrain === 1) {
      const border = x === 0 || x === 6 || y === 0 || y === 6;
      if (border) { ctx.fillStyle = '#94a978'; ctx.beginPath(); ctx.roundRect(px + 9, py + 13, 56, 46, 17); ctx.fill(); ctx.fillStyle = '#a2b782'; ctx.beginPath(); ctx.arc(px + 24, py + 30, 16, 0, Math.PI * 2); ctx.arc(px + 47, py + 34, 18, 0, Math.PI * 2); ctx.fill(); }
      else { ctx.fillStyle = '#7e917c35'; ctx.beginPath(); ctx.ellipse(px + 38, py + 59, 26, 8, 0, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = '#9ba598'; ctx.beginPath(); ctx.moveTo(px + 16, py + 53); ctx.lineTo(px + 23, py + 26); ctx.lineTo(px + 44, py + 20); ctx.lineTo(px + 61, py + 42); ctx.lineTo(px + 56, py + 59); ctx.closePath(); ctx.fill(); ctx.fillStyle = '#bac0ac'; ctx.beginPath(); ctx.moveTo(px + 23, py + 26); ctx.lineTo(px + 44, py + 20); ctx.lineTo(px + 42, py + 40); ctx.lineTo(px + 16, py + 53); ctx.closePath(); ctx.fill(); }
    } else if (terrain === 2) {
      ctx.fillStyle = '#6c8d6c'; ctx.beginPath(); ctx.ellipse(px + 38, py + 47, 27, 14, 0, 0, Math.PI * 2); ctx.fill();
      for (let i = 0; i < 4; i++) { ctx.fillStyle = i % 2 ? '#957391' : '#ab87a3'; ctx.beginPath(); ctx.arc(px + 22 + i * 11, py + 34 + (i % 2) * 9, 9, 0, Math.PI * 2); ctx.fill(); }
    }
    if (index === garden.goal) {
      ctx.fillStyle = '#c6756825'; ctx.beginPath(); ctx.ellipse(px + 38, py + 55, 17, 5, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#cf776e'; ctx.beginPath(); ctx.moveTo(px + 22, py + 31); ctx.bezierCurveTo(px + 18, py + 43, px + 32, py + 57, px + 38, py + 57); ctx.bezierCurveTo(px + 44, py + 55, px + 60, py + 39, px + 52, py + 30); ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#6e945f'; ctx.beginPath(); ctx.moveTo(px + 22, py + 30); ctx.lineTo(px + 34, py + 24); ctx.lineTo(px + 38, py + 32); ctx.lineTo(px + 45, py + 23); ctx.lineTo(px + 54, py + 29); ctx.lineTo(px + 38, py + 37); ctx.closePath(); ctx.fill(); ctx.fillStyle = '#fae8b0'; for (const [dx, dy] of [[30, 39], [43, 40], [36, 49]]) ctx.fillRect(px + dx, py + dy, 2, 3);
    }
  }
  if (mode === 'watch') {
    const x = pos % SIZE, y = Math.floor(pos / SIZE);
    ctx.strokeStyle = '#597a6160'; ctx.setLineDash([6, 5]); ctx.lineWidth = 2;
    ctx.strokeRect(ox + (x - 1) * unit + 3, oy + (y - 1) * unit + 3, unit * 3 - 6, unit * 3 - 6); ctx.setLineDash([]);
  }
  // Recorded steps remain visible behind Pip.
  for (let i = 1; i <= frame; i++) { const p = history[i - 1].pos; ctx.fillStyle = '#658a6440'; ctx.beginPath(); ctx.arc(ox + (p % SIZE + .5) * unit, oy + (Math.floor(p / SIZE) + .5) * unit, 5, 0, Math.PI * 2); ctx.fill(); }
  critter(ctx, ox + (pos % SIZE + .5) * unit, oy + (Math.floor(pos / SIZE) + .5) * unit, .83, running ? Math.sin(time / 140) * 3 : 0, shown.outcome === 'success');
  if (shown.outcome !== 'running') { ctx.fillStyle = '#fcfbefd9'; ctx.beginPath(); ctx.roundRect(187, 621, 326, 40, 20); ctx.fill(); ctx.fillStyle = '#48664e'; ctx.font = '15px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(shown.outcome === 'success' ? 'A strawberry, and a little more confidence.' : shown.outcome === 'hazard' ? 'Ouch. A bramble-shaped lesson.' : 'Sometimes a small brain gets stuck.', 350, 646); }
  portraitCtx.clearRect(0, 0, 140, 140); critter(portraitCtx, 70, 72, 1.65, 0, shown.outcome === 'success');
}

function stop() { running = false; if (animationTimer) clearInterval(animationTimer); animationTimer = undefined; runButton.textContent = '▶ Let Pip explore'; }
function reset() {
  stop(); world = new World(cloneGarden(garden)); history = [{ pos: world.pos, steps: 0, outcome: 'running' }];
  frame = 0; episodeActions = []; demonstration = []; el('teach').classList.add('hidden'); el('teach-note').textContent = ''; update();
}
function update() {
  const shown = history[frame];
  el('steps').innerHTML = `${String(shown.steps).padStart(2, '0')} <small>/ ${LIMIT}</small>`;
  el('outcome').textContent = shown.outcome === 'running' ? (shown.steps ? 'Exploring' : 'Ready') : ({ success: 'Found it!', hazard: 'Bramble!', timeout: 'Got stuck' }[shown.outcome] ?? shown.outcome);
  el('pip-mood').textContent = shown.outcome === 'success' ? 'A very good berry day.' : shown.outcome === 'hazard' ? 'Still figuring things out.' : 'Small paws. Big ambitions.';
  el('garden-id').textContent = `GARDEN #${garden.seed}`;
  el('brain-note').textContent = notes[brain.value];
  timeline.max = String(history.length - 1); timeline.value = String(frame); el('frame-label').textContent = `${frame} / ${history.length - 1}`;
  el('journal-message').textContent = frame ? `${ACTIONS[episodeActions[frame - 1]]}. ${shown.outcome === 'running' ? 'One step at a time.' : shown.outcome === 'success' ? 'The berry is yours.' : 'This is where things got interesting.'}` : 'Every little step leaves a trace.';
  if (policies[brain.value]) {
    const current = new World(cloneGarden(garden)); current.pos = shown.pos;
    const ps = probabilities(policies[brain.value], current.obs()), chosen = ps.indexOf(Math.max(...ps));
    el('probabilities').innerHTML = ps.map((p, i) => `<div class="prob-column ${i === chosen ? 'selected' : ''}"><strong>${Math.round(p * 100)}%</strong><div class="bar-track"><div class="bar-fill" style="height:${p * 100}%"></div></div>${ACTIONS[i]}</div>`).join('');
  }
  draw();
}
function act(action: number) {
  if (world.done || frame !== history.length - 1) return;
  if (mode === 'play') demonstration.push({ observation: world.obs(), action });
  world.step(action); episodeActions.push(action); history.push({ pos: world.pos, steps: world.steps, outcome: world.outcome }); frame = history.length - 1;
  if (world.done) { stop(); if (mode === 'play' && world.outcome === 'success') el('teach').classList.remove('hidden'); }
  update();
}
function setMode(next: string) {
  stop(); mode = next;
  document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(b => { b.classList.toggle('active', b.dataset.mode === mode); b.setAttribute('aria-pressed', String(b.dataset.mode === mode)); });
  el('play-controls').classList.toggle('hidden', mode !== 'play'); el('edit-controls').classList.toggle('hidden', mode !== 'edit');
  runButton.disabled = mode !== 'watch' || !policies[brain.value];
  reset(); canvas.focus({ preventScroll: true });
}
function paint(index: number) {
  if (mode !== 'edit' || index === garden.start || index < 0 || index >= 49) return;
  const x = index % SIZE, y = Math.floor(index / SIZE);
  if (x === 0 || x === 6 || y === 0 || y === 6) return;
  if (tile === 3) { garden.goal = index; garden.cells[index] = 0; }
  else if (index !== garden.goal) garden.cells[index] = tile;
  const valid = route(garden.cells, garden.start, garden.goal) !== null;
  el('map-warning').textContent = valid ? 'A safe route exists. Can Pip find it?' : 'No safe route exists. Try clearing a tile.';
  reset();
}

document.querySelectorAll<HTMLButtonElement>('[data-mode]').forEach(b => b.onclick = () => setMode(b.dataset.mode!));
document.querySelectorAll<HTMLButtonElement>('[data-action]').forEach(b => b.onclick = () => { if (mode === 'play') act(Number(b.dataset.action)); });
document.querySelectorAll<HTMLButtonElement>('[data-tile]').forEach(b => b.onclick = () => { tile = Number(b.dataset.tile); document.querySelectorAll('[data-tile]').forEach(t => t.classList.toggle('active', t === b)); });
el('reset').onclick = reset;
brain.onchange = () => { reset(); runButton.disabled = mode !== 'watch' || !policies[brain.value]; };
runButton.onclick = () => {
  if (!policies[brain.value] || mode !== 'watch') return;
  if (running) { stop(); return; }
  if (world.done || frame !== history.length - 1) reset();
  running = true; runButton.textContent = 'Ⅱ Pause exploring';
  animationTimer = setInterval(() => { const ps = probabilities(policies[brain.value], world.obs()); act(ps.indexOf(Math.max(...ps))); }, 350);
};
el('new-garden').onclick = () => {
  const input = el<HTMLInputElement>('seed'); let seed = Number(input.value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) { input.value = String(garden.seed); return; }
  if (seed === garden.seed) { seed = (seed + 1) >>> 0; input.value = String(seed); }
  garden = generate(seed, el<HTMLInputElement>('shifted').checked); el('map-warning').textContent = ''; reset();
};
el<HTMLInputElement>('shifted').onchange = () => { garden = generate(Number(el<HTMLInputElement>('seed').value) >>> 0, el<HTMLInputElement>('shifted').checked); reset(); };
timeline.oninput = () => { stop(); frame = Number(timeline.value); update(); };
canvas.onclick = event => { const rect = canvas.getBoundingClientRect(); const x = Math.floor(((event.clientX - rect.left) * 700 / rect.width - 84) / 76), y = Math.floor(((event.clientY - rect.top) * 700 / rect.height - 70) / 76); if (x >= 0 && x < 7 && y >= 0 && y < 7) paint(y * 7 + x); };
el('paint-selected').onclick = () => paint(Number(el<HTMLSelectElement>('tile-index').value));
el('tile-index').innerHTML = Array.from({ length: 25 }, (_, i) => { const x = i % 5 + 1, y = Math.floor(i / 5) + 1; return `<option value="${y * 7 + x}">Tile ${x}, ${y}</option>`; }).join('');
document.addEventListener('keydown', event => {
  if (mode !== 'play' || ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes((event.target as HTMLElement).tagName)) return;
  const action = ({ ArrowUp: 0, w: 0, ArrowRight: 1, d: 1, ArrowDown: 2, s: 2, ArrowLeft: 3, a: 3 } as Record<string, number>)[event.key];
  if (action !== undefined) { event.preventDefault(); act(action); }
});
el('teach').onclick = async () => {
  if (mode !== 'play' || world.outcome !== 'success' || !demonstration.length || !policies[brain.value]) return;
  const button = el<HTMLButtonElement>('teach'); button.disabled = true; button.textContent = 'Learning from your footsteps…';
  lessons = [...lessons, ...demonstration].slice(-2400);
  try {
    policies.personal = await teach(policies[brain.value], lessons);
    if (!Array.from(brain.options).some(o => o.value === 'personal')) brain.add(new Option('Your lessons (session only)', 'personal'));
    brain.value = 'personal'; demonstration = [];
    el('teach-note').textContent = `Neural weights updated from ${lessons.length} demonstrated actions. Try a new garden to test transfer.`;
    button.classList.add('hidden'); update();
  } finally { button.disabled = false; button.textContent = 'Teach Pip this route ✳'; }
};
el('download').onclick = () => {
  const record = { version: 1, garden, policy: brain.value, policy_weights: policies[brain.value], actions: episodeActions, frames: history, teaching_actions: lessons.length };
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = `critterlab-${garden.seed}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};

interface Experiment { steps: number; summary: Record<string, Record<string, { mean: number; std: number }>>; disjoint_audit: { splits: Record<string, { episodes: number; summary: Record<string, { mean: number; std: number }> }> }; curves: { seed: number; method: string; points: { steps: number; success_rate: number }[] }[] }
function showExperiment(data: Experiment) {
  const names: Record<string, string> = { 'bc-ppo': 'Demonstrations + rewards', ppo: 'Rewards only', bc: 'Demonstrations only', untrained: 'Untrained' };
  const renderResults = () => {
    const choice = el<HTMLSelectElement>('evaluation-set').value, audit = choice.startsWith('audit-'), split = choice.replace('audit-', '');
    const count = audit ? data.disjoint_audit.splits[split].episodes : 256;
    el('results').innerHTML = ['bc-ppo', 'ppo', 'bc', 'untrained'].map(method => { const score = audit ? data.disjoint_audit.splits[split].summary[method] : data.summary[method][split]; return `<div class="result-row"><span>${names[method]}</span><span><strong>${(score.mean * 100).toFixed(1)}%</strong> <small>± ${(score.std * 100).toFixed(1)}</small></span></div>`; }).join('');
    el('result-caption').textContent = `Mean ± standard deviation across 3 seeds. ${count} gardens per seed. ${audit ? 'Post-hoc common subset; all exact training-world overlaps excluded.' : 'Seed-held-out split; some layouts recur. See the disjoint-world audit.'}`;
  };
  el<HTMLSelectElement>('evaluation-set').onchange = renderResults;
  renderResults();
  const x = (step: number) => 45 + step / data.steps * 555, y = (success: number) => 232 - success * 205;
  let svg = '';
  for (const value of [0, .25, .5, .75, 1]) svg += `<line x1="45" y1="${y(value)}" x2="600" y2="${y(value)}" stroke="#e3e5d8"/><text x="33" y="${y(value) + 4}" text-anchor="end" fill="#8a927f" font-size="10">${value * 100}%</text>`;
  for (const value of [0, 16384, 32768, 49152, 65536]) svg += `<text x="${x(value)}" y="255" text-anchor="middle" fill="#8a927f" font-size="10">${value ? Math.round(value / 1000) + 'k' : '0'}</text>`;
  svg += '<text x="320" y="278" text-anchor="middle" fill="#8a927f" font-size="10">Reward-learning environment steps</text>';
  for (const [method, color] of [['ppo', '#c88552'], ['bc-ppo', '#457858']]) {
    const curves = data.curves.filter(c => c.method === method);
    for (const curve of curves) svg += `<polyline points="${curve.points.map(p => `${x(p.steps)},${y(p.success_rate)}`).join(' ')}" stroke="${color}" opacity=".18" stroke-width="1.5" fill="none"/>`;
    const mean = curves[0].points.map((p, index) => `${x(p.steps)},${y(curves.reduce((sum, c) => sum + c.points[index].success_rate, 0) / curves.length)}`).join(' ');
    svg += `<polyline points="${mean}" stroke="${color}" stroke-width="3" fill="none" stroke-linejoin="round"/>`;
  }
  el('learning-chart').innerHTML = svg;
}

runButton.disabled = true; update();
async function load() {
  try {
    const names = ['untrained', 'bc', 'ppo', 'bc-ppo'];
    const fetched = await Promise.all(names.map(async name => { const r = await fetch(`${import.meta.env.BASE_URL}models/${name}.json`); if (!r.ok) throw new Error('Model unavailable'); return r.json() as Promise<Policy>; }));
    policies = Object.fromEntries(names.map((name, i) => [name, fetched[i]])); runButton.disabled = false; update();
    const response = await fetch(`${import.meta.env.BASE_URL}data/experiment.json`); if (!response.ok) throw new Error('Experiment unavailable'); showExperiment(await response.json());
  } catch { el('brain-note').textContent = 'The model files could not load. Refresh or run the local build from GitHub.'; }
}
void load();
if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
  function animate(time: number) { if (running) draw(time); requestAnimationFrame(animate); }
  requestAnimationFrame(animate);
}
