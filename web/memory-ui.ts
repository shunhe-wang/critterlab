import { ACTIONS, type Policy } from './engine';
import { MemoryWorld, memoryDecision, type RecurrentPolicy, type MemoryTask } from './memory';

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id)! as T;
interface Score { mean: number; std: number; values: number[] }
interface Results { summary: Record<string, Record<string, Score>>; controls: Record<string, Score>; training_transitions: number }
type Painter = (ctx: CanvasRenderingContext2D, x: number, y: number, scale: number, bounce?: number, success?: boolean) => void;

export function mountMemoryLab(drawPet: Painter, onFinish: (task: MemoryTask, outcome: string, method: string) => void, onStart: () => void) {
  const canvas = el<HTMLCanvasElement>('memory-canvas'), ctx = canvas.getContext('2d')!;
  const brain = el<HTMLSelectElement>('memory-brain'), run = el<HTMLButtonElement>('memory-run');
  let world = new MemoryWorld({ column: 3, row: 1, delay: 8, cue: 0 }), hidden: number[] | undefined;
  let mode = 'watch', timer: ReturnType<typeof setInterval> | undefined, actions: number[] = [];
  let policies: Record<string, Policy | RecurrentPolicy> = {}, result: Results | undefined;
  const notes: Record<string, string> = {
    gru: 'A 5,508-parameter GRU trained through whole sequences. Its hidden state carries earlier observations.',
    reactive: 'A 5,892-parameter feed-forward actor, trained on the same 36 sequences. Each decision sees only the current input.',
    current: 'The original garden’s BC + PPO checkpoint, unchanged. It has no cue inputs and was not trained on this task.'
  };
  function stop() { if (timer) clearInterval(timer); timer = undefined; run.textContent = mode === 'watch' ? '▶ Follow the clue' : '▶ Start my turn'; }
  function fresh() {
    stop(); hidden = undefined; actions = [];
    world = new MemoryWorld({ column: Number(el<HTMLSelectElement>('memory-column').value), row: 1,
      delay: Number(el<HTMLSelectElement>('memory-delay').value), cue: crypto.getRandomValues(new Uint32Array(1))[0] % 2 });
    render();
  }
  function controls() {
    run.disabled = !policies[brain.value]; el<HTMLButtonElement>('memory-step').disabled = !policies[brain.value] || mode === 'human';
    el('memory-dpad').classList.toggle('hidden', mode !== 'human');
    for (const name of ['watch', 'human']) { el(`memory-${name}`).classList.toggle('active', mode === name); el(`memory-${name}`).setAttribute('aria-pressed', String(mode === name)); }
    el('memory-brain-note').textContent = policies[brain.value] ? notes[brain.value] : 'Loading the memory checkpoints…';
  }
  function decision() { return memoryDecision(policies[brain.value], world, el<HTMLInputElement>('memory-forget').checked ? undefined : hidden, el<HTMLInputElement>('memory-hide').checked); }
  function act(action?: number) {
    if (world.done || !policies[brain.value]) return;
    if (world.steps === 0) onStart();
    const step = decision(); hidden = step.hidden;
    const chosen = action ?? step.probabilities.indexOf(Math.max(...step.probabilities));
    world.step(chosen); actions.push(chosen);
    if (world.done) { stop(); onFinish(world.task, world.outcome, mode === 'human' ? 'your hands' : brain.value); }
    render();
  }
  function render() {
    controls();
    const cue = world.steps === 0, waiting = world.steps < world.task.delay;
    el('memory-cue').textContent = cue ? `${world.task.cue === 0 ? 'BLUE → LEFT' : 'GOLD → RIGHT'}` : 'CLUE HAS FADED';
    el('memory-cue').className = `pill ${cue ? world.task.cue === 0 ? 'cue-blue' : 'cue-gold' : ''}`;
    el('memory-status').textContent = world.done ? world.outcome === 'success' ? 'The right berry. A clue remembered.' : world.outcome === 'wrong-berry' ? 'A good berry… but the clue meant the other one.' : 'This brain got stuck before choosing a berry.' : cue ? 'Remember the flower before the gate opens.' : waiting ? `Gate opens in ${world.task.delay - world.steps} ticks. The clue is gone.` : 'Gate open. Which berry did the clue mean?';
    el('memory-ticks').textContent = `Tick ${world.steps} / 96`;
    el('memory-hint').textContent = mode === 'human' ? 'Start my turn runs the gate countdown. Then focus this garden and use arrows, WASD, or the touch buttons. Blue meant left; gold meant right.' : 'The gate prevents encoding the cue in your starting position. Decisions and failures come from the actual network.';
    const state = el('memory-state'); state.replaceChildren();
    if (brain.value === 'gru') {
      for (const [i, value] of (hidden ?? Array<number>(32).fill(0)).entries()) {
        const unit = document.createElement('span'); unit.style.background = value >= 0 ? `rgba(84,124,153,${.12 + Math.abs(value) * .85})` : `rgba(185,140,90,${.12 + Math.abs(value) * .85})`;
        unit.title = `Unit ${i + 1}: ${value.toFixed(4)}`; unit.setAttribute('aria-label', unit.title); state.append(unit);
      }
    } else state.textContent = 'This actor has no recurrent state.';
    if (policies[brain.value] && !world.done) {
      const ps = decision().probabilities;
      el('memory-probs').textContent = ps.map((p, i) => `${ACTIONS[i]} ${Math.round(p * 100)}%`).join(' · ');
    } else el('memory-probs').textContent = world.done ? 'Trial complete. Start a new clue to compare another brain.' : '';
    ctx.clearRect(0, 0, 700, 540);
    const unit = 64, ox = 126, oy = 25;
    for (let i = 0; i < 49; i++) {
      const x = ox + i % 7 * unit, y = oy + Math.floor(i / 7) * unit;
      ctx.fillStyle = world.cells[i] ? '#b4c6a0' : i % 2 ? '#e0e8ca' : '#d9e3c0';
      ctx.beginPath(); ctx.roundRect(x + 1, y + 1, 62, 62, world.cells[i] ? 16 : 5); ctx.fill();
      if (world.cells[i]) { ctx.fillStyle = '#a6bd90'; ctx.beginPath(); ctx.arc(x + 25, y + 25, 15, 0, Math.PI * 2); ctx.arc(x + 40, y + 38, 15, 0, Math.PI * 2); ctx.fill(); }
    }
    for (const x of [1, 5]) {
      const px = ox + (x + .5) * unit, py = oy + (world.task.row + .5) * unit;
      ctx.fillStyle = '#c97873'; ctx.beginPath(); ctx.ellipse(px, py + 4, 14, 18, .15, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#6c935d'; ctx.beginPath(); ctx.ellipse(px, py - 11, 12, 5, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#f3e1b3'; for (const dx of [-6, 3, -1]) ctx.fillRect(px + dx, py + dx + 5, 2, 3);
    }
    if (waiting) {
      ctx.strokeStyle = '#c09565'; ctx.lineWidth = 7;
      const gy = oy + 5 * unit; ctx.beginPath(); ctx.moveTo(ox + world.task.column * unit + 5, gy); ctx.lineTo(ox + (world.task.column + 1) * unit - 5, gy); ctx.stroke();
    }
    drawPet(ctx, ox + (world.pos % 7 + .5) * unit, oy + (Math.floor(world.pos / 7) + .5) * unit, .8, 0, world.outcome === 'success');
    if (cue) {
      const x = ox + (world.task.column + .5) * unit, y = oy + 5 * unit - 49;
      ctx.fillStyle = '#fbfaf3'; ctx.beginPath(); ctx.roundRect(x - 70, y - 25, 140, 39, 14); ctx.fill();
      ctx.fillStyle = world.task.cue === 0 ? '#5b88ae' : '#b99042'; ctx.font = '20px sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(world.task.cue === 0 ? '✿  ←' : '✿  →', x, y + 2);
    }
    ctx.font = '11px sans-serif'; ctx.fillStyle = '#6b7f62'; ctx.textAlign = 'center';
    ctx.fillText('BLUE FLOWER → LEFT BERRY     ·     GOLD FLOWER → RIGHT BERRY', 350, 510);
  }
  function start() {
    if (!policies[brain.value]) return;
    if (timer) { stop(); return; }
    if (world.done) fresh();
    onStart(); run.textContent = 'Ⅱ Pause trial';
    timer = setInterval(() => {
      if (mode === 'human' && world.steps >= world.task.delay) { stop(); render(); canvas.focus({preventScroll:true}); return; }
      act(mode === 'human' ? 0 : undefined);
    }, 400);
  }
  run.onclick = start; el('memory-new').onclick = fresh;
  el('memory-step').onclick = () => { stop(); act(); };
  for (const id of ['memory-brain', 'memory-delay', 'memory-column', 'memory-hide', 'memory-forget']) el(id).onchange = fresh;
  for (const next of ['watch', 'human']) el(`memory-${next}`).onclick = () => { mode = next; fresh(); canvas.focus({preventScroll:true}); };
  document.querySelectorAll<HTMLButtonElement>('[data-memory-action]').forEach(button => button.onclick = () => { if (mode === 'human' && !timer) act(Number(button.dataset.memoryAction)); });
  el('memory-lab').addEventListener('keydown', event => {
    if (mode !== 'human' || timer || ['INPUT','SELECT','TEXTAREA','BUTTON'].includes((event.target as HTMLElement).tagName)) return;
    const action = ({ArrowUp:0,w:0,ArrowRight:1,d:1,ArrowDown:2,s:2,ArrowLeft:3,a:3} as Record<string,number>)[event.key];
    if (action !== undefined) { event.preventDefault(); act(action); }
  });
  el('memory-download').onclick = () => {
    const record = { version:1, experiment:'m1', task:world.task, brain:brain.value, mode, hide_cue:el<HTMLInputElement>('memory-hide').checked,
      reset_each_tick:el<HTMLInputElement>('memory-forget').checked, actions, outcome:world.outcome, checkpoint:`models/${brain.value === 'current' ? 'bc-ppo' : 'm1-' + brain.value}.json` };
    const url = URL.createObjectURL(new Blob([JSON.stringify(record,null,2)],{type:'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = 'critterlab-memory-trial.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  function showScores() {
    if (!result) return;
    const format = (score: Score) => `${(score.mean * 100).toFixed(1)}% ± ${(score.std * 100).toFixed(1)} pp`;
    const split = el<HTMLSelectElement>('memory-split').value;
    el('memory-scores').replaceChildren(...[['gru','Recurrent GRU'],['reactive','Retrained reactive'],['current','Frozen current brain']].map(([key,name])=> {
      const row = document.createElement('div'); row.className = 'result-row'; const label = document.createElement('span'), score = document.createElement('strong');
      label.textContent = name; score.textContent = format(result!.summary[key][split]); row.append(label,score); return row;
    }));
    el('memory-control-scores').replaceChildren(...[['no-cue','Erase cue'],['flip-cue','Swap cue'],['reset','Forget each tick']].map(([key,name])=> {
      const row = document.createElement('div'); row.className = 'result-row'; const label = document.createElement('span'), score = document.createElement('strong');
      label.textContent = name; score.textContent = format(result!.controls[key]); row.append(label,score); return row;
    }));
    el('memory-result-note').textContent = `Mean ± sample standard deviation in percentage points across 3 model seeds. ${split === 'standard' ? 18 : 12} episodes per seed, paired by clue. Seed variation matters; the demo always uses final seed 0.`;
  }
  el('memory-split').onchange = showScores;
  document.addEventListener('visibilitychange',()=> {if(document.hidden)stop();});
  fresh();
  void (async()=> {
    try {
      const names = ['gru','reactive','current'];
      const loaded = await Promise.all(names.map(async name=> {const response=await fetch(`${import.meta.env.BASE_URL}models/${name === 'current' ? 'bc-ppo' : 'm1-' + name}.json`); if(!response.ok)throw new Error('Missing checkpoint'); return response.json();}));
      policies=Object.fromEntries(names.map((name,i)=>[name,loaded[i]])); render();
    } catch {el('memory-brain-note').textContent='Memory checkpoints could not load. Refresh to try again.';}
    try { const response=await fetch(`${import.meta.env.BASE_URL}data/memory.json`); if(!response.ok)throw new Error('Missing results'); result=await response.json(); showScores(); }
    catch {el('memory-result-note').textContent='Measured results could not load. The full report is linked below.';}
  })();
  return { pause: stop };
}
