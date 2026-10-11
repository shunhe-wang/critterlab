import { DELTAS, observe, probabilities, type Policy } from './engine';

export interface MemoryTask { column: number; row: number; delay: number; cue: number }
export interface RecurrentPolicy {
  kind: 'gru'; input_size: number; hidden_size: number; seed: number;
  weight_ih: number[][]; weight_hh: number[][]; bias_ih: number[]; bias_hh: number[];
  head: { weight: number[][]; bias: number[] };
}
export class MemoryWorld {
  cells = Array<number>(49).fill(1); start: number; junction: number; pos: number;
  steps = 0; done = false; outcome = 'running';
  constructor(public task: MemoryTask) {
    if (![2, 3, 4].includes(task.column) || ![1, 2, 3].includes(task.row) || !Number.isInteger(task.delay) || task.delay < 1 || task.delay > 32 || ![0, 1].includes(task.cue)) throw new Error('Invalid memory task');
    for (let y = task.row; y <= 5; y++) this.cells[y * 7 + task.column] = 0;
    for (let x = 1; x <= 5; x++) this.cells[task.row * 7 + x] = 0;
    this.start = 35 + task.column; this.junction = task.row * 7 + task.column; this.pos = this.start;
  }
  obs(): number[] {
    return [...observe(this.cells, this.pos, this.junction), Number(this.steps === 0 && this.task.cue === 0), Number(this.steps === 0 && this.task.cue === 1)];
  }
  step(action: number) {
    if (this.done) throw new Error('Episode is already over');
    if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error('Action must be 0..3');
    if (this.steps >= this.task.delay) {
      const [dx, dy] = DELTAS[action], nx = this.pos % 7 + dx, ny = Math.floor(this.pos / 7) + dy;
      if (nx >= 0 && nx < 7 && ny >= 0 && ny < 7 && this.cells[ny * 7 + nx] === 0) this.pos = ny * 7 + nx;
      if ([this.task.row * 7 + 1, this.task.row * 7 + 5].includes(this.pos)) {
        this.done = true; this.outcome = Number(this.pos % 7 === 5) === this.task.cue ? 'success' : 'wrong-berry';
      }
    }
    this.steps++;
    if (this.steps >= 96 && !this.done) { this.done = true; this.outcome = 'timeout'; }
  }
}
const affine = (matrix: number[][], input: number[], bias: number[]) => matrix.map((row, i) => row.reduce((sum, weight, j) => sum + weight * input[j], bias[i]));
export function recurrentStep(policy: RecurrentPolicy, obs: number[], previous?: number[]) {
  const h = previous ?? Array<number>(policy.hidden_size).fill(0), size = policy.hidden_size;
  const input = affine(policy.weight_ih, obs, policy.bias_ih), state = affine(policy.weight_hh, h, policy.bias_hh);
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const hidden = h.map((value, i) => {
    const reset = sigmoid(input[i] + state[i]), update = sigmoid(input[size + i] + state[size + i]);
    const candidate = Math.tanh(input[2 * size + i] + reset * state[2 * size + i]);
    return (1 - update) * candidate + update * value;
  });
  const logits = affine(policy.head.weight, hidden, policy.head.bias), peak = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - peak)), total = exps.reduce((a, b) => a + b, 0);
  return { hidden, probabilities: exps.map(x => x / total) };
}
export function memoryDecision(policy: Policy | RecurrentPolicy, world: MemoryWorld, hidden?: number[], hideCue = false) {
  const obs = world.obs(); if (hideCue) obs.splice(20, 2, 0, 0);
  if ('kind' in policy && policy.kind === 'gru') return recurrentStep(policy, obs, hidden);
  const reactive = policy as Policy;
  return { probabilities: probabilities(reactive, obs.slice(0, reactive.architecture[0])), hidden: undefined };
}
