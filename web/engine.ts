export const SIZE = 7;
export const LIMIT = 48;
export const DELTAS = [[0, -1], [1, 0], [0, 1], [-1, 0]] as const;
export const ACTIONS = ['North', 'East', 'South', 'West'];
export interface Garden { cells: number[]; start: number; goal: number; seed: number }
export interface Policy { label: string; seed: number; environment_steps: number; architecture: number[]; activation: string; layers: { weight: number[][]; bias: number[] }[] }

export class RNG {
  state: number;
  constructor(seed: number) { this.state = seed >>> 0; }
  next() { this.state = (Math.imul(1664525, this.state) + 1013904223) >>> 0; return this.state / 4294967296; }
  integer(n: number) { return Math.floor(this.next() * n); }
}

export function route(cells: number[], start: number, goal: number): number[] | null {
  const queue: [number, number[]][] = [[start, []]], visited = new Set([start]);
  for (let head = 0; head < queue.length; head++) {
    const [pos, actions] = queue[head];
    if (pos === goal) return actions;
    const x = pos % SIZE, y = Math.floor(pos / SIZE);
    DELTAS.forEach(([dx, dy], action) => {
      const nx = x + dx, ny = y + dy, next = ny * SIZE + nx;
      if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && cells[next] === 0 && !visited.has(next)) {
        visited.add(next); queue.push([next, [...actions, action]]);
      }
    });
  }
  return null;
}

export function generate(seed: number, shifted = false): Garden {
  const rng = new RNG(seed);
  for (let attempt = 0; attempt < 100; attempt++) {
    const cells: number[] = [];
    for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
      const r = rng.next();
      cells.push(x === 0 || x === SIZE - 1 || y === 0 || y === SIZE - 1 ? 1 : r < (shifted ? .16 : .08) ? 1 : r < (shifted ? .26 : .14) ? 2 : 0);
    }
    const start = 1 + rng.integer(5) + (1 + rng.integer(5)) * SIZE;
    const goal = 1 + rng.integer(5) + (1 + rng.integer(5)) * SIZE;
    cells[start] = cells[goal] = 0;
    const path = route(cells, start, goal);
    if (path && path.length >= 4) return { cells, start, goal, seed };
  }
  throw new Error('Could not generate garden');
}

export function observe(cells: number[], pos: number, goal: number) {
  const x = pos % SIZE, y = Math.floor(pos / SIZE);
  const result = [(goal % SIZE - x) / SIZE, (Math.floor(goal / SIZE) - y) / SIZE];
  for (const tile of [1, 2]) for (const dy of [-1, 0, 1]) for (const dx of [-1, 0, 1]) {
    const nx = x + dx, ny = y + dy;
    const value = nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE ? cells[ny * SIZE + nx] : 1;
    result.push(Number(value === tile));
  }
  return result;
}

export function probabilities(policy: Policy, observation: number[]) {
  let input = observation;
  policy.layers.forEach((layer, index) => {
    const out = layer.weight.map((row, i) => row.reduce((sum, weight, j) => sum + weight * input[j], layer.bias[i]));
    input = index === policy.layers.length - 1 ? out : out.map(Math.tanh);
  });
  const peak = Math.max(...input), exps = input.map(x => Math.exp(x - peak)), total = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / total);
}

export class World {
  pos: number; steps = 0; done = false; outcome = 'running'; reward = 0;
  constructor(public garden: Garden) { this.pos = garden.start; }
  obs() { return observe(this.garden.cells, this.pos, this.garden.goal); }
  step(action: number) {
    if (this.done) throw new Error('Episode is already over');
    if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error('Action must be 0..3');
    const x = this.pos % SIZE, y = Math.floor(this.pos / SIZE), [dx, dy] = DELTAS[action];
    const nx = x + dx, ny = y + dy, next = ny * SIZE + nx;
    const gx = this.garden.goal % SIZE, gy = Math.floor(this.garden.goal / SIZE);
    const before = Math.abs(gx - x) + Math.abs(gy - y);
    this.steps++; this.reward = -.02;
    if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE || this.garden.cells[next] === 1) this.reward = -.08;
    else {
      this.pos = next;
      this.reward += .1 * (before - Math.abs(gx - nx) - Math.abs(gy - ny));
      if (this.garden.cells[next] === 2) { this.reward = -1; this.done = true; this.outcome = 'hazard'; }
      else if (next === this.garden.goal) { this.reward = 1; this.done = true; this.outcome = 'success'; }
    }
    if (this.steps >= LIMIT && !this.done) { this.done = true; this.outcome = 'timeout'; }
    return { observation: this.obs(), reward: this.reward, done: this.done };
  }
}

export function cloneGarden(garden: Garden): Garden { return { ...garden, cells: [...garden.cells] }; }
