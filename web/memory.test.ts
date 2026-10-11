import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MemoryWorld, recurrentStep, memoryDecision, type RecurrentPolicy, type MemoryTask } from './memory';
import type { Policy } from './engine';

test('browser follows the hand-worked cue, gate and wrong-berry contract', () => {
  const w = new MemoryWorld({ column: 3, row: 2, delay: 2, cue: 1 });
  assert.equal(w.pos, 38); assert.deepEqual(w.obs().slice(-2), [0, 1]);
  w.step(3); w.step(1); assert.equal(w.pos, 38); assert.deepEqual(w.obs().slice(-2), [0, 0]);
  for (const action of [0, 0, 0, 3, 3]) w.step(action);
  assert.equal(w.steps, 7); assert.equal(w.pos, 15); assert.equal(w.outcome, 'wrong-berry');
});
test('browser worlds and carried GRU state agree with independent PyTorch sequence fixtures', () => {
  const vectors = JSON.parse(readFileSync('tests/fixtures/memory-parity.json', 'utf8'));
  const policy: RecurrentPolicy = JSON.parse(readFileSync('public/models/m1-gru.json', 'utf8'));
  for (const vector of vectors) {
    const world = new MemoryWorld(vector.task); let hidden: number[] | undefined;
    assert.deepEqual(world.cells, vector.cells);
    for (const frame of vector.frames) {
      assert.equal(world.pos, frame.pos);
      world.obs().forEach((x, i) => assert.ok(Math.abs(x - frame.observation[i]) < 1e-7));
      const step = recurrentStep(policy, world.obs(), hidden); hidden = step.hidden;
      hidden.forEach((x, i) => assert.ok(Math.abs(x - frame.hidden[i]) < 2e-6));
      step.probabilities.forEach((x, i) => assert.ok(Math.abs(x - frame.probabilities[i]) < 2e-6));
      world.step(frame.action);
    }
    assert.equal(world.outcome, vector.outcome);
  }
});
test('browser runs produce the published seed-0 outcomes, including memory controls', () => {
  const result = JSON.parse(readFileSync('results/memory-m1.json', 'utf8'));
  for (const method of ['gru', 'reactive', 'current']) {
    const run = result.runs.find((r: {seed: number; method: string}) => r.seed === 0 && r.method === method);
    const policy: Policy | RecurrentPolicy = JSON.parse(readFileSync(`public/models/${method === 'current' ? 'bc-ppo' : 'm1-' + method}.json`, 'utf8'));
    const groups = [...Object.values(run.evaluation).map(value => ({condition:'normal', value})), ...Object.entries(run.controls ?? {}).filter(([name]) => name !== 'flip-cue').map(([condition, value]) => ({condition, value}))];
    for (const {condition, value} of groups) for (const episode of (value as {episodes: {task: MemoryTask; actions: number[]; outcome: string}[]}).episodes) {
      const world = new MemoryWorld(episode.task); let hidden: number[] | undefined;
      for (const action of episode.actions) {
        const step = memoryDecision(policy, world, condition === 'reset' ? undefined : hidden, condition === 'no-cue');
        hidden = step.hidden;
        assert.equal(step.probabilities.indexOf(Math.max(...step.probabilities)), action);
        world.step(action);
      }
      assert.equal(world.outcome, episode.outcome);
    }
  }
});
