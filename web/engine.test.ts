import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { World, generate, probabilities, type Policy } from './engine';
import { teach } from './learn';

function handWorkedWorld() {
  const cells: number[] = Array.from({ length: 49 }, (_, i) => i % 7 === 0 || i % 7 === 6 || i < 7 || i >= 42 ? 1 : 0);
  cells[15] = 2;
  return new World({ cells, start: 8, goal: 10, seed: 0 });
}

test('browser follows the hand-worked arrival, collision and hazard contract', () => {
  const w = handWorkedWorld();
  assert.equal(w.step(0).reward, -.08); assert.equal(w.pos, 8);
  assert.ok(Math.abs(w.step(1).reward - .08) < 1e-12);
  assert.equal(w.step(1).reward, 1); assert.equal(w.outcome, 'success');
  assert.throws(() => w.step(0), /already over/);
  const hazard = handWorkedWorld(); hazard.step(2); assert.equal(hazard.outcome, 'hazard');
});

test('browser generation, observations and transitions agree with the Python reference', () => {
  const vectors = JSON.parse(readFileSync('tests/fixtures/world-parity.json', 'utf8'));
  for (const vector of vectors) {
    const garden = generate(vector.seed, vector.shifted);
    assert.deepEqual(garden, vector.garden);
    const world = new World(garden);
    world.obs().forEach((v, i) => assert.ok(Math.abs(v - vector.observation[i]) < 1e-6));
    for (const transition of vector.transitions) {
      const step = world.step(transition.action);
      assert.equal(world.pos, transition.pos); assert.equal(world.outcome, transition.outcome);
      assert.ok(Math.abs(step.reward - transition.reward) < 1e-6);
    }
  }
});

test('exported neural probabilities agree with PyTorch across all four policies', () => {
  const vectors = JSON.parse(readFileSync('tests/fixtures/policy-parity.json', 'utf8'));
  for (const vector of vectors) {
    const policy: Policy = JSON.parse(readFileSync(`public/models/${vector.policy}.json`, 'utf8'));
    const ps = probabilities(policy, vector.observation);
    assert.ok(Math.abs(ps.reduce((a, b) => a + b) - 1) < 1e-12);
    ps.forEach((p, i) => assert.ok(Math.abs(p - vector.probabilities[i]) < 1e-6));
  }
});

test('human teaching changes weights toward a demonstrated action without modifying the source', async () => {
  const policy: Policy = JSON.parse(readFileSync('public/models/untrained.json', 'utf8'));
  const observation = Array(20).fill(0), source = JSON.stringify(policy);
  const before = probabilities(policy, observation)[1];
  const trained = await teach(policy, [{ observation, action: 1 }], 20);
  assert.ok(probabilities(trained, observation)[1] > before);
  assert.equal(JSON.stringify(policy), source);
  assert.notEqual(JSON.stringify(trained.layers), JSON.stringify(policy.layers));
});
