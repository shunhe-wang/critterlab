import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { World, probabilities, type Policy } from './engine';
import { teach } from './learn';
import { newPet, advance, care, adventure, decodeSave, intent, stage, type Save } from './pet';

// Expected values come from docs/pet-contract.md, not from the functions under test.
const HOUR = 3_600_000;
function record(): Save {
  const cells = Array.from({ length: 49 }, (_, i) => i % 7 === 0 || i % 7 === 6 || i < 7 || i >= 42 ? 1 : 0);
  return { version: 1, pet: newPet(0), garden: { cells, start: 8, goal: 10, seed: 0 }, shifted: false,
    brain: 'bc-ppo', mode: 'watch', actions: [], demonstration: [], lessons: [] };
}
test('away-time needs follow the published rates, cap absence and ignore backwards clocks', () => {
  const pet = newPet(HOUR), later = advance(pet, 4 * HOUR);
  assert.equal(later.fullness, 68); assert.equal(later.energy, 79); assert.equal(later.affection, 67);
  const away = advance(pet, 100 * HOUR);
  assert.equal(away.fullness, 20); assert.equal(away.energy, 37); assert.equal(away.affection, 46);
  assert.deepEqual(advance(pet, 0), pet); assert.equal(pet.fullness, 80);
});
test('care caps needs, has a cooldown, and cannot farm friendship points', () => {
  const fed = care(newPet(0), 'feed', 0).pet;
  assert.equal(fed.fullness, 100); assert.equal(fed.growth, 4);
  assert.deepEqual(care(fed, 'feed', 0).pet, fed);
  const again = care(fed, 'feed', 60_000).pet;
  assert.equal(again.fullness, 100); assert.equal(again.growth, 4);
  assert.equal(care(again, 'feed', 900_000).pet.growth, 8);
  assert.equal(care(newPet(0), 'cuddle', 0).pet.affection, 82);
});
test('a nap grants only time actually slept, including return after a long absence', () => {
  const sleepy = care({ ...newPet(0), energy: 40 }, 'nap', 0).pet;
  assert.equal(sleepy.energy, 40);
  assert.ok(Math.abs(advance(sleepy, 10_000).energy - (47.5 - 2 * 10_000 / HOUR)) < 1e-9);
  assert.equal(advance(sleepy, HOUR).energy, 60.5);
  const woken = { ...advance(sleepy, 10_000), sleepUntil: 0 };
  assert.ok(advance(woken, 30_000).energy < 47.5);
});
test('garden rewards use actual geometry, survive round trips and reset on the UTC day', () => {
  const save = record(), first = adventure(save.pet, save.garden, 'success', 0);
  assert.equal(first.growth, 8); assert.equal(first.energy, 82); assert.equal(first.fullness, 88);
  const restored = decodeSave(JSON.stringify({ ...save, pet: first }));
  const repeat = adventure(restored.pet, { ...save.garden, seed: 999 }, 'success', 0);
  assert.equal(repeat.growth, 8); assert.equal(repeat.adventures, 2);
  assert.equal(adventure(repeat, save.garden, 'timeout', 24 * HOUR).growth, 10);
  assert.equal(stage({ ...first, growth: 39 }), 'Hatchling');
  assert.equal(stage({ ...first, growth: 40 }), 'Scout');
  assert.equal(stage({ ...first, growth: 120 }), 'Trailkeeper');
});
test('a restored human episode and actually adapted weights retain their decisions', async () => {
  const save = record(), world = new World(save.garden);
  save.mode = 'play'; save.actions = [1, 1];
  save.demonstration = save.actions.map(action => { const observation = world.obs(); world.step(action); return { observation, action }; });
  const source: Policy = JSON.parse(readFileSync('public/models/untrained.json', 'utf8'));
  save.personal = await teach(source, save.demonstration, 5); save.brain = 'personal'; save.lessons = save.demonstration;
  const restored = decodeSave(JSON.stringify(save));
  assert.deepEqual(restored, save);
  assert.deepEqual(probabilities(restored.personal!, world.obs()), probabilities(save.personal, world.obs()));
  const resumed = new World(restored.garden); restored.actions.forEach(action => resumed.step(action));
  assert.equal(resumed.outcome, 'success'); assert.equal(resumed.steps, 2);
});
test('invalid backups fail before replacing a save: illegal replay, dimensions, text and version', () => {
  const save = record(), json = (v: unknown) => JSON.stringify(v);
  assert.throws(() => decodeSave(json({ ...save, actions: [1, 1, 1] })), /episode end/);
  assert.throws(() => decodeSave(json({ ...save, version: 2 })), /version/);
  assert.throws(() => decodeSave(json({ ...save, pet: { ...save.pet, name: '' } })), /text/);
  assert.throws(() => decodeSave(json({ ...save, brain: 'personal' })), /session/);
  assert.throws(() => decodeSave(json({ ...save, personal: { architecture: [20, 64, 64, 4], activation: 'tanh', layers: [{ weight: [[0]], bias: [0] }] } })), /dimensions/);
  assert.throws(() => decodeSave(' '.repeat(1_000_001)), /1 MB/);
  assert.deepEqual(decodeSave(json(save)), save);
});
test('small command vocabulary avoids substring matches and negated care', () => {
  assert.equal(intent('How are you?'), 'needs'); assert.equal(intent('What do you remember?'), 'memories');
  assert.equal(intent('Let’s explore'), 'explore'); assert.equal(intent('a snack please'), 'feed');
  assert.equal(intent('don’t feed me'), 'unknown'); assert.equal(intent('petroleum'), 'unknown');
  assert.equal(intent('explain quantum physics'), 'unknown');
});
