import { World, type Garden, type Policy } from './engine';
import type { Lesson } from './learn';

export const SAVE_KEY = 'critterlab.companion.v1';
export const MAX_BACKUP_BYTES = 1_000_000;
const HOUR = 3_600_000, MINUTE = 60_000;
export type Care = 'feed' | 'cuddle' | 'nap';
export interface Memory { at: number; text: string }
export interface Chat { speaker: 'you' | 'pet'; text: string }
export interface Pet {
  name: string; born: number; clock: number; fullness: number; energy: number;
  affection: number; growth: number; sleepUntil: number; adventures: number;
  lastCare: Partial<Record<Care, number>>; lastGrowth: Partial<Record<Care, number>>;
  rewarded: string[]; memories: Memory[]; chat: Chat[];
}
export interface Save {
  version: 1; pet: Pet; garden: Garden; shifted: boolean; brain: string;
  mode: 'watch' | 'play' | 'edit'; actions: number[]; demonstration: Lesson[];
  lessons: Lesson[]; personal?: Policy;
  resetMemory?: boolean;
}
export function newPet(now: number): Pet {
  return { name: 'Pip', born: now, clock: now, fullness: 80, energy: 85, affection: 70,
    growth: 0, sleepUntil: 0, adventures: 0, lastCare: {}, lastGrowth: {}, rewarded: [],
    memories: [{ at: now, text: 'Moved into a little garden. Everything smells new.' }],
    chat: [{ speaker: 'pet', text: 'Hi! I’m Pip. Will you show me around? Try “explore”, “snack” or “how are you?”.' }] };
}
const clamp = (n: number) => Math.max(20, Math.min(100, n));
export function advance(pet: Pet, now: number): Pet {
  const elapsed = Math.min(Math.max(0, now - pet.clock), 24 * HOUR);
  const slept = Math.min(elapsed, Math.max(0, pet.sleepUntil - pet.clock));
  return { ...pet, clock: Math.max(now, pet.clock), fullness: clamp(pet.fullness - elapsed / HOUR * 4),
    energy: clamp(pet.energy - elapsed / HOUR * 2 + slept / 1000 * .75),
    affection: clamp(pet.affection - elapsed / HOUR) };
}
export function remember(pet: Pet, text: string): Pet {
  return { ...pet, memories: [...pet.memories, { at: pet.clock, text }].slice(-80) };
}
export function stage(pet: Pet) { return pet.growth >= 120 ? 'Trailkeeper' : pet.growth >= 40 ? 'Scout' : 'Hatchling'; }
export function mood(pet: Pet) {
  return pet.sleepUntil > pet.clock ? 'Dreaming of strawberries.' : pet.fullness < 45 ? 'A little peckish. Snack time?' :
    pet.energy < 45 ? 'Tiny paws could use a rest.' : pet.affection < 45 ? 'A cuddle would be nice.' : 'Safe, curious, and happy you’re here.';
}
export function care(source: Pet, kind: Care, now: number): { pet: Pet; message: string } {
  let pet = advance(source, now);
  if (kind === 'nap' && pet.sleepUntil > pet.clock) return { pet, message: 'Already curled up. You can wake me whenever you like.' };
  if (pet.lastCare[kind] !== undefined && pet.clock - pet.lastCare[kind]! < MINUTE)
    return { pet, message: 'Let’s enjoy that for a moment. Try again in a minute.' };
  const gain = pet.lastGrowth[kind] === undefined || pet.clock - pet.lastGrowth[kind]! >= 15 * MINUTE;
  pet = { ...pet, lastCare: { ...pet.lastCare, [kind]: pet.clock },
    lastGrowth: gain ? { ...pet.lastGrowth, [kind]: pet.clock } : pet.lastGrowth, growth: Math.min(10000, pet.growth + (gain ? 4 : 0)) };
  const text = kind === 'feed' ? 'A strawberry snack, shared with you.' : kind === 'cuddle' ? 'A very good cuddle. Little paws, big feelings.' : 'Curled up for a 30-second garden nap.';
  if (kind === 'feed') pet = { ...pet, fullness: clamp(pet.fullness + 25) };
  if (kind === 'cuddle') pet = { ...pet, affection: clamp(pet.affection + 12) };
  if (kind === 'nap') pet = { ...pet, sleepUntil: pet.clock + 30_000 };
  return { pet: remember(pet, text), message: kind === 'feed' ? 'Mmm. Berry moustache. Thank you!' : kind === 'cuddle' ? 'I like this part of the day.' : 'Zzz… Wake me early if an adventure calls.' };
}
export function adventure(source: Pet, garden: Garden, outcome: string, now: number): Pet {
  let pet = advance(source, now);
  const day = Math.floor(pet.clock / (24 * HOUR));
  const key = `${day}:${garden.start}:${garden.goal}:${garden.cells.join('')}`;
  const rewarded = pet.rewarded.filter(entry => entry.startsWith(`${day}:`));
  const fresh = !rewarded.includes(key) && rewarded.length < 128, success = outcome === 'success';
  pet = { ...pet, adventures: pet.adventures + 1, energy: clamp(pet.energy - 3),
    fullness: clamp(pet.fullness + (success ? 8 : 0)), affection: clamp(pet.affection + (success ? 3 : 0)),
    growth: Math.min(10000, pet.growth + (fresh ? success ? 8 : 2 : 0)),
    rewarded: fresh ? [...rewarded, key] : rewarded };
  return remember(pet, success ? `Found a berry in garden #${garden.seed}. A little braver now.` :
    `Tried garden #${garden.seed}. ${outcome === 'hazard' ? 'Met a prickly bramble.' : 'Got stuck, but kept trying.'}`);
}
export function chat(pet: Pet, speaker: Chat['speaker'], text: string): Pet {
  return { ...pet, chat: [...pet.chat, { speaker, text }].slice(-24) };
}
export type Intent = Care | 'wake' | 'explore' | 'needs' | 'memories' | 'lessons' | 'hello' | 'unknown';
export function intent(text: string): Intent {
  const words: string[] = text.toLowerCase().replace(/[’']/g, '').match(/[a-z]+/g) ?? [];
  const has = (...keys: string[]) => keys.some(key => words.includes(key));
  if (has('not', 'dont', 'no', 'stop')) return 'unknown';
  if (has('wake', 'awake')) return 'wake';
  if (has('feed', 'food', 'snack', 'berry', 'berries', 'hungry')) return 'feed';
  if (has('cuddle', 'hug', 'pet', 'pat', 'love')) return 'cuddle';
  if (has('sleep', 'nap', 'rest')) return 'nap';
  if (has('explore', 'adventure', 'walk', 'go')) return 'explore';
  if (has('remember', 'memories', 'memory', 'history')) return 'memories';
  if (has('learn', 'learned', 'learning', 'lesson', 'lessons', 'teach', 'brain')) return 'lessons';
  if (has('feel', 'feeling', 'needs') || /how\s+are\s+you/.test(text.toLowerCase())) return 'needs';
  if (has('hi', 'hello', 'hey', 'morning')) return 'hello';
  return 'unknown';
}

// Validate untrusted storage and backups before they can replace a companion.
type Obj = Record<string, unknown>;
function object(value: unknown): Obj { if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected an object'); return value as Obj; }
function number(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) throw new Error('Invalid number'); return value;
}
function integer(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  const n = number(value, min, max); if (!Number.isInteger(n)) throw new Error('Invalid integer'); return n;
}
function string(value: unknown, limit: number): string {
  if (typeof value !== 'string' || !value.trim() || value.length > limit || /[\x00-\x1f]/.test(value)) throw new Error('Invalid text'); return value;
}
function array(value: unknown, limit: number): unknown[] { if (!Array.isArray(value) || value.length > limit) throw new Error('Invalid array'); return value; }
function lesson(value: unknown): Lesson {
  const v = object(value), observation = array(v.observation, 20).map(x => number(x, -1, 1));
  if (observation.length !== 20) throw new Error('Invalid observation');
  return { observation, action: integer(v.action, 0, 3) };
}
function policy(value: unknown): Policy {
  const v = object(value), architecture = array(v.architecture, 4).map(x => integer(x));
  if (architecture.join(',') !== '20,64,64,4' || v.activation !== 'tanh') throw new Error('Unsupported network');
  const layers = array(v.layers, 3).map((raw, i) => {
    const layer = object(raw), weight = array(layer.weight, architecture[i + 1]).map(row => array(row, architecture[i]).map(x => number(x, -1000, 1000)));
    const bias = array(layer.bias, architecture[i + 1]).map(x => number(x, -1000, 1000));
    if (weight.length !== architecture[i + 1] || bias.length !== weight.length || weight.some(row => row.length !== architecture[i])) throw new Error('Invalid network dimensions');
    return { weight, bias };
  });
  if (layers.length !== 3) throw new Error('Missing layers');
  return { label: string(v.label, 80), seed: integer(v.seed, 0, 4294967295), environment_steps: integer(v.environment_steps), architecture, activation: 'tanh', layers };
}
export function decodeSave(text: string): Save {
  if (new TextEncoder().encode(text).length > MAX_BACKUP_BYTES) throw new Error('Backup exceeds 1 MB');
  const v = object(JSON.parse(text)); if (v.version !== 1) throw new Error('Unsupported backup version');
  const p = object(v.pet), g = object(v.garden);
  const careTimes = (raw: unknown) => {
    const o = object(raw), result: Partial<Record<Care, number>> = {};
    for (const kind of ['feed', 'cuddle', 'nap'] as const) if (o[kind] !== undefined) result[kind] = number(o[kind]);
    return result;
  };
  const pet: Pet = { name: string(p.name, 24), born: number(p.born), clock: number(p.clock),
    fullness: number(p.fullness, 20, 100), energy: number(p.energy, 20, 100), affection: number(p.affection, 20, 100),
    growth: integer(p.growth, 0, 10000), sleepUntil: number(p.sleepUntil), adventures: integer(p.adventures),
    lastCare: careTimes(p.lastCare), lastGrowth: careTimes(p.lastGrowth), rewarded: array(p.rewarded, 128).map(x => string(x, 120)),
    memories: array(p.memories, 80).map(raw => { const m = object(raw); return { at: number(m.at), text: string(m.text, 400) }; }),
    chat: array(p.chat, 24).map(raw => { const c = object(raw); if (!['you', 'pet'].includes(String(c.speaker))) throw new Error('Invalid speaker'); return { speaker: c.speaker as Chat['speaker'], text: string(c.text, 500) }; }) };
  if (pet.clock < pet.born || pet.sleepUntil > pet.clock + 30_000 || Object.values(pet.lastCare).some(t => t > pet.clock) || Object.values(pet.lastGrowth).some(t => t > pet.clock)) throw new Error('Invalid pet clock');
  const garden: Garden = { cells: array(g.cells, 49).map(x => integer(x, 0, 2)), start: integer(g.start, 0, 48), goal: integer(g.goal, 0, 48), seed: integer(g.seed, 0, 4294967295) };
  const interior = (i: number) => i % 7 > 0 && i % 7 < 6 && i >= 7 && i < 42;
  if (garden.cells.length !== 49 || !interior(garden.start) || !interior(garden.goal) || garden.start === garden.goal || garden.cells[garden.start] !== 0 || garden.cells[garden.goal] !== 0 || garden.cells.some((cell, i) => !interior(i) && cell !== 1)) throw new Error('Invalid garden');
  const actions = array(v.actions, 48).map(x => integer(x, 0, 3)), demonstration = array(v.demonstration, 48).map(lesson);
  const world = new World(garden);
  actions.forEach((action, i) => {
    if (world.done) throw new Error('Actions after episode end');
    if (demonstration.length && (demonstration[i]?.action !== action || demonstration[i]?.observation.some((x, j) => Math.abs(x - world.obs()[j]) > 1e-9))) throw new Error('Demonstration disagrees with episode');
    world.step(action);
  });
  if (demonstration.length && demonstration.length !== actions.length) throw new Error('Incomplete demonstration');
  const personal = v.personal === undefined ? undefined : policy(v.personal);
  if (!['watch', 'play', 'edit'].includes(String(v.mode)) || !['bc-ppo', 'bc', 'ppo', 'untrained', 'personal', 'r1-gru', 'r1-reactive'].includes(String(v.brain)) || (v.brain === 'personal' && !personal) || typeof v.shifted !== 'boolean' || (v.resetMemory !== undefined && typeof v.resetMemory !== 'boolean')) throw new Error('Invalid session');
  return { version: 1, pet, garden, shifted: v.shifted, brain: v.brain as string, mode: v.mode as Save['mode'], actions, demonstration, lessons: array(v.lessons, 2400).map(lesson), ...(personal ? { personal } : {}), ...(v.resetMemory === undefined ? {} : {resetMemory:v.resetMemory as boolean}) };
}
