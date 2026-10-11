import { World, cloneGarden, probabilities, type Garden, type Policy } from './engine';
import { recurrentStep, type RecurrentPolicy } from './memory';

export type GardenPolicy = Policy | RecurrentPolicy;
export const isRecurrent = (policy: GardenPolicy): policy is RecurrentPolicy => 'kind' in policy && policy.kind === 'gru';
export function gardenDecision(policy: GardenPolicy, world: World, hidden?: number[], forget = false) {
  return isRecurrent(policy) ? recurrentStep(policy, world.obs(), forget ? undefined : hidden) : {probabilities: probabilities(policy,world.obs()),hidden:undefined};
}
export interface GardenFrame { pos: number; steps: number; outcome: string }
export class GardenAgent {
  world: World; hidden?: number[]; actions: number[] = []; frames: GardenFrame[];
  constructor(public policy: GardenPolicy, garden: Garden, public forget = false) {
    this.world = new World(cloneGarden(garden));
    this.frames = [{pos:this.world.pos,steps:0,outcome:'running'}];
  }
  decision() { return gardenDecision(this.policy,this.world,this.hidden,this.forget); }
  advance(action?: number) {
    if (this.world.done) throw new Error('Episode is already over');
    const decision = this.decision(); this.hidden = decision.hidden;
    const chosen = action ?? decision.probabilities.indexOf(Math.max(...decision.probabilities));
    this.world.step(chosen); this.actions.push(chosen);
    this.frames.push({pos:this.world.pos,steps:this.world.steps,outcome:this.world.outcome});
    return chosen;
  }
}
export function inspectGarden(policy: GardenPolicy, garden: Garden, actions: number[], frame: number, forget = false) {
  if (!Number.isInteger(frame) || frame < 0 || frame > actions.length) throw new Error('Invalid replay frame');
  const agent = new GardenAgent(policy,garden,forget);
  for (const action of actions.slice(0,frame)) agent.advance(action);
  return {world:agent.world,hidden:agent.hidden,probabilities:agent.decision().probabilities};
}
export function shortCycle(positions: number[]) {
  const tail = positions.slice(-6);
  return tail.length===6 && tail.every((pos,i)=>pos===tail[i%2]);
}
export function runGarden(policy: GardenPolicy, garden: Garden, forget = false) {
  const agent = new GardenAgent(policy,garden,forget);
  while (!agent.world.done) agent.advance();
  const positions = agent.frames.map(f=>f.pos);
  return {actions:agent.actions,frames:agent.frames,outcome:agent.world.outcome,
    anyCycle:positions.some((_,i)=>shortCycle(positions.slice(0,i+1))),
    loopTimeout:agent.world.outcome==='timeout' && shortCycle(positions)};
}
