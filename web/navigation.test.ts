import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {generate} from './engine';
import {GardenAgent,inspectGarden,runGarden,type GardenPolicy} from './navigation';
import {newPet,decodeSave,type Save} from './pet';

const policy = (method:string):GardenPolicy => JSON.parse(readFileSync(`public/models/${method==='current'?'bc-ppo':'r1-'+method}.json`,'utf8'));
test('garden GRU follows full carried-state PyTorch fixtures on actual trajectories',()=>{
  const fixtures=JSON.parse(readFileSync('tests/fixtures/navigation-parity.json','utf8'));
  for(const vector of fixtures){
    const garden=generate(vector.map_seed,vector.shifted);assert.deepEqual(garden,vector.garden);
    const agent=new GardenAgent(policy(vector.method),garden);
    for(const frame of vector.frames){
      assert.equal(agent.world.pos,frame.pos);
      const decision=agent.decision();
      agent.world.obs().forEach((v,i)=>assert.ok(Math.abs(v-frame.observation[i])<1e-7));
      decision.probabilities.forEach((v,i)=>assert.ok(Math.abs(v-frame.probabilities[i])<2e-6));
      decision.hidden?.forEach((v,i)=>assert.ok(Math.abs(v-frame.hidden[i])<2e-6));
      assert.equal(agent.advance(),frame.action);
    }
    assert.equal(agent.world.outcome,vector.outcome);
  }
});
test('browser reproduces every published seed-0 garden action, loop flag and reset control',()=>{
  const result=JSON.parse(readFileSync('results/navigation-r1.json','utf8'));
  for(const run of result.runs.filter((r:{seed:number})=>r.seed===0)){
    const groups=[...Object.values(run.evaluation),...(run.controls?Object.values(run.controls):[])];
    const episodes=groups.flatMap(value=>(value as {episodes:unknown[]}).episodes);
    for(const record of [...episodes,...run.diagnostics,...(run.reset_diagnostics??[])] as {map_seed:number;shifted:boolean;actions:number[];outcome:string;any_cycle:boolean;loop_timeout:boolean}[]){
      const reset=(Object.values(run.controls??{}) as {episodes:unknown[]}[]).some(e=>e.episodes.includes(record))||(run.reset_diagnostics??[]).includes(record);
      const actual=runGarden(policy(run.method),generate(record.map_seed,record.shifted),reset);
      assert.deepEqual(actual.actions,record.actions,`${run.method} seed ${record.map_seed}, reset=${reset}`);
      assert.equal(actual.outcome,record.outcome);assert.equal(actual.anyCycle,record.any_cycle);assert.equal(actual.loopTimeout,record.loop_timeout);
    }
  }
});
test('replay and portable saves reconstruct the same garden neural state at every frame',()=>{
  const actor=policy('gru'),garden=generate(67,true),agent=new GardenAgent(actor,garden);
  const save:Save={version:1,pet:newPet(0),garden,shifted:true,brain:'r1-gru',mode:'watch',actions:[],demonstration:[],lessons:[],resetMemory:false};
  while(!agent.world.done){
    agent.advance();save.actions=[...agent.actions];
    const restored=decodeSave(JSON.stringify(save));assert.equal(restored.brain,'r1-gru');
    const view=inspectGarden(actor,restored.garden,restored.actions,restored.actions.length,restored.resetMemory);
    assert.equal(view.world.pos,agent.world.pos);assert.deepEqual(view.hidden,agent.hidden);
    assert.deepEqual(view.probabilities,agent.decision().probabilities);
  }
  // Expected diagnostic outcome is in the committed Python result, not a score floor.
  const result=JSON.parse(readFileSync('results/navigation-r1.json','utf8'));
  const diagnostic=result.runs.find((r:{method:string;seed:number})=>r.method==='gru'&&r.seed===0).diagnostics.find((r:{map_seed:number;shifted:boolean})=>r.map_seed===67&&r.shifted);
  assert.equal(agent.world.outcome,diagnostic.outcome);assert.equal(agent.world.steps,diagnostic.steps);
  assert.throws(()=>decodeSave(JSON.stringify({...save,resetMemory:'false'})),/session/);
  const forgotten=decodeSave(JSON.stringify({...save,resetMemory:true}));assert.equal(forgotten.resetMemory,true);
});
