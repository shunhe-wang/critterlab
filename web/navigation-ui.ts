import { type Garden } from './engine';
import { runGarden, shortCycle, type GardenPolicy } from './navigation';

const el = <T extends HTMLElement = HTMLElement>(id:string) => document.getElementById(id)! as T;
type Painter = (ctx:CanvasRenderingContext2D,x:number,y:number,scale:number,bounce?:number,success?:boolean)=>void;
interface Score { mean:number; std:number; values:number[] }
interface Evidence { summary:Record<string,Record<string,Record<string,Score>>> }

export function mountNavigationComparison(drawPet:Painter,getGarden:()=>Garden,getPolicies:()=>Record<string,GardenPolicy>,onStart:()=>void,onDiagnostic:()=>void) {
  const button = el<HTMLButtonElement>('navigation-run');
  let timer:ReturnType<typeof setInterval>|undefined, frame = 0, patch = getGarden(), evidence:Evidence|undefined;
  let records:Record<string,ReturnType<typeof runGarden>> = {};
  function pause() { if(timer)clearInterval(timer);timer=undefined;button.textContent=Object.keys(records).length?'▶ Replay comparison':'▶ Compare this patch'; }
  function refresh() { pause();frame=0;records={};patch=getGarden();button.disabled=!getPolicies()['bc-ppo']||!getPolicies()['r1-gru'];render(); }
  function render() {
    el('navigation-patch').textContent=`GARDEN #${patch.seed}`;
    for(const method of ['current','gru','reset']) {
      const record=records[method], shown=record?.frames[Math.min(frame,record.frames.length-1)] ?? {pos:patch.start,steps:0,outcome:'running'};
      const ctx=el<HTMLCanvasElement>(`navigation-${method}`).getContext('2d')!;
      ctx.clearRect(0,0,280,280);
      patch.cells.forEach((cell,i)=> {
        const x=7+i%7*38,y=7+Math.floor(i/7)*38;
        ctx.fillStyle=cell===1?'#b6c5a5':cell===2?'#d9bca7':'#e7edda';ctx.beginPath();ctx.roundRect(x+1,y+1,36,36,cell===1?9:4);ctx.fill();
        if(cell===2){ctx.strokeStyle='#ac8778';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x+12,y+13);ctx.lineTo(x+24,y+25);ctx.moveTo(x+24,y+13);ctx.lineTo(x+12,y+25);ctx.stroke();}
      });
      if(record){ctx.strokeStyle=method==='current'?'#c88552':'#6e91aa';ctx.lineWidth=2;ctx.beginPath();record.frames.slice(0,frame+1).forEach((f,i)=>{const x=26+f.pos%7*38,y=26+Math.floor(f.pos/7)*38;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
      const gx=26+patch.goal%7*38,gy=26+Math.floor(patch.goal/7)*38;
      ctx.fillStyle='#c97873';ctx.beginPath();ctx.ellipse(gx,gy+2,8,10,0,0,Math.PI*2);ctx.fill();ctx.fillStyle='#72945d';ctx.fillRect(gx-6,gy-8,12,3);
      drawPet(ctx,26+shown.pos%7*38,26+Math.floor(shown.pos/7)*38,.48,0,shown.outcome==='success');
      const repeating=record && shortCycle(record.frames.slice(0,Math.min(frame+1,record.frames.length)).map(f=>f.pos));
      el(`navigation-${method}-status`).textContent=!record?'Ready to compare.':shown.outcome==='success'?`Berry found · ${shown.steps} moves`:shown.outcome==='hazard'?`Bramble · ${shown.steps} moves`:shown.outcome==='timeout'?`Timeout · 48 moves${record.loopTimeout?' · repeating tiles':''}`:`Move ${shown.steps} / 48${repeating?' · repeating tiles':''}`;
    }
  }
  function compare() {
    if(timer){pause();return;}
    const policies=getPolicies();if(!policies['bc-ppo']||!policies['r1-gru'])return;
    onStart();patch=getGarden();frame=0;
    records={current:runGarden(policies['bc-ppo'],patch),gru:runGarden(policies['r1-gru'],patch),reset:runGarden(policies['r1-gru'],patch,true)};
    button.textContent='Ⅱ Pause comparison';render();
    timer=setInterval(()=>{frame++;render();if(Object.values(records).every(r=>frame>=r.frames.length-1))pause();},250);
  }
  button.onclick=compare;
  el('navigation-diagnostic').onclick=()=>{onDiagnostic();compare();};
  function scores() {
    if(!evidence)return;
    const split=el<HTMLSelectElement>('navigation-split').value;
    el('navigation-scores').replaceChildren(...[['current','Current brain'],['gru','Memory GRU'],['reactive','Retrained reactive'],['gru-reset','GRU, state reset']].map(([key,label])=>{
      const row=document.createElement('div');row.className='result-row';const name=document.createElement('span'),value=document.createElement('strong');
      const metrics=evidence!.summary[key][split];name.textContent=label;
      value.textContent=`${(metrics.success_rate.mean*100).toFixed(1)}% · ${(metrics.loop_timeout_rate.mean*100).toFixed(1)}%`;row.append(name,value);return row;
    }));
    el('navigation-caption').textContent='The GRU reduces repeating-tile timeouts, but enters more brambles and has lower overall success. Mean across 3 model seeds; 256 unique gardens per split, excluded from all old/new training. Final seed 0 powers these replays; full variability is in the report.';
  }
  el('navigation-split').onchange=scores;
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  refresh();
  void(async()=>{try{const response=await fetch(`${import.meta.env.BASE_URL}data/navigation.json`);if(!response.ok)throw new Error('Missing r1 result');evidence=await response.json();scores();}catch{el('navigation-caption').textContent='Measured results could not load. Read the linked report or refresh to try again.';}})();
  return {pause,refresh};
}
