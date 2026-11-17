import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {metrics,visibleWindow,linearPredict,splitDesign,parseCSV,csvText} from '../assets/core.js';
const load=async p=>JSON.parse(await readFile(new URL('../'+p,import.meta.url),'utf8'));

test('RMSE, benchmark ratio and R² distinguish distinct scientific claims',()=>{
  const s=metrics([1,-2,3],[0,-1,2],[0,0,0],true);
  assert.equal(s.rmse,1);assert.equal(s.mae,1);assert.equal(s.da,2/3);
  assert.ok(Math.abs(s.relative-Math.sqrt(3/14))<1e-14);
  assert.equal(metrics([10,20],[9,22],[10,10],false).da,null);
  assert.equal(metrics([],[],[],true).rmse,null);
});
test('visible replay windows stay in a contiguous matched test block',()=>{
  const d={y_true:Array(8).fill(0),matched_fold:[0,0,0,0,1,1,1,1]};
  assert.deepEqual(visibleWindow(d,5,6),{start:4,end:6});
  assert.deepEqual(visibleWindow(d,7,2),{start:6,end:8});
});
test('exact split semantics preserve purge, embargo, rolling size and deep warm-up',()=>{
  const r=splitDesign(8809,'Linear','rolling',2);assert.equal(r.trainEnd-r.trainStart,500);assert.equal(r.testStart-r.trainEnd,10);assert.equal(r.testEnd-r.testStart,252);assert.equal(r.folds,6);
  const d=splitDesign(8809,'TFT','expanding',2);assert.equal(d.trainStart,0);assert.equal(d.testStart,1014);assert.equal(d.testEnd-d.testStart-d.warmup,231);assert.equal(d.folds,4);
});
test('what-if inference uses actual standardized inputs and coefficients',()=>{
  const fit={intercept:.1,mean:[10,20,30,40],scale:[2,2,5,10],weights:[.5,-.1,.2,0]};
  assert.equal(linearPredict(fit,[10,20,30,40]),.1);
  assert.ok(Math.abs(linearPredict(fit,[12,22,35,40])-.7)<1e-14);
});
test('CSV preview/export preserves quoted delimiters, quotes and multiline values',()=>{
  const input=[['a','b'],['comma,inside','quote"inside'],['line\nbreak','7']];
  assert.deepEqual(parseCSV(csvText(input)),input);
  assert.equal(parseCSV(csvText(input),2).length,2);
});
test('all 140 browser metric cells reproduce the corrected CSV summaries',async()=>{
  const summary=await load('data/summary.json'),cache=new Map();
  for(const r of summary.matched_metrics){
    const key=`${r.series}-h${r.horizon}-${r.scheme}`;
    if(!cache.has(key))cache.set(key,await load('data/forecasts/'+key+'.json'));
    const d=cache.get(key),s=metrics(d.y_true,d.predictions[r.model],d.random_walk_pred,r.series!=='monash_sunspot');
    for(const [a,b] of [['rmse','rmse'],['mae','mae'],['r2','r2'],['relative','rmse_relative_to_persistence']])assert.ok(Math.abs(s[a]-r[b])<2e-11,`${key}/${r.model}/${a}`);
    if(r.series==='monash_sunspot')assert.equal(s.da,null);else assert.ok(Math.abs(s.da-r.da)<1e-14);
  }
});
