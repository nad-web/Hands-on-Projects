import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {MODELS} from '../assets/core.js';
import {ranks,spearman,cosine,topFeature,mixForecasts,softmax,lstmCell,bhAdjust,dmHAC,pairwiseEvidence} from '../assets/scientific.js';
import katex from '../assets/vendor/katex/katex.mjs';
const load=async p=>JSON.parse(await readFile(new URL('../'+p,import.meta.url),'utf8'));
const near=(a,b,tol=1e-12)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('signed ranks handle ties; positive rescaling does not measure magnitude change',()=>{
 assert.deepEqual(ranks([3,1,1,2]),[4,1.5,1.5,3]);
 const a=[-.2,.01,.1,-.05],b=a.map(x=>x*7);
 near(spearman(a,b),1);near(cosine(a,b),1);near(spearman(a,a.map(x=>-x)),-1);
 assert.equal(topFeature(a),0);assert.equal(topFeature([0,0,0,0]),null);assert.equal(spearman([0,0],[1,2]),null);assert.equal(cosine([0,0],[1,2]),null);
});
test('ensemble uses identical origins, normalizes weights, and rejects the empty mixture',()=>{
 const p=Object.fromEntries(MODELS.map((m,j)=>[m,[j,2*j]]));
 const v=mixForecasts(p,{Linear:10,TFT:30});assert.deepEqual(v.values,[4.5,9]);near(v.normalized.TFT,.75);
 assert.equal(mixForecasts(p,{}),null);assert.equal(mixForecasts(p,{Linear:-3}),null);
});
test('neural teaching calculations obey normalization and gate arithmetic',()=>{
 const p=softmax([1000,1001,1002]);near(p.reduce((s,x)=>s+x,0),1);assert.ok(p[2]>p[1]);
 const v=lstmCell(.8,.7,.4,.5,.8);near(v.memory,.76);near(v.hidden,.8*Math.tanh(.76));
});
test('BH correction preserves missing values and DM rejects degenerate variance',()=>{
 assert.deepEqual(bhAdjust([.01,.04,null,.03]),[.03,.04,null,.04]);
 assert.equal(dmHAC(Array(30).fill(0),Array(30).fill(0),1).p,null);
});
test('all 420 browser DM calculations agree with recorded block-aware tests',async()=>{
 const s=await load('data/summary.json'),groups=new Map();let count=0;
 for(const r of s.dm_pairwise){const key=`${r.series}-h${r.horizon}-${r.scheme}`;
  if(!groups.has(key))groups.set(key,pairwiseEvidence(await load('data/forecasts/'+key+'.json')));
  const result=groups.get(key).find(x=>x.model_a===r.model_a&&x.model_b===r.model_b);assert.ok(result);
  near(result.dm,r.dm_statistic_a_minus_b,1e-10);near(result.p,r.p_raw,2e-7);near(result.q,r.p_bh_21_pairs,4e-7);assert.equal(result.q<.05,r.significant_q05);count++;
 }
 assert.equal(count,420);
});
test('the classroom covers all models, 3 questions, 15 appendix abbreviations and valid mathematics',async()=>{
 const c=await load('data/learning.json');assert.equal(c.lessons.length,9);assert.equal(c.questions.length,3);assert.equal(c.models.length,7);assert.ok(c.equations.length>=31);assert.ok(c.terms.length>=54);assert.ok(c.quizzes.length>=16);
 assert.deepEqual(c.models.map(m=>m.id),MODELS);
 assert.deepEqual(c.terms.filter(t=>t.listed).map(t=>t.key).sort(),['AIC','ARIMA','BH','DA','DM','HAC','LIME','LSTM','MAE','OLS','PFI','RMSE','SHAP','TFT','XAI'].sort());
 for(const e of c.equations)for(const tex of [e.tex,...e.symbols.map(s=>s[0])])assert.doesNotThrow(()=>katex.renderToString(tex,{throwOnError:true,strict:'error',trust:false}));
 for(const q of c.quizzes)assert.ok(q.correct>=0&&q.correct<q.options.length);
 for(const r of c.questions)assert.ok(r.answer&&r.findings.length&&c.lessons.some(l=>l.id===r.lesson));
});
