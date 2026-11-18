import {MODELS,MODEL,SERIES,LAGS,esc,urlPath as encodePath,finite,mean,median,fmt,pct,num,dateLabel,sizeLabel,clamp,metrics,visibleWindow,linearPredict,splitDesign,parseCSV,csvText} from './core.js';
import {readScenarioQuery,writeScenarioQuery} from './scenario-data.js';
import {mountScenarioExplorer} from './scenarios.js';
import {sparkline,forecastChart,attributionBars} from './charts.js';
import {marked} from './vendor/marked.js';
import {setupAcademic,academicHash,academicView,academicAfterRender,blendView} from './academic.js';
const APP_ROOT=new URL('../',import.meta.url);
const urlPath=path=>new URL(encodePath(path),APP_ROOT).href;

const $ = s=>document.querySelector(s);
const main=$('#main'), cache=new Map();
const PATHS={home:['Observatory','grid'],questions:['Research Questions','compass'],replay:['Forecast Lab','activity'],arena:['Model Arena','layers'],explain:['Explanation Studio','scan'],protocol:['Walk-forward Design','timeline'],learn:['Learn','book'],reader:['Research Guide','book'],vault:['Evidence Vault','folder']};
const ICONS={grid:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',activity:'M2 12h4l4-8 4 16 4-8h4',layers:'m12 3 10 5-10 5L2 8 12 3z M2 12l10 5 10-5 M2 16l10 5 10-5',scan:'M8 3H3v5 M16 3h5v5 M3 16v5h5 M21 16v5h-5 M8 12h8 M12 8v8',timeline:'M5 3v18 M19 3v18 M3 7h18 M3 17h18 M9 12h6',book:'M12 5v16 M3 3c4-1 7 0 9 2 2-2 5-3 9-2v16c-4-1-7 0-9 2-2-2-5-3-9-2V3z',folder:'M3 7V4h6l3 3h9v13H3V7z',compass:'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z m4-16-2 8-8 2 2-8 8-2z',download:'M12 3v12 m-5-5 5 5 5-5 M4 16v5h16v-5',search:'M10.5 18a7.5 7.5 0 1 0 0-15 7.5 7.5 0 0 0 0 15z M16 16l5 5',link:'m10 13 4-4 M8 16l-1 1a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0 M16 8l1-1a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0',sun:'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M12 2v2 M12 20v2 M2 12h2 M20 12h2 M5 5l1.5 1.5 M17.5 17.5 19 19 M5 19l1.5-1.5 M17.5 6.5 19 5',menu:'M3 6h18 M3 12h18 M3 18h18',arrow:'M4 12h16 m-6-6 6 6-6 6',external:'M14 3h7v7 M21 3 10 14 M10 3H3v18h18v-7',play:'m8 5 11 7-11 7V5z',pause:'M8 5v14 M16 5v14',back:'m14 6-6 6 6 6',next:'m10 6 6 6-6 6',close:'m6 6 12 12 M6 18 18 6',check:'m4 12 5 5L20 6',code:'m8 6-6 6 6 6 M16 6l6 6-6 6 M14 3l-4 18',file:'M5 2h9l5 5v15H5V2z M14 2v6h5 M8 12h8 M8 16h8',eye:'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',reset:'M3 10a9 9 0 1 1 2 9 M3 4v6h6',shield:'M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6l-9-4z m-4 10 3 3 5-6'};
const icon=(name,cls='')=>`<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${ICONS[name]||ICONS.file}"/></svg>`;
function icons(){document.querySelectorAll('[data-icon]').forEach(el=>{el.innerHTML=icon(el.dataset.icon);});}
const state={view:'home',series:'sp500',h:1,scheme:'rolling',model:'Linear',cursor:175,span:120,compare:'ARIMA',mode:'return',all:false,replayTab:'replay',scope:'financial',fold:0,probe:0,readerTab:'guide',chapter:'chapter-1',vaultCategory:'All',vaultMode:'files',vaultQuery:'',vaultPage:0,readerQuery:'',litCategory:'All'};
let manifest,summary,inventory,guideData,linearData,renderId=0,playing=false,timer=null,tourStep=-1,plotData=null,whatIf=null,csvPreview=null;
function toast(message){$('#toast').textContent=message;$('#toast').classList.add('show');clearTimeout(toast.timeout);toast.timeout=setTimeout(()=>$('#toast').classList.remove('show'),3200);}
async function getJSON(path){if(!cache.has(path))cache.set(path,fetch(urlPath(path)).then(r=>{if(!r.ok)throw new Error(`${path} (${r.status})`);return r.json();}).catch(e=>{cache.delete(path);throw e;}));return cache.get(path);}
const forecasts=()=>getJSON(`data/forecasts/${state.series}-h${state.h}-${state.scheme}.json`);
const rawSeries=()=>getJSON(`data/series/${state.series}.json`);
const attrs=()=>getJSON(`data/attributions/${state.model}-${state.series}-h${state.h}-${state.scheme}.json`);
const financial=()=>state.series!=='monash_sunspot';
const unit=(v,d=2)=>financial()?pct(v,d):fmt(v,d);
let activeScenario = null;

function scenarioKey() {
  return [state.series, state.h, state.scheme, state.cursor].join('|');
}
function currentScenarioSettings() {
  const saved = state.scenario;
  if (saved?.key === scenarioKey()) return saved.options;
  const settings = readScenarioQuery(new URLSearchParams());
  if (saved) {
    settings.pool = saved.options.pool;
    settings.count = saved.options.count;
  }
  return settings;
}
function openWorkedScenario(example) {
  const context = example==='dax'
    ? {series:'dax', h:5, scheme:'expanding', cursor:231, model:'Linear'}
    : {series:'monash_sunspot', h:1, scheme:'rolling', cursor:923, model:'ARIMA'};
  const key = [context.series, context.h, context.scheme, context.cursor].join('|');
  navigate('replay', {
    ...context, replayTab:'scenarios',
    scenario:{key, options:{pool:'all', count:5, profile:null, selectedDate:null}}
  });
}

function hashState(){const q=new URLSearchParams({s:state.series,h:String(state.h),w:state.scheme,m:state.model,t:String(state.cursor)});if(state.view==='reader'){q.set('chapter',state.chapter);q.set('r',state.readerTab);}if(state.view==='arena')q.set('scope',state.scope);if(state.view==='replay'){q.set('span',String(state.span));q.set('c',state.compare);q.set('u',state.mode);if(state.all)q.set('all','1');if(state.replayTab!=='replay')q.set('lab',state.replayTab);}if(state.view==='explain'||state.view==='protocol'){q.set('f',String(state.fold));if(state.view==='explain')q.set('p',String(state.probe));}academicHash(q);
  if(state.view==='replay' && state.replayTab==='scenarios') writeScenarioQuery(q, currentScenarioSettings());
  return `#${state.view}?${q}`;}
function readHash(){const [view,query='']=location.hash.slice(1).split('?');if(PATHS[view])state.view=view;const q=new URLSearchParams(query);if(SERIES[q.get('s')])state.series=q.get('s');if(MODEL[q.get('m')])state.model=q.get('m');if(['rolling','expanding'].includes(q.get('w')))state.scheme=q.get('w');if(['1','5'].includes(q.get('h')))state.h=Number(q.get('h'));if(q.has('t'))state.cursor=clamp(Math.round(Number(q.get('t')))||0,0,923);if(q.has('chapter'))state.chapter=q.get('chapter');if(state.view==='replay'){
    state.replayTab=['whatif','blend','scenarios'].includes(q.get('lab'))?q.get('lab'):'replay';
    if(state.replayTab==='scenarios')state.scenario={key:scenarioKey(),options:readScenarioQuery(q)};
  }if(q.has('span'))state.span=clamp(Math.round(Number(q.get('span')))||120,20,231);if(q.has('c')&&(q.get('c')===''||MODEL[q.get('c')]))state.compare=q.get('c');if(['return','level'].includes(q.get('u')))state.mode=q.get('u');if(q.has('all'))state.all=q.get('all')==='1';if(['financial','cell'].includes(q.get('scope')))state.scope=q.get('scope');if(['guide','literature'].includes(q.get('r')))state.readerTab=q.get('r');if(q.has('f'))state.fold=clamp(Math.round(Number(q.get('f')))||0,0,5);if(q.has('p'))state.probe=clamp(Math.round(Number(q.get('p')))||0,0,39);}
function remember(){history.replaceState(null,'',hashState());}
function navigate(view,patch={}){stopPlayback();Object.assign(state,patch,{view});remember();render();window.scrollTo({top:0,behavior:'instant'});$('#sidebar').classList.remove('open');$('#sidebar-scrim').classList.remove('open');}
function header(eyebrow,title,desc,actions=''){return `<div class="page-header"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${desc}</p></div>${actions?`<div class="flex wrap">${actions}</div>`:''}</div>`;}
const button=(label,action,ic='arrow',primary=false,extra='')=>`<button class="btn${primary?' primary':''}" data-action="${action}" ${extra}>${label}${icon(ic)}</button>`;
const sourceLink=(name='matched_metrics.csv',text='Open source CSV ')=>`<button class="quiet source-link" data-file="research/results/cross_model/tables/${name}">${text}</button>`;
const modelName=m=>MODEL[m].name;
function contextBar(extra=''){return `<div class="context-bar"><label class="field"><span>Dataset</span><select aria-label="Dataset" data-global="series">${Object.entries(SERIES).map(([id,s])=>`<option value="${id}" ${state.series===id?'selected':''}>${esc(s)}</option>`).join('')}</select></label><label class="field compact"><span>Forecast horizon</span><select aria-label="Forecast horizon" data-global="h"><option value="1" ${state.h===1?'selected':''}>1 ${financial()?'session':'day'}</option><option value="5" ${state.h===5?'selected':''}>5 ${financial()?'sessions':'days'}</option></select></label><label class="field compact"><span>Walk-forward window</span><select aria-label="Walk-forward window" data-global="scheme"><option value="rolling" ${state.scheme==='rolling'?'selected':''}>Rolling</option><option value="expanding" ${state.scheme==='expanding'?'selected':''}>Expanding</option></select></label>${extra}</div>`;}
function modelTabs(){return `<div class="model-strip" aria-label="Select model">${MODELS.map(m=>`<button class="model-tab ${state.model===m?'active':''}" data-model="${m}" aria-pressed="${state.model===m}"><span class="model-dot" style="background:${MODEL[m].color}"></span>${m==='RandomForest'?'Random Forest':m==='TFT'?'TFT adapter':m}</button>`).join('')}</div>`;}
function modelCard(m=state.model){const d=MODEL[m];return `<div class="panel model-card"><div class="between"><h3>${esc(d.name)}</h3><span class="pill">${m==='Linear'||m==='ARIMA'?'BASELINE':m==='LSTM'||m==='TFT'?'DEEP LEARNING':'ENSEMBLE'}</span></div><div class="family">${d.family.toUpperCase()}</div><p>${esc(d.config)}</p><div class="keyvals">${d.params.map(([k,v])=>`<div class="keyval"><span>${k}</span><strong>${v}</strong></div>`).join('')}</div><button class="quiet source-link mt" data-file="research/notebooks/${d.notebook}">Inspect the executed notebook </button></div>`;}
function metricBox(title,value,sub){return `<div class="metric"><span>${title}</span><strong>${value}</strong><small>${sub}</small></div>`;}
function setMain(html){main.innerHTML=`<div class="page-enter">${html}</div>`;icons();}
async function render(){
  const id=++renderId;
  activeScenario=null;
  // Keep filters responsive during a data switch, but inactivate the previous
  // setting's sliders until the new archive and observations are both ready.
  const previousScenario=$('#scenario-explorer');
  if(previousScenario)previousScenario.inert=true;
  main.setAttribute('aria-busy','true');
  $('#navigation').innerHTML=Object.entries(PATHS).map(([key,[label,ic]],i)=>`<a class="nav-item ${state.view===key?'active':''}" href="#${key}" data-route="${key}" ${state.view===key?'aria-current="page"':''}>${icon(ic)}<span>${label}</span><span class="nav-num">0${i+1}</span></a>`).join('');
  $('#breadcrumb').textContent=PATHS[state.view][0];document.title=`${PATHS[state.view][0]} · Forecast Atlas`;
  try{
    let html='', replayObservations, scenarioInputs;
    if(state.view==='home'){const d=await getJSON('data/forecasts/sp500-h1-rolling.json');if(id!==renderId)return;html=homeView(d);}
    if(state.view==='replay'){
      const isScenario=state.replayTab==='scenarios';
      const [data,raw,extra]=await Promise.all([
        forecasts(), rawSeries(), getJSON(isScenario ? 'data/notebooks.json' : 'data/linear.json')
      ]);
      if(id!==renderId)return;
      plotData=data;
      replayObservations=raw;
      if(isScenario)scenarioInputs={forecasts:data,observations:raw,notebooks:extra};
      else linearData=extra;
      html=replayView(data,raw);
    }
    if(state.view==='arena')html=arenaView();
    if(state.view==='explain'){const a=await attrs();if(id!==renderId)return;plotData=a;html=explainView(a);}
    if(state.view==='protocol'){const raw=await rawSeries();if(id!==renderId)return;plotData=raw;html=protocolView(raw);}
    if(state.view==='reader'){guideData=await getJSON('data/guide.json');if(id!==renderId)return;html=readerView();}
    if(state.view==='questions'||state.view==='learn'){html=await academicView(state.view);if(id!==renderId)return;}
    if(state.view==='vault')html=vaultView();
    if(id!==renderId)return;setMain(html);remember();
    if(scenarioInputs){
      activeScenario=mountScenarioExplorer($('#scenario-explorer'),{
        ...scenarioInputs, observationFile:manifest.series.find(s=>s.id===state.series).file, url:urlPath,
        settings:currentScenarioSettings(), anchorIndex:state.cursor, model:state.model,
        onChange(index,model,options){
          if(id!==renderId)return;
          state.cursor=index;
          state.model=model;
          state.scenario={key:scenarioKey(),options};
          remember();
        },
        onReplay:(cursor,model)=>navigate('replay',{cursor,model,replayTab:'replay'}),
        onExport:(name,rows)=>download(name,csvText(rows),'text/csv;charset=utf-8'),
        onExample:openWorkedScenario
      });
    }else if(state.view==='replay'){
      if(state.replayTab==='replay')updateReplay();
      else if(state.replayTab==='whatif')initWhatIf(replayObservations);
      else if(state.replayTab==='blend')academicAfterRender('blend');
    }
    if(state.view==='questions'||state.view==='learn')academicAfterRender(state.view);
    if(state.view==='explain')updateProbe();
    if(state.view==='protocol')updateTimeline();
  }catch(error){if(id!==renderId)return;setMain(`<div class="browser-error"><h1>The archive could not be opened</h1><p>${esc(error.message)}</p><p>For a local preview, serve this folder over HTTP:</p><code>python3 -m http.server 8000</code><p class="mt">Then open <a href="http://localhost:8000">http://localhost:8000</a>. On GitHub Pages, check that the data and research folders were uploaded alongside index.html.</p>${button('Try again','retry','reset')}</div>`);console.error(error);}
  finally{if(id===renderId)main.removeAttribute('aria-busy');}
}

function homeView(d){
  const ranked=summary.financial_model_summary.slice().sort((a,b)=>a.mean_relative_rmse-b.mean_relative_rmse),max=Math.max(...ranked.map(r=>r.mean_relative_rmse));
  const chart=d.predictions.Linear.slice(10,190);
  const chapterNames=['Research questions','Data & targets','Seven model families','Experimental design','Prediction & inference','Explanation methods','Stability & faithfulness','Interpretation & limits'];
  return `<section class="hero"><div><div class="eyebrow">FINANCIAL FORECASTING / EXPLAINABLE AI</div><h1>A forecast is only as good as its <em>evidence.</em></h1><p class="hero-desc">Step inside a master’s thesis on trustworthy machine learning. Replay the forecasts. Question the explanations. Follow every result back to its source.</p><div class="hero-buttons">${button('Enter the forecast lab','go-replay','arrow',true)}<button class="quiet small" data-route="reader">Explore the research guide ${icon('external')}</button></div><div class="hero-note">SEVEN MODELS. THREE QUESTIONS. ONE TRANSPARENT EXPERIMENT.</div></div><div class="hero-visual"><div class="hero-chart"><div class="hero-chart-top"><span class="flex"><span class="status-dot"></span> S&P 500 / FORECAST REPLAY</span><span class="pill">h = 1</span></div><div class="hero-chart-value">${pct(d.predictions.Linear[189],3)}</div><div class="tiny muted">Recorded Linear Regression forecast · cumulative log return</div><div style="margin:22px 0 8px">${sparkline(chart,'var(--accent)',480,125)}</div><div class="hero-chart-label"><span>${d.origin_date[10]}</span><span>${d.origin_date[189]}</span></div></div><div class="hero-float">${icon('shield')}<div>Evidence you can inspect<small>${fmt(manifest.matched_predictions,0)} matched predictions · source linked</small></div></div></div></section>
    <div class="stats-ribbon">${[[7,'executed forecasting models'],['18,480','shared forecast origins'],[manifest.references,'cited references'],[manifest.files,'original research files']].map(([n,l])=>`<div class="stat-ribbon"><div class="number">${n}</div><div class="label">${l}</div></div>`).join('')}</div>
    <div class="section-head"><h2>The complexity question</h2><p>MORE COMPLEX DOESN’T AUTOMATICALLY MEAN MORE ACCURATE</p></div>
    <div class="dashboard-grid"><section class="panel"><div class="panel-head"><div><h3>Can the models beat persistence?</h3><p>Mean relative RMSE across 16 financial settings · lower is better</p></div><span class="pill">RQ1</span></div><div class="ranking-chart">${ranked.map((r,i)=>`<button class="quiet rank-row" style="display:grid;width:100%" data-model-home="${r.model}" aria-label="Explore ${modelName(r.model)}"><span class="rank-model"><span class="model-dot" style="background:${MODEL[r.model].color}"></span>${r.model==='RandomForest'?'Random Forest':r.model}</span><span class="track"><span class="fill" style="width:${(r.mean_relative_rmse-1)/.02*100}%;background:${MODEL[r.model].color};display:block"></span></span><span class="value">${fmt(r.mean_relative_rmse,4)}</span></button>`).join('')}<div class="rank-scale"><span>1.000 / PERSISTENCE</span><span>1.020</span></div></div><div class="ranking-summary"><strong>Every model is above 1.000 on average.</strong> Post-hoc transparency does not create predictive skill.</div><div class="panel-foot">${sourceLink('financial_model_summary.csv','financial_model_summary.csv ')}<button class="quiet tiny" data-route="arena">Compare all models →</button></div></section>
    <section class="panel thesis-card"><div class="between"><span class="eyebrow">THE RESEARCH</span>${icon('book')}</div><h3 style="font-size:22px;margin-top:15px;line-height:1.4">Towards Trustworthy Financial Time Series Forecasting</h3><p class="small" style="margin-top:12px">An experimental evaluation of explainable machine learning models for stock price prediction.</p><div class="chapter-lines">${chapterNames.map((t,i)=>`<div class="chapter-line"><span>0${i+1}</span><span>${t}</span></div>`).join('')}</div><button class="btn" data-route="reader">Open the study guide ${icon('arrow')}</button></section></div>
    <div class="section-head"><h2>Three questions. Look beneath the surface.</h2><p>EXPLORE THE FINDINGS</p></div>
    <div class="grid3">${[
      ['01','Accuracy & interpretability','Can post-hoc explanations bridge the gap between simple models and black boxes?','Only partially. Complexity brings no general predictive advantage.','arena','Compare the evidence'],
      ['02','Explanations through time','Do feature attributions remain stable as the training window and market regime change?','Stability depends on the model, diagnostic, and window.','explain','Explore temporal stability'],
      ['03','Faithful is not causal','Do explanations reflect model mechanics—and reveal genuine economic drivers?','Additivity is supported; broader faithfulness is mixed.','explain','Inspect the explanations']
    ].map(([n,title,q,a,route,link])=>`<article class="panel rq-card" data-ac-question="${Number(n)}" tabindex="0" role="button"><div class="rq-top"><span class="rq-number">RESEARCH QUESTION / ${n}</span>${icon('arrow')}</div><h3>${title}</h3><p>${q}</p><p style="color:var(--text)">${a}</p><div class="rq-link"><span>${link}</span><span></span></div></article>`).join('')}</div>
    <div class="section-head"><h2>Meet the model families</h2><p>SELECT A MODEL TO START EXPLORING</p></div>${modelTabs()}
    <div class="callout mt"><strong>A living interface to a fixed experiment.</strong> The controls recalculate views of the supplied research. Replay uses recorded out-of-sample forecasts; the Linear what-if lab performs real inference with verified reconstructed OLS coefficients. Blend all seven recorded models, examine each research question, or learn the methods step by step. Full training code is available in the eight documented notebooks.</div>`;
}

function replayView(d,raw){
  const modes=[['replay','Forecast replay'],['scenarios','Historical scenarios'],['whatif','Linear what-if lab'],['blend','Blend models']];
  const tabs=`<div class="segmented lab-modes" role="group" aria-label="Laboratory mode">${modes.map(([id,label])=>
    `<button data-replay-tab="${id}" class="${state.replayTab===id?'active':''}" aria-pressed="${state.replayTab===id}">${label}</button>`).join('')}</div>`;
  if(state.replayTab==='scenarios'){
    return header('FORECAST LAB','Historical scenarios','Choose four lagged observations. Find similar dates and compare what each model forecast with what happened next.')+contextBar(tabs)+'<div id="scenario-explorer"></div>';
  }
  const head=header('03 / EXPERIMENT IN MOTION','The Forecast Lab','Move through the original out-of-sample forecasts. Every point comes from the executed study.',(state.replayTab==='replay'?button('Save chart','export-chart','download'):'')+button('Export view','export-replay','download'))+contextBar(tabs);
  if(state.replayTab==='whatif')return head+whatIfView(d);
  if(state.replayTab==='blend')return head+blendView(d);
  const s=manifest.series.find(s=>s.id===state.series);
  return head+`<div class="model-tabs-wrap">${modelTabs()}<label class="field compact"><span>Compare against</span><select id="compare-model" aria-label="Comparison model"><option value="">None</option>${MODELS.map(m=>`<option value="${m}" ${state.compare===m?'selected':''}>${MODEL[m].name}</option>`).join('')}</select></label></div>
    <section class="panel"><div class="replay-layout"><div class="replay-plot"><div class="panel-head"><div><h3>${esc(SERIES[state.series])} <span class="muted">/</span> <span id="chart-model-name">${MODEL[state.model].name}</span></h3><p id="chart-subtitle">${financial()?'Cumulative future log return':'Native Sunspot target value'} · archived out-of-sample forecasts</p><div id="chart-legend" class="legend"></div></div><span class="pill green">RECORDED DATA</span></div><div id="forecast-chart" class="chart-shell"></div><div class="replay-options"><span>Visible window</span><input id="window-range" aria-label="Visible forecast window" type="range" min="20" max="231" step="1" value="${state.span}"><output id="window-output">${state.span} origins</output><label class="flex" style="margin-left:auto;gap:5px"><input id="all-models" type="checkbox" ${state.all?'checked':''}> All models</label>${financial()?`<select id="chart-mode" aria-label="Plot units"><option value="return" ${state.mode==='return'?'selected':''}>Log return</option><option value="level" ${state.mode==='level'?'selected':''}>Implied level</option></select>`:''}</div></div><aside class="replay-inspector"><div class="section-label">AT THIS FORECAST ORIGIN</div><div id="inspect-date" class="inspect-date"></div><div id="inspect-target" class="inspect-sub"></div><div class="inspect-line"><span>MODEL FORECAST</span><strong id="inspect-pred" class="accent"></strong><div id="inspect-pred-sub" class="inspect-sub"></div></div><div class="inspect-line"><span>REALIZED OUTCOME</span><strong id="inspect-actual"></strong><div class="inspect-sub">Observed held-out target</div></div><div class="inspect-line"><span>ABSOLUTE ERROR</span><strong id="inspect-error"></strong><div id="inspect-hit" class="inspect-sub"></div></div><div class="inspector-footer tiny faint" style="margin-top:auto;padding-top:20px">Forecast values are cumulative log returns for financial indices, and future native values for Sunspot.</div></aside></div>
    <div class="replay-controls"><button class="icon-btn" data-action="step-back" aria-label="Previous forecast">${icon('back')}</button><button id="play-button" class="play-btn" data-action="play" aria-label="Play forecast replay">${icon('play')}</button><button class="icon-btn" data-action="step-next" aria-label="Next forecast">${icon('next')}</button><div class="range-track"><input id="replay-range" aria-label="Forecast timeline" type="range" min="0" max="923" value="${state.cursor}"><div class="range-labels"><span>${d.origin_date[0]}</span><span id="cursor-label"></span><span>${d.origin_date[923]}</span></div></div><select id="play-speed" aria-label="Replay speed"><option value="400">1×</option><option value="160">2.5×</option><option value="80">5×</option></select></div>
    <div id="replay-metrics" class="metrics-grid"></div><div class="panel-foot"><span>Metrics recalculate over the visible window in the current matched block.</span>${sourceLink('matched_observation_predictions.csv','Inspect original predictions ')}</div></section>
    <div class="grid2 mt">${modelCard()}<div class="panel panel-pad"><div class="between"><h3>The source series</h3><span class="pill">${fmt(s.rows,0)} observations</span></div><div class="spark">${sparkline(raw.level.filter((_,i)=>i%Math.max(1,Math.floor(raw.level.length/400))===0),'var(--blue)',480,80)}</div><div class="between tiny muted"><span>${s.start}</span><span>${s.end}</span></div><p class="small mt">${financial()?'Lagged log returns are model inputs. The target is the cumulative future log return; “implied level” converts a forecast using the observed origin level.':'The robustness block uses native Sunspot values verified against the supplied TSF. It is separate from financial performance summaries.'}</p><div class="keyvals"><div class="keyval"><span>Matched origins / model / setting</span><strong>924</strong></div><div class="keyval"><span>Matched blocks</span><strong>4 × 231 observations</strong></div><div class="keyval"><span>Current window</span><strong>${state.scheme} · h = ${state.h}</strong></div></div><button class="quiet source-link mt" data-file="${s.file}">Inspect source observations </button></div></div>`;
}

function updateReplay(){
  if(!plotData||state.view!=='replay'||state.replayTab!=='replay'||!$('#forecast-chart'))return;
  const d=plotData,i=state.cursor,{start,end}=visibleWindow(d,i,state.span),pred=d.predictions[state.model][i],truth=d.y_true[i];
  $('#forecast-chart').innerHTML=forecastChart(d,{...state,compact:matchMedia('(max-width:760px)').matches});
  $('#chart-legend').innerHTML=`<span><i style="--lc:var(--muted)"></i> Realized</span><span><i style="--lc:${MODEL[state.model].color}"></i> ${MODEL[state.model].name}</span>${state.compare&&state.compare!==state.model?`<span><i style="--lc:${MODEL[state.compare].color}"></i> ${MODEL[state.compare].name}</span>`:''}${state.all?MODELS.filter(m=>m!==state.model&&m!==state.compare).map(m=>`<span><i style="--lc:${MODEL[m].color}"></i> ${MODEL[m].name}</span>`).join(''):''}<span><i class="dashed" style="--lc:var(--blue)"></i> ${financial()?'Zero-return':'Current-value'} persistence</span>`;
  $('#inspect-date').textContent=dateLabel(d.origin_date[i]);$('#inspect-target').textContent=`Target: ${d.target_date[i]} · h = ${state.h}`;
  $('#inspect-pred').textContent=unit(pred,3);$('#inspect-actual').textContent=unit(truth,3);$('#inspect-error').textContent=unit(Math.abs(pred-truth),3);
  $('#inspect-pred-sub').textContent=financial()?`Implied target level ${fmt(d.origin_level[i]*Math.exp(pred),2)}`:`Persistence ${fmt(d.random_walk_pred[i],1)}`;
  $('#inspect-hit').textContent=financial()?(Math.sign(pred)===Math.sign(truth)?'Direction matches realized target':'Direction differs from realized target'):'Absolute difference in native units';
  $('#cursor-label').textContent=`${i+1} / 924 · block ${d.matched_fold[i]+1}`;$('#replay-range').value=i;
  const stats=metrics(d.y_true.slice(start,end),d.predictions[state.model].slice(start,end),d.random_walk_pred.slice(start,end),financial());
  $('#replay-metrics').innerHTML=metricBox('WINDOW RMSE',unit(stats.rmse,3),`${stats.n} visible held-out targets`)+metricBox('RMSE / PERSISTENCE',fmt(stats.relative,4),'Below 1.000 means lower error')+metricBox('DIRECTIONAL ACCURACY',pct(stats.da,1),financial()?'Sign agreement · not profitability':'Not defined for the Sunspot task')+metricBox('WINDOW R²',fmt(stats.r2,4),'Relative to this window’s target mean');
  $('#window-output').textContent=`${state.span} origins`;attachPlotPointer();remember();
}
function attachPlotPointer(){
  const svg=$('#forecast-svg'),pointer=$('#plot-pointer');if(!svg||!pointer)return;
  const tooltip=document.createElement('div');tooltip.className='chart-tooltip';tooltip.hidden=true;$('#forecast-chart').append(tooltip);
  pointer.addEventListener('pointermove',e=>{const rect=svg.getBoundingClientRect(),u=(e.clientX-rect.left)/rect.width*svg.viewBox.baseVal.width;const index=Number(svg.dataset.start)+clamp(Math.round((u-Number(svg.dataset.left))/Number(svg.dataset.plotwidth)*(Number(svg.dataset.count)-1)),0,Number(svg.dataset.count)-1);const d=plotData;tooltip.innerHTML=`${esc(d.origin_date[index])}<p>${MODEL[state.model].name}: ${unit(d.predictions[state.model][index],3)}<br>Realized: ${unit(d.y_true[index],3)}</p>`;tooltip.hidden=false;const left=clamp(e.clientX-rect.left+12,4,Math.max(4,rect.width-210));tooltip.style.left=left+'px';tooltip.style.top='15px';});
  pointer.addEventListener('pointerleave',()=>tooltip.hidden=true);
  pointer.addEventListener('click',e=>{const rect=svg.getBoundingClientRect(),u=(e.clientX-rect.left)/rect.width*svg.viewBox.baseVal.width;state.cursor=Number(svg.dataset.start)+clamp(Math.round((u-Number(svg.dataset.left))/Number(svg.dataset.plotwidth)*(Number(svg.dataset.count)-1)),0,Number(svg.dataset.count)-1);updateReplay();});
}
function stopPlayback(){playing=false;clearTimeout(timer);if($('#play-button')){$('#play-button').innerHTML=icon('play');$('#play-button').setAttribute('aria-label','Play forecast replay');}}
function playTick(){if(!playing)return;if(state.cursor>=923){stopPlayback();return;}state.cursor++;updateReplay();timer=setTimeout(playTick,Number($('#play-speed')?.value||400));}

function whatIfView(d){return `<div class="notice mb">LIVE INFERENCE / Linear Regression only. These sliders change inputs to a reconstructed OLS model; edited outputs are hypothetical and do not enter the thesis results.</div><div class="whatif-grid"><section class="panel panel-pad"><div class="between"><div><h3>Change the four lagged inputs</h3><p class="small" style="margin-top:7px">Move a slider to perturb one input by training standard deviations.</p></div>${button('Reset','reset-whatif','reset',false)}</div><div id="feature-sliders"></div><div class="flex mt"><span class="pill green">VERIFIED OLS RECONSTRUCTION</span><span class="tiny muted">All 30,240 original forecasts checked</span></div><p class="tiny mt">The ${state.scheme} training fold at origin <strong>${d.origin_date[state.cursor]}</strong> is fixed. Feature changes are independent interventions; correlated lags may make an edited combination unrealistic.</p></section><section class="panel panel-pad"><div class="eyebrow">HYPOTHETICAL MODEL OUTPUT</div><div id="whatif-pred" class="whatif-value"></div><div id="whatif-delta" class="small muted"></div><div class="whatif-formula">f(x) = intercept + Σ βⱼ · zⱼ<br><span class="faint">zⱼ = (inputⱼ − training meanⱼ) / training stdⱼ</span></div><h4>Exact linear contributions</h4><div id="whatif-contributions" style="margin-top:10px"></div><div class="keyvals"><div class="keyval"><span>Unedited recorded forecast</span><strong id="whatif-baseline"></strong></div><div class="keyval"><span>Training observations</span><strong id="whatif-train"></strong></div><div class="keyval"><span>Reconstruction max. absolute error</span><strong>${manifest.linear_verification.max_abs_error.toExponential(2)}</strong></div></div><p class="tiny mt">For ARIMA, tree ensembles, LSTM and TFT, use recorded replay or the original notebooks. The input package contains no serialized fitted weights for fresh browser inference with those models.</p></section></div><div class="panel mt"><div class="replay-controls"><span class="small">Forecast origin</span><div class="range-track"><input id="whatif-origin" type="range" min="0" max="923" value="${state.cursor}" aria-label="Linear what-if forecast origin"><div class="range-labels"><span>${d.origin_date[0]}</span><span id="whatif-origin-label">${d.origin_date[state.cursor]}</span><span>${d.origin_date[923]}</span></div></div></div></div>`;}
function initWhatIf(raw){
  const d=plotData,fold=d.folds.Linear[state.cursor],fit=linearData.folds[`${state.series}-h${state.h}-${state.scheme}-f${fold}`],offset=financial()?253:252;
  const rawPosition=d.origin_index[state.cursor]+offset,features=LAGS.map(l=>raw.signal[rawPosition-l]);
  whatIf={fit,features,baseline:linearPredict(fit,features),deltas:[0,0,0,0]};
  $('#feature-sliders').innerHTML=LAGS.map((lag,j)=>`<div class="feature-slider"><div class="between"><h4 class="mono">lag_${lag}</h4><output id="feature-value-${j}">${unit(features[j],3)}</output></div><input class="feature-range" type="range" min="-3" max="3" step=".01" value="0" data-feature="${j}" aria-label="Perturb lag ${lag} in training standard deviations"><div class="feature-foot"><span>−3σ from observed input</span><span id="feature-delta-${j}">observed</span><span>+3σ</span></div></div>`).join('');
  $('#whatif-baseline').textContent=unit(d.predictions.Linear[state.cursor],4);$('#whatif-train').textContent=`${fmt(fit.n_train,0)} · fold ${fold+1}`;$('#whatif-origin-label').textContent=d.origin_date[state.cursor];updateWhatIf();remember();
}
function updateWhatIf(){if(!whatIf)return;const {fit,features,deltas,baseline}=whatIf;const edited=features.map((x,j)=>x+deltas[j]*fit.scale[j]);const pred=linearPredict(fit,edited);$('#whatif-pred').textContent=unit(pred,4);$('#whatif-delta').textContent=`${pred-baseline>=0?'+':''}${unit(pred-baseline,4)} from the unedited prediction`;
  edited.forEach((v,j)=>{$(`#feature-value-${j}`).textContent=unit(v,3);$(`#feature-delta-${j}`).textContent=deltas[j]===0?'observed':`${deltas[j]>0?'+':''}${fmt(deltas[j],2)}σ`;});
  $('#whatif-contributions').innerHTML=attributionBars(edited.map((v,j)=>fit.weights[j]*(v-fit.mean[j])/fit.scale[j]),financial());
}

function arenaView(){
  const cell=state.scope==='cell';
  const rows=cell?summary.matched_metrics.filter(r=>r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme).map(r=>({...r,mean_relative_rmse:r.rmse_relative_to_persistence,mean_rank:r.mean_rmse_rank,mean_da:r.da,mean_r2:r.r2})):summary.financial_model_summary;
  const sorted=rows.slice().sort((a,b)=>a.mean_relative_rmse-b.mean_relative_rmse);
  const tests=summary.dm_pairwise.filter(r=>cell?r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme:r.series!=='monash_sunspot');
  const sig=tests.filter(t=>t.significant_q05).length;
  const scopeTabs=`<div class="segmented"><button data-scope="financial" class="${!cell?'active':''}">Financial aggregate</button><button data-scope="cell" class="${cell?'active':''}">Selected experiment</button></div>`;
  return header('04 / PREDICTIVE EVIDENCE','The Model Arena','Compare the seven models at identical forecast origins. Keep predictive error, direction, and statistical evidence separate.',button('Export comparison','export-arena','download'))+
    `<div class="arena-top">${scopeTabs}<span class="pill">${cell?'924 matched origins per model':'16 financial settings per model'}</span></div>`+(cell?contextBar():'')+
    `<div class="panel"><div class="table-title"><div><h3>${cell?esc(SERIES[state.series])+' / h = '+state.h+' / '+state.scheme:'Financial model comparison'}</h3><p class="tiny" style="margin-top:5px">Sorted by ${cell?'relative RMSE':'mean relative RMSE'} · persistence = 1.000 · smaller is better</p></div><span class="pill green">MATCHED ORIGINS</span></div><div class="table-wrap"><table><thead><tr><th>#</th><th>Model</th><th class="numeric">${cell?'Relative':'Mean relative'} RMSE</th>${cell?'<th class="numeric">RMSE</th>':'<th class="numeric">Median relative RMSE</th>'}<th class="numeric">Mean rank</th><th class="numeric">${cell?'':'Mean '}DA</th><th class="numeric">${cell?'':'Mean '}R²</th><th></th></tr></thead><tbody>${sorted.map((r,i)=>`<tr><td class="table-rank">0${i+1}</td><td><span class="model-cell"><span class="model-dot" style="background:${MODEL[r.model].color}"></span>${MODEL[r.model].name}</span></td><td class="numeric ${i===0?'table-best':''}">${fmt(r.mean_relative_rmse,5)}</td><td class="numeric">${cell?num(r.rmse):fmt(r.median_relative_rmse,5)}</td><td class="numeric">${fmt(r.mean_rank,3)}</td><td class="numeric">${pct(r.mean_da,2)}</td><td class="numeric ${r.mean_r2<0?'negative':''}">${fmt(r.mean_r2,5)}</td><td><button class="icon-btn" data-model-home="${r.model}" title="Replay ${modelName(r.model)}" aria-label="Replay ${modelName(r.model)}">${icon('arrow')}</button></td></tr>`).join('')}</tbody></table></div><div class="panel-foot"><span>${cell?'Metrics use all 924 shared origins in this setting.':'Each setting contributes equally to the reported financial means. Sunspot is excluded.'}</span>${sourceLink(cell?'matched_metrics.csv':'financial_model_summary.csv')}</div></div>
    <div class="callout mt"><strong>${!cell?'There is no general black-box accuracy advantage.':sorted[0].mean_relative_rmse<1?MODEL[sorted[0].model].name+' has the lowest descriptive error in this setting.':'The lowest-error model is still above persistence in this setting.'}</strong> ${!cell?'LightGBM has the lowest mean relative RMSE; ARIMA has the best mean rank. Linear Regression has the strongest financial pairwise significance profile. A descriptive lead is distinct from a significant benchmark improvement.':'A rank within one setting does not establish overall superiority. Use the adjusted pairwise tests below to inspect statistical support.'}</div>
    <div class="grid2 mt"><section class="panel"><div class="panel-head"><div><h3>${cell?'Pairwise significance map':'Where the pairwise wins occur'}</h3><p>${cell?'BH-adjusted q < 0.05 · row model versus column model':'Significant wins of the row model over the column model'}</p></div><span class="pill">HAC–DM</span></div><div class="heatmap">${pairHeatmap(tests,cell)}</div><div class="panel-foot"><span>${sig} significant pairs / ${tests.length} pairwise tests</span><span class="tiny">Click a cell to inspect its evidence </span></div></section><section class="panel panel-pad"><div class="eyebrow">HOW TO READ THE EVIDENCE</div><h3 style="margin:15px 0">Three different claims</h3><div class="diagnostic-row"><div>Lower descriptive error<small>RMSE and relative RMSE compare realized forecast loss.</small></div><span class="pill">RANKING</span></div><div class="diagnostic-row"><div>Better than another model<small>Observation-level HAC–DM; BH within each 21-pair setting.</small></div><span class="pill">PAIRWISE</span></div><div class="diagnostic-row"><div>Better than a simple benchmark<small>Supplementary native-coverage tests against persistence and the training-target mean.</small></div><span class="pill">BENCHMARK</span></div><p class="small mt">No financial model–benchmark test is a significant model win under the BH14 primary family or the BH2 sensitivity analysis. Those 224 tests retain native forecast coverage and are separate from the matched seven-model ranking.</p><button class="quiet source-link mt" data-file="research/results/cross_model/tables/native_benchmark_block_dm.csv">Inspect the 224 financial benchmark tests </button></section></div>
    <div class="section-head"><h2>What this means for RQ1</h2><button class="quiet small" data-chapter="chapter-6">Read the discussion </button></div><div class="callout"><strong>Post-hoc explanations mitigate the trade-off partially and conditionally.</strong> They make fitted nonlinear models more inspectable, but neither improve the underlying predictions nor turn a black-box architecture into an intrinsically interpretable model. The experiment supports setting-specific inspection, not a universal winner.</div>`;
}
function pairHeatmap(tests,cell){
  let html='<div></div>'+MODELS.map(m=>`<div class="heat-label">${MODEL[m].short}</div>`).join('');
  for(const a of MODELS){html+=`<div class="heat-label row">${a==='RandomForest'?'Random F.':a}</div>`;for(const b of MODELS){
    if(a===b){html+='<div class="heat-cell diag">—</div>';continue;}
    const pairs=tests.filter(t=>[t.model_a,t.model_b].includes(a)&&[t.model_a,t.model_b].includes(b));
    const wins=pairs.filter(t=>t.significant_q05&&t.lower_loss_model===a).length,losses=pairs.filter(t=>t.significant_q05&&t.lower_loss_model===b).length;
    const val=cell?(wins?'W':losses?'L':'·'):wins;
    const shade=cell?(wins?28:losses?20:0):Math.min(40,wins*6);
    const color=cell&&losses?'var(--red)':'var(--accent2)';
    html+=`<button class="heat-cell" data-pair-a="${a}" data-pair-b="${b}" style="background:color-mix(in srgb,${color} ${shade}%,var(--surface2));color:${shade?'var(--text)':'var(--faint)'}" aria-label="${modelName(a)} versus ${modelName(b)}: ${wins} significant wins and ${losses} losses" title="${cell?`${wins?'Significant win':losses?'Significant loss':'Not significant'}; q = ${fmt(pairs[0]?.p_bh_21_pairs,5)}`:`${wins} significant wins / ${pairs.length} settings`}">${val}</button>`;
  }}return html;
}
function showPair(a,b){
  const rows=summary.dm_pairwise.filter(t=>[t.model_a,t.model_b].includes(a)&&[t.model_a,t.model_b].includes(b)&&(state.scope==='cell'?t.series===state.series&&t.horizon===state.h&&t.scheme===state.scheme:t.series!=='monash_sunspot'));
  showModal(`${MODEL[a].name} / ${MODEL[b].name}`,`<p class="small mb">HAC Diebold–Mariano tests use observation-level squared errors and remain within forecast blocks. The q value is adjusted within the 21 pairs of each setting.</p><div class="table-wrap"><table><thead><tr><th>Series</th><th>Horizon</th><th>Scheme</th><th>Lower loss</th><th class="numeric">Raw p</th><th class="numeric">BH q</th><th>Decision</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(SERIES[r.series])}</td><td>${r.horizon}</td><td>${r.scheme}</td><td>${r.lower_loss_model}</td><td class="numeric">${fmt(r.p_raw,6)}</td><td class="numeric">${fmt(r.p_bh_21_pairs,6)}</td><td class="${r.significant_q05?'accent':'muted'}">${r.significant_q05?'Significant':'Not significant'}</td></tr>`).join('')}</tbody></table></div><div class="mt">${sourceLink('dm_pairwise.csv')}</div>`);
}

function explainView(a){
  state.fold=clamp(state.fold,0,a.folds.length-1);state.probe=clamp(state.probe,0,a.probes.length-1);
  const sample=financial()?'financial':'sunspot';
  const imp=summary.relative_importance_by_cell.filter(r=>r.model===state.model&&r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme);
  const importance=LAGS.map(l=>imp.find(r=>r.feature==='lag_'+l));
  const faith=summary.faithfulness.filter(r=>r.model===state.model&&r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme);
  const diag=(method,metric,agg='mean')=>{const values=faith.filter(r=>(method?r.method===method:true)&&r.metric===metric).map(r=>typeof r.value==='boolean'?Number(r.value):r.value);return agg==='max'?values.length?Math.max(...values):null:mean(values);};
  const diagnostics=[['Maximum additivity residual',diag(null,'max_abs_residual','max'),'max. across explained folds','sci'],['Deletion success fraction',diag(null,'faithful_deletion'),'fraction of folds beating random orderings','pct'],['Insertion success fraction',diag(null,'faithful_insertion'),'fraction of folds beating random orderings','pct'],['PFI / SHAP rank agreement',diag('PFI_vs_SHAP','spearman_rho'),'mean Spearman ρ across explained folds','rho'],['LIME / SHAP rank agreement',diag('LIME_vs_SHAP','spearman_rho'),'mean Spearman ρ across explained folds','rho'],['LIME hold-out MSE',diag('LIME_fidelity','median_holdout_mse'),'mean of the exported per-fold medians','sci']];
  return header('05 / LOOK INSIDE THE PREDICTOR','The Explanation Studio','Explore signed lag attributions, their stability across folds, and the diagnostics that test their limits.')+contextBar()+`<div class="model-tabs-wrap">${modelTabs()}<span class="pill">${MODEL[state.model].method}</span></div>
    <div class="explain-layout"><section class="panel"><div class="panel-head"><div><h3>One probe. Four contributions.</h3><p>${state.model==='ARIMA'?'Native selected-lag counterfactuals · these are not SHAP values':'Signed contributions from the recorded explanation channel'}</p></div><span id="probe-regime" class="pill green"></span></div><div id="probe-bars" class="attribution-list"></div><div class="probe-controls"><label class="field compact"><span>Explained fold</span><select id="attribution-fold" aria-label="Explained fold">${a.folds.map((f,i)=>`<option value="${i}" ${state.fold===i?'selected':''}>Fold ${f+1}</option>`).join('')}</select></label><div class="range-track"><input id="probe-range" type="range" min="0" max="${a.probes.length-1}" value="${state.probe}" aria-label="Attribution probe"><div class="range-labels"><span>Probe 1</span><span id="probe-number"></span><span>Probe ${a.probes.length}</span></div></div></div><div class="probe-meta"><span id="probe-date"></span><span id="probe-origin"></span><span>${financial()?'Units: cumulative log return':'Units: native Sunspot value'}</span></div><div class="panel-foot"><span>Explanation probes are a separate evaluation sample from forecast replay.</span><button class="quiet source-link" data-file="research/results/${state.model}_study/tables/attributions.csv">Source attributions </button></div></section>
    <section class="panel panel-pad"><div class="between"><h3>Relative reliance on each lag</h3><span class="pill">SELECTED SETTING</span></div><p class="small" style="margin:10px 0 20px">Mean within-probe relative absolute attribution. Normalization precedes averaging across probes and folds.</p>${importance.map((r,j)=>`<div class="importance-row"><div class="between"><span>lag_${LAGS[j]}</span><span class="accent">${pct(r?.mean,1)}</span></div><div class="importance-track"><span style="width:${(r?.mean||0)*100}%"></span></div></div>`).join('')}<div class="callout mt"><strong>Input reliance is not an economic mechanism.</strong> This study uses only lagged returns or native Sunspot values. A prominent lag does not identify what causes market movements.</div><div class="mt">${sourceLink('relative_importance_by_cell.csv','Inspect relative-importance exports ')}</div></section></div>
    <div class="section-head"><h2>Does the explanation survive a different training window?</h2><span class="pill">RQ2</span></div><section class="panel"><div class="panel-head"><div><h3>Consecutive-fold attribution stability</h3><p>Median Spearman ρ on fixed probe origins · ${financial()?'pooled financial series and both horizons':'Sunspot, both horizons'} · selected model</p></div><span class="pill">−1 ≤ ρ ≤ 1</span></div><div class="stability-table">${stabilityHeatmap(state.model,sample)}</div><div class="panel-foot"><span>n = probe–fold pairs. Regimes are ex post data-driven strata; crisis is not an independently dated event label.</span>${sourceLink('stability_by_scheme.csv')}</div></section>
    <div class="grid2 mt"><section class="panel panel-pad"><div class="between"><h3>Faithfulness, diagnostic by diagnostic</h3><span class="pill">RQ3</span></div><p class="tiny" style="margin-top:10px">${esc(SERIES[state.series])} · h = ${state.h} · ${state.scheme}. Missing diagnostics are shown as unavailable.</p>${diagnostics.map(([label,v,sub,type])=>`<div class="diagnostic-row"><div>${label}<small>${sub}</small></div><strong>${finite(v)?type==='pct'?pct(v,1):type==='sci'?v.toExponential(2):fmt(v,3):'N/A'}</strong></div>`).join('')}<p class="tiny mt">An additive decomposition can reconstruct a prediction while still being unstable or failing intervention checks. None of these tests establishes causality.</p><div class="mt">${sourceLink('faithfulness.csv','Inspect diagnostics for this setting ')}</div></section>${state.model==='TFT'?tftNativeCard():state.model==='ARIMA'?arimaNativeCard():modelCard()}</div>
    ${['RandomForest','XGBoost','LightGBM'].includes(state.model)?seedAgreementCard():''}<div class="callout mt"><strong>Keep the evidence channels separate.</strong> Temporal stability compares refitted folds. Seed agreement measures stochastic sensitivity. Local additivity, deletion/insertion tests, permutation agreement, and surrogate fidelity answer different questions. The app preserves those distinctions.</div>`;
}
function seedAgreementCard(){const rows=summary.seed_stability_summary.filter(r=>r.model===state.model&&r.sample===(financial()?'financial':'sunspot'));return `<section class="panel mt"><div class="panel-head"><div><h3>How much does the random seed matter?</h3><p>Two-seed attribution agreement · ${financial()?'all financial series, horizons, and schemes':'Sunspot, both horizons and schemes'} · ${MODEL[state.model].name}</p></div><span class="pill">20261017 / 20261018</span></div><div class="table-wrap"><table><thead><tr><th>Regime stratum</th><th class="numeric">Median Spearman ρ</th><th class="numeric">Top-feature agreement</th><th class="numeric">Probe–fold comparisons</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${esc(r.regime==='na'?'No market regime':r.regime)}</td><td class="numeric">${fmt(r.rho_median,3)}</td><td class="numeric">${pct(r.top1_match_rate,1)}</td><td class="numeric">${fmt(r.count,0)}</td></tr>`).join('')}</tbody></table></div><div class="panel-foot"><span>Seed agreement and temporal stability measure different sources of variation.</span>${sourceLink('seed_stability_summary.csv')}</div></section>`;}
function stabilityHeatmap(model,sample){
  const regimes=sample==='financial'?['bear','boundary','bull','crisis']:['na'];
  const columns=`grid-template-columns:85px repeat(${regimes.length},1fr)`;
  return `<div class="stability-line" style="${columns}"><div></div>${regimes.map(r=>`<div class="stability-header">${r==='na'?'NO MARKET REGIME':r.toUpperCase()}</div>`).join('')}</div>`+['rolling','expanding'].map(scheme=>`<div class="stability-line" style="${columns}"><div class="stability-title">${scheme[0].toUpperCase()+scheme.slice(1)}</div>${regimes.map(regime=>{const r=summary.stability_by_scheme.find(r=>r.model===model&&r.sample===sample&&r.scheme===scheme&&r.regime===regime);const value=r?.rho_median;return `<div class="stability-cell" style="background:color-mix(in srgb,${value<0?'var(--red)':'var(--accent2)'} ${finite(value)?10+Math.abs(value)*23:0}%,var(--surface2))">${fmt(value,3)}<small>n = ${fmt(r?.count,0)}</small></div>`;}).join('')}</div>`).join('');
}
function updateProbe(){const a=plotData;if(!a||!$('#probe-bars'))return;state.probe=clamp(state.probe,0,a.probes.length-1);state.fold=clamp(state.fold,0,a.folds.length-1);$('#probe-bars').innerHTML=attributionBars(a.values[state.fold][state.probe],financial());$('#probe-regime').textContent=a.regimes[state.probe]==='na'?'SUNSPOT / NO REGIME':a.regimes[state.probe].toUpperCase();$('#probe-date').textContent=dateLabel(a.dates[state.probe]);$('#probe-origin').textContent='Frame origin '+a.origins[state.probe];$('#probe-number').textContent=`${state.probe+1} / ${a.probes.length}`;remember();}
function tftNativeCard(){
  const rows=summary.tft_native_signals.filter(r=>r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme);
  return `<section class="panel panel-pad"><div class="between"><h3>Inside the TFT adapter</h3><span class="pill purple">NATIVE SIGNALS</span></div><p class="small" style="margin:12px 0 16px">Mean native variable-selection weights over all ${rows.length} forecasting folds in this setting. These are different from the KernelSHAP contributions above.</p>${LAGS.map(l=>{const v=mean(rows.map(r=>r['vs_lag_'+l]));return `<div class="importance-row"><div class="between mono tiny"><span>lag_${l}</span><span>${pct(v,2)}</span></div><div class="importance-track"><span style="width:${(v||0)*100}%;background:var(--purple)"></span></div></div>`;}).join('')}<div class="keyvals"><div class="keyval"><span>Mean attention position (0–20)</span><strong>${fmt(mean(rows.map(r=>r.attention_recency)),3)}</strong></div></div><p class="tiny mt">Native attention and variable selection are inspection signals, not proof of faithful explanation or economic causality.</p><div class="mt">${sourceLink('tft_native_signals.csv')}</div></section>`;
}
function arimaNativeCard(){
  const rows=summary.arima_native_params.filter(r=>r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme);
  return `<section class="panel panel-pad"><div class="eyebrow">STATISTICAL MODEL MECHANICS</div><h3 style="margin-top:13px">The selected ARIMA orders</h3><p class="small" style="margin:12px 0 18px">Seven candidate orders compete on training AIC. At each forecast origin the fitted state consumes all available observed raw rows, with parameters held fixed.</p><div class="table-wrap"><table><thead><tr><th>Native fold</th><th>Selected (p,d,q)</th><th class="numeric">AIC</th></tr></thead><tbody>${rows.map(r=>`<tr><td>Fold ${r.fold+1}</td><td class="mono">${esc(r.order)}</td><td class="numeric">${fmt(r.aic,2)}</td></tr>`).join('')}</tbody></table></div><p class="tiny mt">The lag counterfactual replaces selected observations with the raw training mean. The forecast benchmark uses the horizon-specific supervised-target training mean. They serve different purposes.</p><div class="mt">${sourceLink('arima_native_params.csv')}</div></section>`;
}

function protocolView(raw){
  const deep=['LSTM','TFT'].includes(state.model),n=raw.dates.length-(financial()?258:257),design=splitDesign(n,state.model,state.scheme,state.fold);state.fold=design.fold;
  return header('06 / NO PEEKING INTO THE FUTURE','The Walk-forward Design','See exactly where training ends, where the temporal gap sits, and which observations are evaluated.')+contextBar()+`<div class="model-tabs-wrap">${modelTabs()}<span class="pill">${deep?'4 neural folds · 231 forecasts / fold':'6 static folds · 252 forecasts / fold'}</span></div><section class="panel"><div class="panel-head"><div><h3>Time moves in one direction.</h3><p id="timeline-subtitle"></p></div><span class="pill green">EXACT EXECUTED SPLITS</span></div><div style="padding:25px 24px 0"><div id="timeline-chart"></div><div class="legend mb"><span><i style="--lc:var(--accent2)"></i> Training</span><span><i style="--lc:var(--gold)"></i> Purge · 5</span><span><i style="--lc:var(--red)"></i> Embargo · 5</span><span><i style="--lc:var(--blue)"></i> Test block · 252</span>${deep?'<span><i style="--lc:var(--purple)"></i> Neural warm-up · 21</span>':''}</div><div class="timeline-controls"><span class="small">Inspect fold <strong id="timeline-fold-label" class="mono accent">${state.fold+1}</strong></span><div class="range-track"><input id="timeline-range" type="range" min="0" max="${design.folds-1}" value="${state.fold}" aria-label="Walk-forward fold"></div></div></div><div id="split-dates" class="split-dates"></div><div class="panel-foot"><span>Frame indices are after the 252-lag warm-up and five-step target tail are removed.</span><button class="quiet source-link" data-file="research/notebooks/${MODEL[state.model].notebook}">Inspect split construction in the notebook </button></div></section>
    <div class="protocol-grid mt"><article class="panel protocol-card"><span>01 / TARGET & INPUTS</span><h3>${financial()?'Predict returns, not a price trend.':'Preserve the native target.'}</h3><p>${financial()?'Financial target: log(Pₜ₊ₕ / Pₜ), the sum of the next h daily log returns. Features are lagged log returns at 1, 5, 21, and 252 observations.':'Sunspot uses the future native value at h = 1 or 5 days, with the same four lag positions. It is a nonfinancial robustness block and is excluded from financial aggregates.'}</p><div class="keyvals"><div class="keyval"><span>Scaling</span><strong>Fit on training rows</strong></div><div class="keyval"><span>Temporal separation</span><strong>5 purge + 5 embargo</strong></div></div></article><article class="panel protocol-card"><span>02 / COMMON FORECAST ORIGINS</span><h3>Match before comparing.</h3><p>Static models export 1,512 native origins per setting; neural models export 924. Primary comparisons retain the intersection: 924 origins per model, arranged in four matched blocks of 231.</p><div class="keyvals"><div class="keyval"><span>Datasets × horizons × schemes</span><strong>5 × 2 × 2 = 20</strong></div><div class="keyval"><span>Shared origins / all settings</span><strong>18,480</strong></div></div></article><article class="panel protocol-card"><span>03 / EXPLANATION PROBES</span><h3>Hold the probe sample fixed.</h3><p>Consecutive refitted models explain identical out-of-sample probe origins. Static models use four explained folds and 40 probes; neural models use three explained folds and 30 probes. Explanation coverage is separate from forecast coverage.</p><div class="keyvals"><div class="keyval"><span>Tree seeds</span><strong>20261017 / 20261018</strong></div><div class="keyval"><span>Other model seeds</span><strong>20261017</strong></div></div></article></div>
    <div class="section-head"><h2>From an observation to a research claim</h2><button class="quiet small" data-chapter="chapter-4">Read the full methodology </button></div><section class="panel panel-pad"><div class="flow-steps">${[['01','Source data','Frozen index observations and TSF-verified Sunspot values.'],['02','Walk-forward fit','Training-only standardization; model-specific validation tails.'],['03','Separate evaluations','Matched prediction loss, fixed explanation probes, and native benchmark tests.'],['04','Bounded interpretation','Adjusted inference and diagnostic-specific conclusions for RQ1–RQ3.']].map(([n,h,p])=>`<div class="flow-step"><span>${n}</span><h4 style="margin-bottom:9px">${h}</h4><p class="tiny">${p}</p></div>`).join('')}</div></section>`;
}
function updateTimeline(){
  if(!$('#timeline-chart'))return;const raw=plotData,n=raw.dates.length-(financial()?258:257),offset=financial()?253:252;
  const design=splitDesign(n,state.model,state.scheme,state.fold),all=Array.from({length:design.folds},(_,f)=>splitDesign(n,state.model,state.scheme,f));
  const min=state.scheme==='rolling'?all[0].trainStart:0,max=all.at(-1).testEnd,W=900,H=75+design.folds*48,L=65,R=30,pw=W-L-R,x=v=>L+(v-min)/(max-min)*pw;
  const day=p=>raw.dates[clamp(p+offset,0,raw.dates.length-1)];
  let svg=`<svg class="timeline-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${state.scheme} walk-forward training and test intervals, selected fold ${design.fold+1}">`;
  for(let j=0;j<5;j++){const p=Math.round(min+(max-min)*j/4);svg+=`<line x1="${x(p)}" x2="${x(p)}" y1="22" y2="${H-34}" class="chart-grid"/><text x="${x(p)}" y="${H-10}" text-anchor="${j===0?'start':j===4?'end':'middle'}" class="chart-axis">${day(p)}</text>`;}
  for(const d of all){const y=29+d.fold*48,active=d.fold===design.fold;svg+=`<g opacity="${active?1:.4}"><text x="12" y="${y+14}" class="chart-axis">F${d.fold+1}</text><rect x="${x(d.trainStart)}" y="${y}" width="${x(d.trainEnd)-x(d.trainStart)}" height="20" rx="3" fill="var(--accent2)"/><rect x="${x(d.trainEnd)}" y="${y}" width="${Math.max(1,x(d.embargoStart)-x(d.trainEnd))}" height="20" fill="var(--gold)"/><rect x="${x(d.embargoStart)}" y="${y}" width="${Math.max(1,x(d.testStart)-x(d.embargoStart))}" height="20" fill="var(--red)"/><rect x="${x(d.testStart)}" y="${y}" width="${x(d.testEnd)-x(d.testStart)}" height="20" rx="3" fill="var(--blue)"/>${d.warmup?`<rect x="${x(d.testStart)}" y="${y}" width="${x(d.testStart+d.warmup)-x(d.testStart)}" height="20" rx="2" fill="var(--purple)"/>`:''}${active?`<rect x="${L-8}" y="${y-8}" width="${pw+16}" height="36" rx="6" fill="none" stroke="var(--border)"/>`:''}</g>`;}
  svg+='</svg>';$('#timeline-chart').innerHTML=svg;$('#timeline-fold-label').textContent=design.fold+1;
  $('#timeline-subtitle').textContent=`${MODEL[state.model].name} · ${state.scheme} scheme · ${fmt(design.trainEnd-design.trainStart,0)} training rows in selected fold`;
  $('#split-dates').innerHTML=[['TRAINING INTERVAL',`${day(design.trainStart)} → ${day(design.trainEnd-1)}`],['FIRST EVALUATED ORIGIN',day(design.testStart+design.warmup)],['LAST EVALUATED ORIGIN',day(design.testEnd-1)]].map(([l,v])=>`<div class="split-date"><span>${l}</span><strong>${v}</strong></div>`).join('');
  remember();
}

function highlight(text,query){if(!query)return esc(text);const escaped=query.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');return esc(text).replace(new RegExp(esc(escaped),'gi'),m=>`<mark>${m}</mark>`);}
function referenceTopic(r){const t=r.text.toLowerCase();return /shap|explain|interpretab|attribution|lime|attention|black.box/.test(t)?'Explainability':/forecast|financial|stock|asset pricing|time series|time-series|market/.test(t)?'Forecasting':'Evaluation & methods';}
function readerView(){
  const tabs=`<div class="segmented"><button data-reader-tab="guide" class="${state.readerTab==='guide'?'active':''}">Study guide</button><button data-reader-tab="literature" class="${state.readerTab==='literature'?'active':''}">Literature · ${guideData.references.length}</button></div>`;
  return header('RESEARCH GUIDE / CONCEPTS & SOURCES','Follow the argument. Inspect the evidence.','Newly written explanations connect the experiment, the findings, and the literature.',`<a class="btn" href="${urlPath('documentation/')}">Complete documentation ${icon('external')}</a>`)+`<div class="reader-search"><input id="reader-search" type="search" placeholder="Search guide topics or bibliography…" aria-label="Search research guide and literature" value="${esc(state.readerQuery)}">${tabs}</div><div id="reader-body">${readerBody()}</div>`;
}
function readerBody(){
  if(state.readerTab==='literature')return literatureBody();
  const topics=[['Research questions','Accuracy–interpretability, temporal stability, and explanation faithfulness are separate questions. Begin with a direct conclusion and explore its supporting data.','questions'],['Data and targets','Four stock indices use cumulative future log returns. Sunspot is a contrasting non-financial series with native future values. Four lags and two horizons define the supplied experiment.','learn'],['Comparing seven models','Linear and ARIMA provide interpretable and statistical baselines. Tree ensembles and two neural architectures add complexity. The TFT implementation is a simplified thesis adapter.','arena'],['Temporal evaluation','Rolling and expanding windows respect time order. Five purged and five embargoed rows separate training and testing; neural sequences add a 21-row warm-up.','protocol'],['Prediction results','All seven models have mean relative financial RMSE above persistence. Descriptive rankings differ from statistically significant comparisons on matched observations.','questions'],['Explanation evidence','Linear contributions, TreeSHAP, grouped KernelSHAP, native ARIMA counterfactuals, and TFT native signals explain different objects. Keep the method and its scope visible.','explain'],['Reproducibility','All eight notebooks retain their full code and stored outputs in the documentation portal. Source CSVs and figures can be opened and their SHA-256 hashes verified.','vault'],['Interpretation limits','An additive explanation can reconstruct a model output without identifying economic causes. Sparse regime strata and the small lag feature space constrain broader claims.','learn']];
  const rows=topics.filter(r=>!state.readerQuery||r.join(' ').toLowerCase().includes(state.readerQuery.toLowerCase()));
  return `<div class="callout mb"><strong>A public research companion.</strong> These are newly authored summaries. The complete thesis manuscript is excluded. The documentation contains the research methods, results, learning handbook, and all eight source notebooks.</div><div class="grid2">${rows.map(([title,text,route],i)=>`<article class="panel panel-pad"><span class="eyebrow">STUDY GUIDE / ${String(i+1).padStart(2,'0')}</span><h3 class="mt">${highlight(title,state.readerQuery)}</h3><p class="small mt">${highlight(text,state.readerQuery)}</p><button class="quiet source-link mt" data-route="${route}">Explore this topic →</button></article>`).join('')||'<p>No guide topics match. Try the literature tab for authors and titles.</p>'}</div>`;
}
function literatureBody(){
  const rows=guideData.references.map((r,i)=>({...r,index:i+1,topic:referenceTopic(r)})).filter(r=>(state.litCategory==='All'||r.topic===state.litCategory)&&(!state.readerQuery||r.text.toLowerCase().includes(state.readerQuery.toLowerCase())));
  return `<div class="between wrap mb"><div class="filter-pills">${['All','Forecasting','Explainability','Evaluation & methods'].map(c=>`<button class="filter-pill ${state.litCategory===c?'active':''}" data-lit-category="${c}">${c}</button>`).join('')}</div><span class="tiny muted">${rows.length} / ${guideData.references.length} references</span></div><div class="callout mb"><strong>The literature covered by the thesis.</strong> Citation text and links are preserved from the submitted bibliography. Topic labels help navigation. External full texts are not included in the package.</div><div class="grid2">${rows.map(r=>`<article class="panel reference-card"><div class="between"><span class="reference-number">REFERENCE / ${String(r.index).padStart(2,'0')}</span><span class="pill">${r.topic}</span></div><p>${highlight(r.text,state.readerQuery)}</p>${r.urls.length?`<a class="quiet" target="_blank" rel="noopener noreferrer" href="${esc(r.urls[0])}">Open cited source ${icon('external')}</a>`:'<span class="tiny faint">No external link supplied in the bibliography.</span>'}</article>`).join('')||'<div class="empty-state"><h2>No references match</h2><p>Try another author or topic.</p></div>'}</div>`;
}

function vaultView(){
  const modes=`<div class="segmented"><button data-vault-mode="files" class="${state.vaultMode==='files'?'active':''}">All files</button><button data-vault-mode="figures" class="${state.vaultMode==='figures'?'active':''}">Figure gallery · 32</button></div>`;
  return header('09 / TRACE IT TO THE SOURCE','The Evidence Vault','Original files, stored notebook outputs, complete result tables, and integrity hashes. Trace each displayed result back to its submitted research export.',button('Provenance & integrity','provenance','shield'))+
    `<div class="vault-summary"><div class="panel vault-stat"><strong>${manifest.files}</strong><span>original research files<br>preserved byte for byte</span></div><div class="panel vault-stat"><strong>8</strong><span>executed notebooks<br>stored code & outputs</span></div><div class="panel vault-stat"><strong>32</strong><span>original figures<br>stored model studies</span></div></div><div class="vault-toolbar"><input id="vault-search" type="search" value="${esc(state.vaultQuery)}" placeholder="Search filenames, paths, or file types…" aria-label="Search evidence files">${modes}</div><div class="between wrap mb"><div class="filter-pills">${['All','Notebooks','Data & tables','Figures','Audit & evidence'].map(c=>`<button class="filter-pill ${state.vaultCategory===c?'active':''}" data-vault-category="${esc(c)}">${c}</button>`).join('')}</div></div><div id="vault-results">${vaultResults()}</div><div class="callout mt"><strong>Source precedence matters.</strong> Charts use the corrected submission CSV exports. All eight executed notebooks and their stored outputs are available in the documentation portal. The complete thesis manuscript is excluded from this public package.</div>`;
}
function filteredFiles(){const q=state.vaultQuery.toLowerCase();return inventory.filter(f=>(state.vaultMode==='figures'?f.category==='Figures':state.vaultCategory==='All'||f.category===state.vaultCategory)&&(!q||(f.name+' '+f.path+' '+f.ext+' '+f.category).toLowerCase().includes(q)));}
function figureTitle(f){return f.name.replace(/\.png$/,'').replace(/^Figure_(\d+)_(\d+)_/,'Figure $1.$2 · ').replace(/^\d+_/, '').replaceAll('_',' ');}
function vaultResults(){
  const rows=filteredFiles(),per=state.vaultMode==='figures'?18:35,pages=Math.max(1,Math.ceil(rows.length/per));state.vaultPage=clamp(state.vaultPage,0,pages-1);const page=rows.slice(state.vaultPage*per,(state.vaultPage+1)*per);
  if(!rows.length)return `<div class="panel empty-state"><h2>No files match</h2><p>Try another filename or select a different category.</p></div>`;
  const body=state.vaultMode==='figures'?`<div class="figure-grid">${page.map(f=>`<article class="figure-card" role="button" tabindex="0" data-file="${esc(f.path)}"><img src="${urlPath(f.path)}" loading="lazy" alt="${esc(figureTitle(f))}"><div class="figure-caption"><h3>${esc(figureTitle(f))}</h3><p>${esc(f.path.includes('/cross_model/')?'Cross-model synthesis':f.path.includes('/figures/figures/')?'Thesis figure':f.path.split('/')[2])} · ${sizeLabel(f.bytes)}</p></div></article>`).join('')}</div>`:
    `<div class="panel"><div class="table-wrap"><table class="file-table"><thead><tr><th>Research artifact</th><th>Type</th><th class="numeric">Size</th><th></th></tr></thead><tbody>${page.map(f=>`<tr><td><div class="file-name"><span class="file-icon">${icon(f.ext==='ipynb'?'code':f.category==='Figures'?'scan':f.category==='Manuscripts'?'book':'file')}</span><div class="file-info"><button class="quiet" data-file="${esc(f.path)}"><strong>${esc(f.name)}</strong></button><small>${esc(f.path.replace(/^research\//,''))}${f.legacy?' · LEGACY MATERIAL':''}</small></div></div></td><td><span class="pill">${f.ext.toUpperCase()}</span></td><td class="numeric muted">${sizeLabel(f.bytes)}</td><td><div class="file-actions"><button class="icon-btn" data-file="${esc(f.path)}" title="Preview ${esc(f.name)}" aria-label="Preview ${esc(f.name)}">${icon('eye')}</button><a class="icon-btn" href="${urlPath(f.path)}" download title="Download ${esc(f.name)}" aria-label="Download ${esc(f.name)}">${icon('download')}</a></div></td></tr>`).join('')}</tbody></table></div></div>`;
  return body+`<div class="pagination"><span>${state.vaultPage*per+1}–${Math.min(rows.length,(state.vaultPage+1)*per)} of ${rows.length} files</span><div class="flex"><button class="icon-btn" data-action="vault-prev" aria-label="Previous file page" ${state.vaultPage===0?'disabled':''}>${icon('back')}</button><span>${state.vaultPage+1} / ${pages}</span><button class="icon-btn" data-action="vault-next" aria-label="Next file page" ${state.vaultPage===pages-1?'disabled':''}>${icon('next')}</button></div></div>`;
}

function showModal(title,body,actions=''){
  $('#modal-content').innerHTML=`<div class="modal-header"><h2>${esc(title)}</h2><div class="flex">${actions}<button class="icon-btn" data-action="close-modal" aria-label="Close dialog">${icon('close')}</button></div></div><div class="modal-body">${body}</div>`;
  if(!$('#modal').open)$('#modal').showModal();$('#modal').scrollTop=0;
}
function safeHTML(html){
  const template=document.createElement('template');template.innerHTML=html;
  const allowed=new Set(['P','BR','DIV','SPAN','H1','H2','H3','H4','H5','H6','UL','OL','LI','STRONG','EM','B','I','DEL','S','CODE','PRE','BLOCKQUOTE','TABLE','THEAD','TBODY','TR','TH','TD','A','IMG','SUP','SUB','HR']);
  for(const node of [...template.content.querySelectorAll('*')]){
    if(!allowed.has(node.tagName)){node.remove();continue;}
    for(const attr of [...node.attributes]){
      const name=attr.name,v=attr.value;
      if(name==='href'&&node.tagName==='A'&&/^(https?:\/\/|mailto:|#)/i.test(v))continue;
      if(name==='src'&&node.tagName==='IMG'&&/^data:image\/(png|jpeg|gif);base64,/i.test(v))continue;
      if(['colspan','rowspan'].includes(name)&&/^\d+$/.test(v))continue;
      if(name==='alt')continue;node.removeAttribute(name);
    }
    if(node.tagName==='A'){node.rel='noopener noreferrer';node.target='_blank';}
  }
  return template.innerHTML;
}
function markdown(text){return safeHTML(marked.parse(text));}
let fileRenderId=0;
async function openFile(path){
  const f=inventory.find(f=>f.path===path);if(!f){toast('This source is not in the research inventory.');return;}
  const id=++fileRenderId;csvPreview=null;
  const actions=`<a class="btn small" href="${urlPath(f.path)}" download>Download original ${icon('download')}</a>`;
  const provenance=`<div class="file-provenance"><span>${f.ext.toUpperCase()} / ${sizeLabel(f.bytes)}</span><code>SHA-256 ${f.sha256}</code><button class="quiet" data-verify-file="${esc(f.path)}">Verify file </button></div>${f.legacy?'<div class="notice mb">Source evidence. Use the corrected executed notebooks and CSV exports for the final experiment.</div>':''}`;
  showModal(f.name,provenance+'<div class="loading-inline">Opening original research artifact…</div>',actions);
  try{
    let content='';
    if(f.ext==='png')content=`<img src="${urlPath(f.path)}" alt="${esc(figureTitle(f))}">`;

    else if(f.ext==='csv'){
      const response=await fetch(urlPath(f.path));if(!response.ok)throw new Error('Source returned '+response.status);const reader=response.body.getReader();const decoder=new TextDecoder();let text='',done=false;
      while(!done&&text.length<1200000){const part=await reader.read();done=part.done;if(part.value)text+=decoder.decode(part.value,{stream:!done});}
      if(!done){await reader.cancel();text=text.slice(0,text.lastIndexOf('\n'));}
      const rows=parseCSV(text,2001);csvPreview={file:f,headers:rows[0]||[],rows:rows.slice(1,2001),page:0,query:'',partial:!done||rows.length>=2001};
      content=`<div class="between wrap mb"><p class="tiny">${csvPreview.partial?'Preview is capped at the first 2,000 rows. Download the original CSV for all rows.':'Complete CSV loaded for preview.'}</p><input id="csv-search" type="search" placeholder="Filter loaded rows…" aria-label="Filter CSV preview rows"></div><div id="csv-preview">${csvTable()}</div>`;
    }else{
      const response=await fetch(urlPath(f.path));if(!response.ok)throw new Error('Source returned '+response.status);const text=await response.text();
      if(f.ext==='ipynb')content=notebookHTML(JSON.parse(text));
      else if(f.ext==='md')content=`<div class="prose">${markdown(text)}</div>`;
      else content=`<pre class="code-block">${esc(f.ext==='json'?JSON.stringify(JSON.parse(text),null,2):text)}</pre>`;
    }
    if(id!==fileRenderId||!$('#modal').open)return;showModal(f.name,provenance+content,actions);
  }catch(e){if(id===fileRenderId)showModal(f.name,provenance+`<div class="notice">Preview could not be loaded: ${esc(e.message)}. The original download is still available above.</div>`,actions);}
}
function csvTable(){
  const c=csvPreview;if(!c)return '';const rows=c.rows.filter(r=>!c.query||r.join(' ').toLowerCase().includes(c.query.toLowerCase())),pages=Math.max(1,Math.ceil(rows.length/40));c.page=clamp(c.page,0,pages-1);
  return `<div class="table-wrap"><table><thead><tr>${c.headers.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.slice(c.page*40,(c.page+1)*40).map(row=>`<tr>${c.headers.map((_,i)=>`<td class="mono tiny">${esc(row[i])}</td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="pagination"><span>${rows.length} loaded rows match ${c.partial?'· capped preview':''}</span><div class="flex"><button class="icon-btn" data-action="csv-prev" aria-label="Previous CSV page" ${c.page===0?'disabled':''}>${icon('back')}</button><span>${c.page+1} / ${pages}</span><button class="icon-btn" data-action="csv-next" aria-label="Next CSV page" ${c.page===pages-1?'disabled':''}>${icon('next')}</button></div></div>`;
}
function notebookHTML(nb){
  const cells=nb.cells||[],code=cells.filter(c=>c.cell_type==='code'),errors=code.flatMap(c=>c.outputs||[]).filter(o=>o.output_type==='error').length;
  const textOf=x=>Array.isArray(x)?x.join(''):String(x??'');
  const outputs=out=>out.map(o=>{if(o.output_type==='stream')return `<pre>${esc(textOf(o.text))}</pre>`;if(o.output_type==='error')return `<pre>${esc(textOf(o.traceback))}</pre>`;if(o.data?.['image/png'])return `<img loading="lazy" src="data:image/png;base64,${textOf(o.data['image/png']).replace(/\s/g,'')}" alt="Stored figure from the executed notebook">`;if(o.data?.['text/html'])return safeHTML(textOf(o.data['text/html']));if(o.data?.['text/plain'])return `<pre>${esc(textOf(o.data['text/plain']))}</pre>`;return '';}).join('');
  return `<div class="callout mb"><strong>Stored execution, inspectable in your browser.</strong> ${cells.length} cells · ${code.length} code cells · ${errors} stored errors. This viewer displays the supplied code and recorded outputs. To execute or change the full experiment, download the notebook and run it in Jupyter.</div>${cells.map((c,i)=>`<section class="notebook-cell"><div class="cell-meta"><span>CELL ${String(i+1).padStart(2,'0')}</span><span>${c.cell_type.toUpperCase()}${c.cell_type==='code'?' / In ['+esc(c.execution_count??' ')+']':''}</span></div>${c.cell_type==='markdown'?`<div class="prose">${markdown(textOf(c.source))}</div>`:c.cell_type==='code'?`<details><summary>Inspect code (${textOf(c.source).split('\n').length} lines)</summary><pre class="code-block">${esc(textOf(c.source))}</pre></details><div class="notebook-output">${outputs(c.outputs||[])}</div>`:`<pre>${esc(textOf(c.source))}</pre>`}</section>`).join('')}`;
}
async function verifyFile(path){const f=inventory.find(f=>f.path===path);if(!f)return;try{if(!crypto.subtle)throw new Error('SHA-256 verification needs HTTPS or localhost.');toast('Checking original file bytes…');const r=await fetch(urlPath(path));if(!r.ok)throw new Error('Source returned '+r.status);const bytes=await r.arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',bytes),hash=[...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('');toast(hash===f.sha256?'SHA-256 verified: original file bytes match.':'Integrity mismatch: file bytes differ from the compiled inventory.');}catch(e){toast(e.message);}}
async function showProvenance(){
  const p=await getJSON('data/provenance.json');
  showModal('Research provenance & integrity',`<div class="grid2"><div><div class="eyebrow">CANONICAL EVIDENCE</div><h3 style="margin:15px 0">A fixed, auditable research snapshot</h3><p class="small">${esc(manifest.source_note)}</p><div class="keyvals"><div class="keyval"><span>Original files</span><strong>${manifest.files}</strong></div><div class="keyval"><span>Native predictions</span><strong>${fmt(manifest.native_predictions,0)}</strong></div><div class="keyval"><span>Matched predictions</span><strong>${fmt(manifest.matched_predictions,0)}</strong></div><div class="keyval"><span>Attribution rows</span><strong>${fmt(manifest.attribution_rows,0)}</strong></div><div class="keyval"><span>Reconstructed OLS forecasts verified</span><strong>30,240</strong></div></div></div><div><div class="eyebrow">RECORDED EXECUTION ENVIRONMENT</div><pre>${esc(JSON.stringify(p.Linear.environment,null,2))}</pre></div></div><div class="callout mt"><strong>Verification scope.</strong> Original hashes, notebook stored outputs, generated data round-trips, and metrics are checked by the included verification tool. This companion checks the supplied prediction and attribution exports. It does not claim to have rerun all seven training pipelines.</div><div class="flex wrap mt"><button class="btn" data-file="research/results/cross_model/evidence/validation.json">Open stored validation ${icon('arrow')}</button><button class="btn" data-file="research/results/cross_model/evidence/analysis_summary.json">Open analysis summary ${icon('arrow')}</button></div>`);
}

function download(name,text,type='text/plain'){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportReplay(){
  if(state.replayTab==='scenarios'){
    if(!activeScenario?.exportMatches())toast('Choose a reference date with eligible matches first.');
    return;
  }
  const d=plotData;
  if(state.replayTab==='blend'){document.querySelector('[data-ac-export=blend]')?.click();return;}
  if(state.replayTab==='whatif'){
    const {fit,features,deltas}=whatIf,edited=features.map((v,j)=>v+deltas[j]*fit.scale[j]);
    download('linear-what-if-HYPOTHETICAL.csv',csvText([['mode','series','horizon','scheme','origin_date','feature','observed_input','edited_input','training_mean','training_std','standardized_weight'],...LAGS.map((l,j)=>['hypothetical_input_intervention',state.series,state.h,state.scheme,d.origin_date[state.cursor],'lag_'+l,features[j],edited[j],fit.mean[j],fit.scale[j],fit.weights[j]]),['hypothetical_model_output',linearPredict(fit,edited)]]),'text/csv');return;
  }
  const {start,end}=visibleWindow(d,state.cursor,state.span),rows=[['origin_date','target_date','series','horizon','scheme','matched_fold','y_true','persistence',...MODELS.map(m=>m+'_forecast')]];
  for(let i=start;i<end;i++)rows.push([d.origin_date[i],d.target_date[i],state.series,state.h,state.scheme,d.matched_fold[i],d.y_true[i],d.random_walk_pred[i],...MODELS.map(m=>d.predictions[m][i])]);
  download(`forecast-atlas-${state.series}-h${state.h}-${state.scheme}-window.csv`,csvText(rows),'text/csv');
}
function exportChart(){const node=$('#forecast-svg');if(!node){toast('Open forecast replay to export the current chart.');return;}const cloned=node.cloneNode(true);cloned.setAttribute('xmlns','http://www.w3.org/2000/svg');cloned.querySelector('#plot-pointer')?.remove();const colors=getComputedStyle(document.documentElement);let html=cloned.outerHTML.replace(/var\((--[^)]+)\)/g,(_,v)=>colors.getPropertyValue(v).trim());html=html.replace(/(<svg[^>]*>)/,`$1<rect width="${node.viewBox.baseVal.width}" height="${node.viewBox.baseVal.height}" fill="${colors.getPropertyValue('--surface').trim()}"/>`);html=html.replaceAll('class="chart-axis"','font-family="monospace" font-size="9" fill="'+colors.getPropertyValue('--faint').trim()+'"').replaceAll('class="chart-grid"','stroke="'+colors.getPropertyValue('--border').trim()+'" stroke-width="0.7"');download(`forecast-atlas-${state.series}-${state.model}.svg`,html,'image/svg+xml');}
function exportArena(){const rows=state.scope==='cell'?summary.matched_metrics.filter(r=>r.series===state.series&&r.horizon===state.h&&r.scheme===state.scheme):summary.financial_model_summary;const fields=Object.keys(rows[0]);download('forecast-atlas-model-comparison.csv',csvText([fields,...rows.map(r=>fields.map(k=>r[k]))]),'text/csv');}
const TOUR=[
  ['home','The research, at a glance','Begin with the finding that complexity does not guarantee predictive skill. These summaries come from corrected result tables.'],
  ['replay','Move through the experiment','Select a model, dataset, horizon, and window. Play the timeline; the chart and visible-window metrics update from recorded forecasts.'],
  ['replay','Try a real what-if','The Linear lab uses reconstructed OLS coefficients checked against every native Linear forecast. Sliders change actual model inputs.','whatif'],
  ['arena','Separate ranking from inference','Compare identical test origins. The pairwise matrix links to adjusted tests; a descriptive lead is not a significant benchmark win.'],
  ['explain','Interrogate the explanation','Scrub individual probes, switch explained folds, inspect regime stability, and keep faithfulness diagnostics separate.'],
  ['questions','Start with the research questions','Select RQ1, RQ2, or RQ3. The study conclusion appears immediately, followed by interactive evidence.'],
  ['learn','Build an explanation you understand','Study seven models, experiment with equations, test your knowledge, and browse all 47 references.'],
  ['vault','Every claim has a trail','Preview executed notebooks and source tables, download original files, or verify their SHA-256 hashes in your browser.']
];
function showTourStep(){if(tourStep<0||tourStep>=TOUR.length){$('#tour').hidden=true;tourStep=-1;return;}const [view,title,body,tab]=TOUR[tourStep];navigate(view,view==='replay'?{replayTab:tab||'replay',model:'Linear'}:{});$('#tour').hidden=false;$('#tour').innerHTML=`<div class="between"><span class="eyebrow">GUIDED TOUR / ${tourStep+1} OF ${TOUR.length}</span><button class="icon-btn" data-action="tour-close" aria-label="Close guided tour">${icon('close')}</button></div><h3>${title}</h3><p>${body}</p><div class="between"><button class="quiet small" data-action="tour-prev" ${tourStep===0?'disabled':''}>← Back</button><button class="btn primary small" data-action="tour-next">${tourStep===TOUR.length-1?'Finish tour':'Next stop'} ${icon('arrow')}</button></div>`;}
async function prepareOffline(){
  if(!('caches' in window)||!('serviceWorker' in navigator)){showModal('Offline availability','<p>Offline caching is available on HTTPS or localhost in browsers that support service workers. The complete downloaded folder remains usable with the included local launcher.</p>');return;}
  showModal('Keep the research interface offline',`<p class="small">Save the simulation, learning handbook, full notebook documentation, and interactive datasets on this device. Original research CSV downloads are cached as you open them.</p><div class="offline-progress"><span id="offline-progress" style="width:0"></span></div><p id="offline-status" class="tiny">Preparing the research interface…</p>`);
  try{
    const registration=await navigator.serviceWorker.register(new URL('sw.js',APP_ROOT),{scope:APP_ROOT.href,updateViaCache:'none'});await navigator.serviceWorker.ready;
    const pack=await getJSON('cache-manifest.json'),store=await caches.open(`forecast-atlas@${APP_ROOT.pathname}#${pack.revision}`);let count=0;
    for(let i=0;i<pack.files.length;i+=4){const group=pack.files.slice(i,i+4);await Promise.all(group.map(async f=>{await store.add(new URL(f,APP_ROOT));count++;}));if($('#offline-progress'))$('#offline-progress').style.width=(count/pack.files.length*100)+'%';if($('#offline-status'))$('#offline-status').textContent=`Saved ${count} / ${pack.files.length} interface files`;
    }
    if($('#offline-status'))$('#offline-status').textContent='Ready. Reopen this page at least once online after installing, then the cached research interface is available without a connection.';toast('Interactive research data are available offline on this device.');
  }catch(e){if($('#offline-status'))$('#offline-status').textContent='The cache could not be completed: '+e.message;toast('Offline cache was not completed. You can still use the downloaded app locally.');}
}

document.addEventListener('click',async event=>{
  const target=event.target.closest('[data-route],[data-action],[data-model],[data-model-home],[data-file],[data-scope],[data-pair-a],[data-replay-tab],[data-reader-tab],[data-chapter],[data-search-chapter],[data-reader-anchor],[data-lit-category],[data-vault-category],[data-vault-mode],[data-verify-file],[data-image]');if(!target||target.disabled)return;
  const d=target.dataset;
  try{
    if(d.route){event.preventDefault();navigate(d.route);return;}
    if(d.modelHome){navigate('replay',{model:d.modelHome,replayTab:'replay'});return;}
    if(d.model){stopPlayback();state.model=d.model;state.fold=0;state.probe=0;if(state.view==='home')navigate('replay',{replayTab:'replay'});else{remember();render();}return;}
    if(d.file){await openFile(d.file);return;}
    if(d.verifyFile){await verifyFile(d.verifyFile);return;}

    if(d.scope){state.scope=d.scope;render();return;}
    if(d.pairA){showPair(d.pairA,d.pairB);return;}
    if(d.replayTab){stopPlayback();state.replayTab=d.replayTab;if(d.replayTab==='whatif')state.model='Linear';remember();render();return;}
    if(d.readerTab){state.readerTab=d.readerTab;state.readerQuery='';render();return;}
    if(d.litCategory){state.litCategory=d.litCategory;$('#reader-body').innerHTML=readerBody();return;}
    if(d.vaultCategory){state.vaultCategory=d.vaultCategory;state.vaultPage=0;state.vaultMode='files';render();return;}
    if(d.vaultMode){state.vaultMode=d.vaultMode;state.vaultPage=0;render();return;}
    const a=d.action;
    if(a==='go-replay')navigate('replay');
    else if(a==='open-scenarios')navigate('replay',{replayTab:'scenarios'});
    else if(a==='retry'){if(manifest)render();else boot();}
    else if(a==='menu'){$('#sidebar').classList.toggle('open');$('#sidebar-scrim').classList.toggle('open');}
    else if(a==='theme'){
      const root=document.documentElement;
      const theme=root.dataset.theme==='light'?'dark':'light';
      root.dataset.theme=theme;
      document.querySelector('meta[name="theme-color"]').content=getComputedStyle(root).getPropertyValue('--bg').trim();
      try{localStorage.setItem('forecast-atlas-lab-theme',theme);}catch{}
    }
    else if(a==='search'){navigate('vault');setTimeout(()=>$('#vault-search')?.focus(),0);}
    else if(a==='share'){remember();try{await navigator.clipboard.writeText(location.href);toast('Link to this experiment copied.');}catch{showModal('Link to this experiment',`<input style="width:100%" type="text" readonly aria-label="Current experiment link" value="${esc(location.href)}">`);}}
    else if(a==='play'){if(playing)stopPlayback();else{if(state.cursor>=923)state.cursor=0;playing=true;$('#play-button').innerHTML=icon('pause');$('#play-button').setAttribute('aria-label','Pause forecast replay');timer=setTimeout(playTick,Number($('#play-speed')?.value||400));}}
    else if(a==='step-back'||a==='step-next'){stopPlayback();state.cursor=clamp(state.cursor+(a==='step-back'?-1:1),0,923);updateReplay();}
    else if(a==='reset-whatif'){whatIf.deltas=[0,0,0,0];document.querySelectorAll('.feature-range').forEach(el=>el.value=0);updateWhatIf();}
    else if(a==='export-replay')exportReplay();
    else if(a==='export-chart')exportChart();
    else if(a==='export-arena')exportArena();
    else if(a==='close-modal'){fileRenderId++;$('#modal').close();}
    else if(a==='open-reader'){$('#modal').close();navigate('reader');}
    else if(a==='vault-prev'||a==='vault-next'){state.vaultPage+=a==='vault-next'?1:-1;$('#vault-results').innerHTML=vaultResults();$('#vault-results').scrollIntoView({behavior:'smooth',block:'start'});}
    else if(a==='csv-prev'||a==='csv-next'){if(csvPreview){csvPreview.page+=a==='csv-next'?1:-1;$('#csv-preview').innerHTML=csvTable();}}
    else if(a==='provenance')await showProvenance();
    else if(a==='tour'){tourStep=0;showTourStep();}
    else if(a==='tour-next'){tourStep++;showTourStep();}
    else if(a==='tour-prev'){tourStep--;showTourStep();}
    else if(a==='tour-close'){tourStep=-1;$('#tour').hidden=true;}
    else if(a==='offline')await prepareOffline();
  }catch(e){console.error(e);toast('The action could not be completed: '+e.message);}
});
document.addEventListener('change',event=>{
  const el=event.target;
  if(el.dataset.global){stopPlayback();const key=el.dataset.global;state[key]=key==='h'?Number(el.value):el.value;state.fold=0;state.probe=0;state.cursor=175;remember();render();}
  else if(el.id==='compare-model'){state.compare=el.value;updateReplay();}
  else if(el.id==='all-models'){state.all=el.checked;updateReplay();}
  else if(el.id==='chart-mode'){state.mode=el.value;updateReplay();}
  else if(el.id==='attribution-fold'){state.fold=Number(el.value);updateProbe();}
});
let searchDebounce;
document.addEventListener('input',event=>{
  const el=event.target;
  if(el.id==='replay-range'){stopPlayback();state.cursor=Number(el.value);updateReplay();}
  else if(el.id==='window-range'){state.span=Number(el.value);updateReplay();}
  else if(el.id==='probe-range'){state.probe=Number(el.value);updateProbe();}
  else if(el.id==='timeline-range'){state.fold=Number(el.value);updateTimeline();}
  else if(el.dataset.feature!==undefined){whatIf.deltas[Number(el.dataset.feature)]=Number(el.value);updateWhatIf();}
  else if(el.id==='whatif-origin'){state.cursor=Number(el.value);rawSeries().then(initWhatIf).catch(e=>toast(e.message));}
  else if(el.id==='vault-search'){state.vaultQuery=el.value;state.vaultPage=0;$('#vault-results').innerHTML=vaultResults();}
  else if(el.id==='reader-search'){state.readerQuery=el.value;clearTimeout(searchDebounce);searchDebounce=setTimeout(()=>{if($('#reader-body'))$('#reader-body').innerHTML=readerBody();},160);}
  else if(el.id==='csv-search'&&csvPreview){csvPreview.query=el.value;csvPreview.page=0;$('#csv-preview').innerHTML=csvTable();}
});
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&e.target.matches('[role="link"],[role="button"]')){e.target.click();return;}
  if(e.target.matches('input,select,textarea')||$('#modal').open)return;
  if(e.key==='/'){e.preventDefault();navigate('vault');setTimeout(()=>$('#vault-search')?.focus(),0);}
  if(state.view==='replay'&&state.replayTab==='replay'){
    if(e.code==='Space'&&!e.target.matches('button,a,summary')){e.preventDefault();$('#play-button')?.click();}
    else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();stopPlayback();state.cursor=clamp(state.cursor+(e.key==='ArrowLeft'?-1:1),0,923);updateReplay();}
  }
});
$('#modal').addEventListener('click',e=>{if(e.target===$('#modal')){$('#modal').close();fileRenderId++;}});
window.addEventListener('hashchange',()=>{stopPlayback();readHash();render();});
window.addEventListener('pagehide',stopPlayback);
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlayback();});

async function boot(){
  icons();
  readHash();
  if(location.protocol==='file:'){setMain(`<div class="browser-error"><h1>Open the local research app</h1><p>The interactive datasets need a local web server. Double-click <strong>START_HERE.command</strong> on a Mac, or run this command in the master-thesis folder:</p><code>python3 -m http.server 8000</code><p class="mt">Then open <a href="http://localhost:8000">http://localhost:8000</a>. The published GitHub Pages version works directly in the browser.</p></div>`);return;}
  try{[manifest,summary,inventory]=await Promise.all([getJSON('data/manifest.json'),getJSON('data/summary.json'),getJSON('data/inventory.json')]);setupAcademic({state,summary,getJSON,forecasts,attrs,rawSeries,header,contextBar,modelTabs,navigate,remember,toast,url:urlPath,exportRows:(name,rows)=>download(name,csvText(rows),'text/csv')});await render();}
  catch(e){setMain(`<div class="browser-error"><h1>The research data could not load</h1><p>${esc(e.message)}</p><p>Keep the complete folder structure together, including assets, data, and research.</p>${button('Try again','retry','reset')}</div>`);console.error(e);}
  if('serviceWorker' in navigator)navigator.serviceWorker.register(new URL('sw.js',APP_ROOT),{scope:APP_ROOT.href,updateViaCache:'none'}).catch(()=>{});
}
boot();
