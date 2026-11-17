import {esc,finite,fmt,pct,MODEL,visibleWindow} from './core.js';

export function sparkline(values,color='var(--accent2)',width=460,height=110){
  const clean=values.filter(finite);if(!clean.length)return '';
  const min=Math.min(...clean),max=Math.max(...clean),range=max-min||1;
  const points=values.map((v,i)=>`${(i/(values.length-1||1)*(width-4)+2).toFixed(2)},${(height-5-(v-min)/range*(height-12)).toFixed(2)}`).join(' ');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Recorded data trend"><polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`;
}

export function forecastChart(data,{model,cursor=175,span=120,compare='ARIMA',mode='return',all=false,compact=false}={}){
  const {start,end}=visibleWindow(data,cursor,span),n=end-start;
  const financial=data.series!=='monash_sunspot',isLevel=mode==='level'&&financial;
  const transform=(v,i)=>isLevel?data.origin_level[i]*Math.exp(v):v;
  const actual=data.y_true.slice(start,end).map((v,j)=>transform(v,start+j));
  const selected=data.predictions[model].slice(start,end).map((v,j)=>transform(v,start+j));
  const baseline=data.random_walk_pred.slice(start,end).map((v,j)=>transform(v,start+j));
  const comparison=compare&&compare!==model?data.predictions[compare].slice(start,end).map((v,j)=>transform(v,start+j)):[];
  const drawModels=all?Object.keys(data.predictions).filter(m=>m!==model&&m!==compare):[];
  const other=drawModels.map(m=>data.predictions[m].slice(start,end).map((v,j)=>transform(v,start+j)));
  const values=[...actual,...selected,...baseline,...comparison,...other.flat()];
  let min=Math.min(...values),max=Math.max(...values);const pad=(max-min||Math.max(.01,Math.abs(max)*.1))*.13;min-=pad;max+=pad;
  const W=compact?480:780,H=compact?320:310,left=compact?57:60,right=compact?15:24,top=15,bottom=37,pw=W-left-right,ph=H-top-bottom;
  const x=j=>left+(j/(n-1||1))*pw,y=v=>top+(max-v)/(max-min)*ph;
  const path=arr=>arr.map((v,j)=>`${j?'L':'M'}${x(j).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const axisValue=v=>financial&&!isLevel?pct(v,1):fmt(v,isLevel?0:1);
  let grid='';
  for(let j=0;j<5;j++){const v=min+(max-min)*j/4;grid+=`<line class="chart-grid" x1="${left}" y1="${y(v)}" x2="${W-right}" y2="${y(v)}"/><text class="chart-axis" x="${left-10}" y="${y(v)+3}" text-anchor="end">${axisValue(v)}</text>`;}
  const tickIndices=[...new Set(compact?[0,Math.floor((n-1)/2),n-1]:[0,Math.floor((n-1)/3),Math.floor((n-1)*2/3),n-1])];
  for(const j of tickIndices){grid+=`<text class="chart-axis" x="${x(j)}" y="${H-9}" text-anchor="${j===0?'start':j===n-1?'end':'middle'}">${esc(data.origin_date[start+j])}</text>`;}
  const fill=path(selected)+` L${x(n-1)},${y(isLevel?min:Math.max(min,Math.min(max,0)))} L${x(0)},${y(isLevel?min:Math.max(min,Math.min(max,0)))} Z`;
  if(compact)grid=grid.replaceAll('class="chart-axis"','class="chart-axis" style="font-size:12px"');
  return `<svg id="forecast-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="Recorded ${esc(model)} forecasts and realized ${isLevel?'target levels':financial?'log returns':'Sunspot values'}, ${n} matched origins" data-start="${start}" data-count="${n}" data-left="${left}" data-plotwidth="${pw}"><defs><linearGradient id="plot-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${MODEL[model].color}" stop-opacity=".13"/><stop offset="1" stop-color="${MODEL[model].color}" stop-opacity="0"/></linearGradient></defs>${grid}<path d="${fill}" fill="url(#plot-fill)"/>${other.map((a,j)=>`<path d="${path(a)}" stroke="${MODEL[drawModels[j]].color}" fill="none" stroke-width="1" opacity=".5"/>`).join('')}<path d="${path(baseline)}" stroke="var(--blue)" fill="none" stroke-width="1" stroke-dasharray="4 4" opacity=".65"/><path d="${path(actual)}" stroke="var(--muted)" fill="none" stroke-width="1.15" opacity=".75"/>${comparison.length?`<path d="${path(comparison)}" stroke="${MODEL[compare].color}" fill="none" stroke-width="1.4" opacity=".8"/>`:''}<path d="${path(selected)}" stroke="${MODEL[model].color}" fill="none" stroke-width="1.9"/><circle cx="${x(n-1)}" cy="${y(selected[n-1])}" r="4" fill="${MODEL[model].color}" stroke="var(--surface)" stroke-width="2"/><line x1="${x(n-1)}" x2="${x(n-1)}" y1="${top}" y2="${H-bottom}" stroke="var(--border)" stroke-dasharray="3 4"/><rect id="plot-pointer" x="${left}" y="${top}" width="${pw}" height="${ph}" fill="transparent"/></svg>`;
}

export function attributionBars(values,financial){
  const max=Math.max(...values.map(Math.abs),1e-15);
  return values.map((v,i)=>`<div class="attr-row"><span class="mono">lag_${[1,5,21,252][i]}</span><div class="attr-track"><span class="attr-bar" style="left:${v<0?50-Math.abs(v)/max*48:50}%;width:${Math.abs(v)/max*48}%;background:${v<0?'var(--red)':'var(--accent2)'}"></span></div><span class="attr-value ${v<0?'negative':'positive'}">${v>0?'+':''}${financial?pct(v,3):fmt(v,3)}</span></div>`).join('')+`<div class="attr-axis"><span>decreases output</span><span>0</span><span>increases output</span></div>`;
}
