// Pure numeric helpers shared by the interface and the verification tests.
export const MODELS = ['Linear','ARIMA','RandomForest','XGBoost','LightGBM','LSTM','TFT'];
export const LAGS = [1,5,21,252];
export const MODEL = {
  Linear:{name:'Linear Regression',short:'Linear',family:'Interpretable baseline',color:'#b9efcc',method:'Exact linear SHAP',notebook:'01_Linear_Regression_Executed.ipynb',config:'Ordinary least squares on four standardized lagged inputs. Exact linear attributions explain the same fitted predictor.',params:[['Estimator','Ordinary least squares'],['Scaling','Training-only standardization'],['Forecast folds','6'],['Seeds','20261017']]},
  ARIMA:{name:'ARIMA',short:'ARIMA',family:'Statistical baseline',color:'#dec992',method:'Native lag counterfactuals',notebook:'02_ARIMA_Executed.ipynb',config:'AIC selects from seven ARIMA orders. Parameters stay fixed while the state updates with observed history at each origin. Financial forecasts sum the next h return forecasts.',params:[['Selection','Minimum training AIC'],['Candidate orders','7'],['State update','Observed raw rows'],['Seeds','20261017']]},
  RandomForest:{name:'Random Forest',short:'RF',family:'Tree ensemble',color:'#91b5e1',method:'TreeSHAP · full ensemble',notebook:'03_Random_Forest_Executed.ipynb',config:'Forecasts and TreeSHAP values average two independently seeded forests. Each forest contains 100 trees with depth at most 10 and at least 20 observations per leaf.',params:[['Trees / seed','100'],['Maximum depth','10'],['Minimum leaf','20'],['Seeds','20261017 / 20261018']]},
  XGBoost:{name:'XGBoost',short:'XGB',family:'Gradient-boosted trees',color:'#e4a588',method:'TreeSHAP · full ensemble',notebook:'04_XGBoost_Executed.ipynb',config:'Two-seed average with up to 200 boosting rounds, depth 6, and learning rate 0.05. Early stopping uses the final 10% of training rows (at least 50).',params:[['Maximum rounds','200'],['Learning rate','0.05'],['L1 / L2','0.1 / 1.0'],['Early stopping','25 rounds']]},
  LightGBM:{name:'LightGBM',short:'LGBM',family:'Gradient-boosted trees',color:'#91d4ca',method:'TreeSHAP · full ensemble',notebook:'05_LightGBM_Executed.ipynb',config:'Two-seed average with up to 200 rounds, maximum depth 6, learning rate 0.05, and row/column sampling 0.8. A preserved validation tail controls early stopping.',params:[['Maximum rounds','200'],['Learning rate','0.05'],['Row / column sampling','0.8 / 0.8'],['Early stopping','25 rounds']]},
  LSTM:{name:'LSTM',short:'LSTM',family:'Recurrent neural network',color:'#c2b6f0',method:'KernelSHAP · lag groups',notebook:'06_LSTM_Executed.ipynb',config:'Two recurrent layers with 64 hidden units process uninterrupted 21-step histories. KernelSHAP contributions are aggregated into the four lag groups.',params:[['Sequence length','21'],['Hidden units / layers','64 / 2'],['Maximum epochs','25 · patience 8'],['Seeds','20261017']]},
  TFT:{name:'Simplified TFT',short:'TFT',family:'Temporal Fusion Transformer adapter',color:'#e4a6c7',method:'KernelSHAP · lag groups',notebook:'07_Temporal_Fusion_Transformer_Executed.ipynb',config:'The thesis adapter combines variable selection, a 64-unit LSTM encoder, and four-head attention. Native variable-selection weights and attention signals are distinct from KernelSHAP.',params:[['Sequence / hidden size','21 / 64'],['Attention heads','4'],['Maximum epochs','25 · patience 8'],['Seeds','20261017']]}
};
export const SERIES = {sp500:'S&P 500',dax:'DAX',nasdaq:'NASDAQ Composite',djia:'Dow Jones',monash_sunspot:'Monash Sunspot'};
export const esc = s => String(s ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const urlPath = path => path.split('/').map(encodeURIComponent).join('/');
export const finite = x => typeof x==='number' && Number.isFinite(x);
export const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
export const mean = a => {a=a.filter(finite);return a.length ? a.reduce((x,y)=>x+y,0)/a.length:null;};
export const median = a => {a=a.filter(finite).slice().sort((x,y)=>x-y);const n=a.length;return n?(a[Math.floor((n-1)/2)]+a[Math.ceil((n-1)/2)])/2:null;};
export const fmt = (x,d=3) => finite(x)?x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
export const pct = (x,d=2) => finite(x)?`${fmt(x*100,d)}%`:'—';
export const num = x => finite(x)?Math.abs(x)>0&&Math.abs(x)<.00001?x.toExponential(2):fmt(x,5):'—';
export const dateLabel = d => d ? new Date(d+'T12:00:00Z').toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric',timeZone:'UTC'}) : '—';
export const sizeLabel = n => n<1024?`${n} B`:n<1048576?`${fmt(n/1024,0)} KB`:`${fmt(n/1048576,1)} MB`;
export function metrics(actual,pred,baseline,financial=true){
  if(!actual.length || actual.length!==pred.length || (baseline && actual.length!==baseline.length)) return {n:0,rmse:null,mae:null,r2:null,da:null,relative:null};
  const n=actual.length,avg=mean(actual);let mse=0,mae=0,sst=0,hits=0,bmse=0;
  for(let i=0;i<n;i++){mse+=(actual[i]-pred[i])**2;mae+=Math.abs(actual[i]-pred[i]);sst+=(actual[i]-avg)**2;hits+=Math.sign(actual[i])===Math.sign(pred[i]);if(baseline)bmse+=(actual[i]-baseline[i])**2;}
  return {n,rmse:Math.sqrt(mse/n),mae:mae/n,r2:sst?1-mse/sst:mse===0?1:0,da:financial?hits/n:null,relative:baseline&&bmse>0?Math.sqrt(mse/bmse):null};
}
export function visibleWindow(data,cursor,span){
  cursor=clamp(Math.round(cursor),0,data.y_true.length-1);let start=Math.max(0,cursor-span+1);
  while(start<cursor&&data.matched_fold[start]!==data.matched_fold[cursor])start++;
  return {start,end:cursor+1};
}
export function linearPredict(fit,features){return fit.intercept+features.reduce((sum,v,j)=>sum+fit.weights[j]*(v-fit.mean[j])/fit.scale[j],0);}
export function splitDesign(n,model,scheme,fold){
  const deep=model==='LSTM'||model==='TFT',folds=deep?4:6;
  fold=clamp(fold,0,folds-1);
  const testStart=scheme==='rolling'?n-folds*252+fold*252:510+fold*252;
  const trainEnd=testStart-10,trainStart=scheme==='rolling'?Math.max(0,trainEnd-500):0;
  return {folds,fold,trainStart,trainEnd,purgeStart:trainEnd,embargoStart:trainEnd+5,testStart,testEnd:Math.min(n,testStart+252),warmup:deep?21:0};
}
export function parseCSV(text,limit=Infinity){
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
    else if(c===','&&!quoted){row.push(field);field='';}
    else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(field);if(row.some(x=>x!==''))rows.push(row);row=[];field='';if(rows.length>=limit)return rows;}
    else field+=c;
  }
  if(field||row.length){row.push(field);rows.push(row);}return rows;
}
export function csvText(rows){return rows.map(r=>r.map(v=>`"${String(v??'').replace(/"/g,'""')}"`).join(',')).join('\r\n');}
