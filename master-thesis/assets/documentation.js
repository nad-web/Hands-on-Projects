import {esc} from './core.js';
const ROOT=new URL('../',import.meta.url),$=s=>document.querySelector(s);
try{const saved=localStorage.getItem('forecast-atlas-theme');if(saved==='light'||saved==='dark')document.documentElement.dataset.theme=saved;}catch{}
document.addEventListener('click',event=>{
 const target=event.target.closest('[data-doc-theme],[data-expand-all],[data-collapse-all],[data-doc-print]');if(!target)return;
 if(target.hasAttribute('data-doc-theme')){const theme=document.documentElement.dataset.theme==='light'?'dark':'light';document.documentElement.dataset.theme=theme;try{localStorage.setItem('forecast-atlas-theme',theme);}catch{}}
 if(target.hasAttribute('data-expand-all'))document.querySelectorAll('[data-code-cell]').forEach(e=>e.open=true);
 if(target.hasAttribute('data-collapse-all'))document.querySelectorAll('[data-code-cell]').forEach(e=>e.open=false);
 if(target.hasAttribute('data-doc-print'))window.print();
});
document.addEventListener('change',event=>{if(event.target.matches('[data-doc-nav]'))location.href=event.target.value;});
const notebookSearch=$('#notebook-search');
notebookSearch?.addEventListener('input',()=>{
 const query=notebookSearch.value.trim().toLowerCase(),cells=[...document.querySelectorAll('[data-notebook-cell]')];let matches=0;
 for(const cell of cells){const yes=!query||cell.textContent.toLowerCase().includes(query);cell.hidden=!yes;if(yes){matches++;if(query)cell.querySelectorAll('details').forEach(d=>d.open=true);}}
 $('#notebook-search-status').textContent=query?`${matches} / ${cells.length} cells match “${notebookSearch.value}”`:`${cells.length} cells · complete notebook content`;
});
let indexPromise,sequence=0;
$('#docs-global-search')?.addEventListener('input',async event=>{
 const query=event.target.value.trim().toLowerCase(),id=++sequence,results=$('#docs-search-results'),overview=$('#docs-overview-content');
 if(!query){results.hidden=true;overview.hidden=false;return;}results.hidden=false;overview.hidden=true;results.innerHTML='<div class="loading-inline">Searching the complete notebook text…</div>';
 try{indexPromise??=fetch(new URL('data/docs-search.json',ROOT)).then(r=>{if(!r.ok)throw new Error('Search index unavailable');return r.json();});const index=await indexPromise;if(id!==sequence)return;
   const matches=index.filter(row=>row.text.toLowerCase().includes(query));
   results.innerHTML=`<div class="section-head"><h2>${matches.length} matching notebook cells</h2><span class="tiny muted">Complete code, explanations, and recorded text outputs</span></div><div class="doc-search-results">${matches.slice(0,80).map(row=>{const ix=row.text.toLowerCase().indexOf(query),start=Math.max(0,ix-80),snippet=(start?'…':'')+row.text.slice(start,start+330)+(row.text.length>start+330?'…':'');return `<a class="panel doc-search-result" href="notebooks/${row.slug}.html#cell-${row.cell}"><span>NOTEBOOK ${row.number} / CELL ${row.cell} / ${row.type.toUpperCase()}</span><h3>${esc(row.notebook)}</h3><p>${esc(snippet)}</p><span>Open complete cell →</span></a>`;}).join('')||'<div class="panel empty-state"><h2>No matching notebook cell</h2><p>Try another model, function, equation term, or diagnostic.</p></div>'}</div>${matches.length>80?'<p class="tiny mt">Showing the first 80 matches. Refine the search to narrow the result.</p>':''}`;
 }catch(error){if(id===sequence)results.innerHTML=`<div class="notice">${esc(error.message)}. The complete notebook pages remain accessible from the sidebar.</div>`;}
});
function revealHash(){const id=decodeURIComponent(location.hash.slice(1));if(!id)return;const cell=document.getElementById(id);if(cell){cell.hidden=false;cell.querySelectorAll('details').forEach(d=>d.open=true);}}
revealHash();window.addEventListener('hashchange',revealHash);
let beforePrint=[],beforePrintCells=[];window.addEventListener('beforeprint',()=>{beforePrint=[...document.querySelectorAll('[data-code-cell]')].map(d=>[d,d.open]);beforePrintCells=[...document.querySelectorAll('[data-notebook-cell]')].map(c=>[c,c.hidden]);beforePrint.forEach(([d])=>d.open=true);beforePrintCells.forEach(([c])=>c.hidden=false);});window.addEventListener('afterprint',()=>{beforePrint.forEach(([d,open])=>d.open=open);beforePrintCells.forEach(([c,hidden])=>c.hidden=hidden);});
if('serviceWorker'in navigator&&location.protocol!=='file:')navigator.serviceWorker.register(new URL('sw.js',ROOT),{scope:ROOT.href,updateViaCache:'none'}).catch(()=>{});
