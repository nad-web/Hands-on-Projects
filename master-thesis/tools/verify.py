#!/usr/bin/env python3
"""Independent standard-library checks of preserved files and browser data.

Run from anywhere: python3 tools/verify.py
No model is trained, and no source research file is modified.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import sys

APP = Path(__file__).resolve().parents[1]
MODELS = ['Linear','ARIMA','RandomForest','XGBoost','LightGBM','LSTM','TFT']

def load(path): return json.loads((APP/path).read_text())
def rows(path):
    with (APP/path).open(newline='') as f: return list(csv.DictReader(f))
def near(a,b,tol=2e-11):
    assert abs(a-b) <= tol*max(1,abs(b)), (a,b)
def calc(actual,pred,baseline):
    n=len(actual);average=sum(actual)/n
    sse=sum((a-p)**2 for a,p in zip(actual,pred));sst=sum((a-average)**2 for a in actual)
    bse=sum((a-b)**2 for a,b in zip(actual,baseline))
    return {'rmse':math.sqrt(sse/n),'mae':sum(abs(a-p) for a,p in zip(actual,pred))/n,
            'r2':1-sse/sst if sst else (1 if sse==0 else 0),
            'da':sum((a>0)-(a<0)==(p>0)-(p<0) for a,p in zip(actual,pred))/n,
            'rmse_relative_to_persistence':math.sqrt(sse/bse)}

def main():
    inv=load('data/inventory.json');manifest=load('data/manifest.json');summary=load('data/summary.json')
    assert len(inv)==manifest['files']==261
    for f in inv:
        p=(APP/f['path']).resolve();assert p.is_relative_to(APP.resolve()) and p.is_file(), f['path']
        assert p.stat().st_size==f['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'], f['path']
    print('PASS — 261 original files match the inventory SHA-256 hashes.')

    sources=rows('research/results/cross_model/tables/matched_observation_predictions.csv')
    datasets={}
    for row in summary['matched_metrics']:
        key=f"{row['series']}-h{row['horizon']}-{row['scheme']}"
        if key not in datasets: datasets[key]=load(f'data/forecasts/{key}.json')
        d=datasets[key];m=row['model'];assert len(d['origin_index'])==924
        assert collections.Counter(d['matched_fold'])=={0:231,1:231,2:231,3:231}
        assert len(set(d['origin_index']))==924
        mm=calc(d['y_true'],d['predictions'][m],d['random_walk_pred'])
        for metric in ['rmse','mae','r2','rmse_relative_to_persistence']:
            near(mm[metric],row[metric])
        if row['series']!='monash_sunspot':near(mm['da'],row['da'])
        else: assert row['da'] is None
    index={k:{n:i for i,n in enumerate(d['origin_index'])} for k,d in datasets.items()}
    for r in sources:
        key=f"{r['series']}-h{r['horizon']}-{r['scheme']}";d=datasets[key];i=index[key][int(r['origin_index'])]
        assert d['origin_date'][i]==r['origin_date'] and d['target_date'][i]==r['target_date']
        for dest,src in [(d['predictions'][r['model']][i],'y_pred'),(d['training_means'][r['model']][i],'train_mean_pred'),(d['y_true'][i],'y_true'),(d['random_walk_pred'][i],'random_walk_pred')]:
            assert dest==float(r[src]),(key,i,src)
    assert len(sources)==129360 and len(datasets)==20
    print('PASS — all 129,360 matched forecasts round-trip exactly; all 140 metric cells independently recompute.')

    attr_sources=rows('research/results/cross_model/tables/attributions.csv');attr_files={}
    for r in attr_sources:
        key=f"{r['model']}-{r['series']}-h{r['horizon']}-{r['scheme']}"
        if key not in attr_files:attr_files[key]=load(f'data/attributions/{key}.json')
        a=attr_files[key];f=a['folds'].index(int(r['fold']));p=a['probes'].index(int(r['probe']));j=[1,5,21,252].index(int(r['feature'].split('_')[1]))
        assert a['values'][f][p][j]==float(r['attribution'])
        assert a['dates'][p]==r['origin_date'] and a['regimes'][p]==r['regime']
    assert len(attr_sources)==78400
    print('PASS — all 78,400 attributions round-trip exactly, including fold/probe/date/regime labels.')

    fits=load('data/linear.json')['folds'];raw={s['id']:load('data/series/'+s['id']+'.json') for s in manifest['series']};maxerr=0
    native=rows('research/results/Linear_study/tables/observation_predictions.csv')
    for r in native:
        sid=r['series'];fit=fits[f"{sid}-h{r['horizon']}-{r['scheme']}-f{r['fold']}"]
        pos=int(r['origin_index'])+(253 if sid!='monash_sunspot' else 252)
        features=[raw[sid]['signal'][pos-lag] for lag in [1,5,21,252]]
        pred=fit['intercept']+sum(w*(v-a)/b for v,a,b,w in zip(features,fit['mean'],fit['scale'],fit['weights']))
        maxerr=max(maxerr,abs(pred-float(r['y_pred'])))
    assert len(native)==30240 and maxerr<1e-9
    print('PASS — reconstructed OLS inference matches all 30,240 native forecasts; max error',format(maxerr,'.3g'))

    books=[f for f in inv if f['ext']=='ipynb'];assert len(books)==8
    for b in books:
        nb=load(b['path']);codes=[c for c in nb['cells'] if c['cell_type']=='code'];counts=[c['execution_count'] for c in codes]
        assert counts==list(range(1,len(codes)+1))
        assert not [o for c in codes for o in c.get('outputs',[]) if o['output_type']=='error']
    ms=load('data/manuscript.json');assert len(ms['references'])==47 and len(ms['chapters'])==11
    assert hashlib.sha256((APP/ms['source']).read_bytes()).hexdigest()==ms['sha256']
    assert [c['title'] for c in ms['chapters'][1:9]]==['Introduction','Theoretical Background','Literature Review','Methodology','Results','Discussion','Limitations','Conclusion']
    for c in ms['chapters']:
        for block in c['blocks']:
            if block['type']=='image':assert (APP/block['src']).is_file()
    assert len(summary['dm_pairwise'])==420
    for r in summary['dm_pairwise']:assert r['significant_q05']==(r['p_bh_21_pairs']<.05)
    benchmarks=[r for r in summary['native_benchmark_block_dm'] if r['sample']=='financial']
    assert len(benchmarks)==224 and not [r for r in benchmarks if r['outcome']=='model_win']
    print('PASS — 8 notebooks have sequential stored execution and no stored errors; manuscript chapters, 47 references, and inference labels are consistent.')

    for path in ['index.html','assets/app.js','assets/core.js','assets/charts.js','assets/style.css','assets/vendor/marked.js','manifest.webmanifest','sw.js','cache-manifest.json']:
        assert (APP/path).is_file(),path
    offline=load('cache-manifest.json')
    for name in offline['files']:
        p=(APP/name).resolve();assert p.is_file() and p.is_relative_to(APP.resolve()),name
    report={'status':'passed','scope':'Static app data and source preservation; seven-model training was not re-executed.',
            'original_files_sha256_verified':261,'matched_forecasts_exact':129360,'matched_metric_cells_recomputed':140,
            'attributions_exact':78400,'native_ols_predictions_verified':30240,'ols_max_absolute_error':maxerr,
            'executed_notebooks_checked':8,'manuscript_chapters_including_abstract_references_appendices':11,'references':47,
            'pairwise_test_labels_checked':420,'financial_native_benchmark_tests_checked':224,
            'offline_pack_files':len(offline['files'])}
    (APP/'verification-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS — shell assets and offline-pack paths exist. Verification report written.')
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (AssertionError,KeyError,ValueError,FileNotFoundError) as e:
        print('FAIL —',e,file=sys.stderr);raise
