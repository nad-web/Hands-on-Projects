#!/usr/bin/env python3
"""Compile supplied research artifacts into a static, traceable browser dataset.

Usage: python3 tools/build_data.py [path/to/whole_thesis_Package_Submission]
Only regeneration needs numpy. The delivered website has no build dependencies.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import shutil
import sys

APP = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else APP / 'research'
DEST = APP / 'research'
DATA = APP / 'data'
MODELS = ['Linear', 'ARIMA', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM', 'TFT']
SERIES = ['sp500', 'dax', 'nasdaq', 'djia', 'monash_sunspot']
LABELS = ['S&P 500', 'DAX', 'NASDAQ Composite', 'Dow Jones', 'Monash Sunspot']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(s):
    if s == '' or s in ('nan', 'NaN'): return None
    if s in ('True', 'False'): return s == 'True'
    try:
        f = float(s)
        return int(f) if f.is_integer() and abs(f) < 2**53 else f
    except (ValueError, TypeError): return s


def read_csv(path):
    with path.open(newline='') as f:
        return [{k: value(v) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    if not SRC.is_dir(): raise SystemExit('Research source folder not found: ' + str(SRC))
    allowed = json.loads((DATA / 'public-source-list.json').read_text())
    # The explicit submission-file allowlist excludes the complete thesis and private archives.
    if SRC != DEST:
        for relative in allowed:
            source, target = SRC / relative, DEST / relative
            if not source.is_file(): raise SystemExit('Missing approved source: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    actual = {str(f.relative_to(DEST)) for f in DEST.rglob('*') if f.is_file()}
    if actual != set(allowed): raise SystemExit('Research folder differs from the public source allowlist. Use a clean destination.')
    DATA.mkdir(exist_ok=True)
    cross = DEST / 'results/cross_model/tables'
    names = ['financial_model_summary', 'model_summary', 'matched_metrics', 'matched_fold_metrics', 'native_fold_metrics',
             'relative_importance_by_model', 'relative_importance_by_cell', 'faithfulness_summary', 'stability_summary',
             'stability_by_scheme', 'seed_stability_summary', 'regime_contrasts', 'dm_pairwise',
             'native_benchmark_summary', 'native_benchmark_block_dm', 'tft_native_signals', 'arima_native_params']
    tables = {name: read_csv(cross / (name + '.csv')) for name in names}
    tables['faithfulness'] = read_csv(cross / 'faithfulness.csv')
    stability = collections.defaultdict(list)
    for row in read_csv(cross / 'stability.csv'):
        key = tuple(row[k] for k in ['model','method','series','horizon','scheme','regime'])
        stability[key].append(row)
    import statistics
    def med(rows, key):
        vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and math.isfinite(r[key])]
        return statistics.median(vals) if vals else None
    tables['stability_cells'] = [dict(zip(['model','method','series','horizon','scheme','regime'], key),
        rho_median=med(rows,'rho'), cosine_median=med(rows,'cosine'),
        top1_match_rate=sum(bool(r['top1_match']) for r in rows)/len(rows), count=len(rows))
        for key,rows in sorted(stability.items())]
    dump(DATA / 'summary.json', tables)

    series_meta = []
    raw = {}
    for sid, label in zip(SERIES, LABELS):
        fname = f'{sid}_observations.csv' if sid != 'monash_sunspot' else 'monash_sunspot_no_missing_T1_85babb464a3e.csv'
        rows = read_csv(DEST / 'data' / fname)
        dates = [r.get('Date', r.get('date')) for r in rows]
        levels = [r.get('Adj Close', r.get('value')) for r in rows]
        returns = [None] + [math.log(levels[i] / levels[i-1]) for i in range(1, len(levels))] if sid != 'monash_sunspot' else levels
        raw[sid] = {'dates': dates, 'level': levels, 'signal': returns}
        dump(DATA / 'series' / (sid + '.json'), raw[sid])
        series_meta.append({'id': sid, 'label': label, 'rows': len(rows), 'start': dates[0], 'end': dates[-1],
                            'financial': sid != 'monash_sunspot', 'file': 'research/data/' + fname,
                            'sha256': sha(DEST / 'data' / fname)})

    predictions = read_csv(cross / 'matched_observation_predictions.csv')
    groups = collections.defaultdict(list)
    for r in predictions: groups[(r['series'], r['horizon'], r['scheme'])].append(r)
    for (sid, h, scheme), rows in groups.items():
        by_model = {m: sorted((r for r in rows if r['model'] == m), key=lambda r:r['origin_index']) for m in MODELS}
        first = by_model['Linear']
        assert len(first) == 924
        for mm in by_model.values():
            assert [r['origin_index'] for r in mm] == [r['origin_index'] for r in first]
            assert [r['y_true'] for r in mm] == [r['y_true'] for r in first]
        out = {k: [r[k] for r in first] for k in ['origin_index', 'origin_date', 'target_date', 'y_true', 'random_walk_pred', 'matched_fold']}
        out.update(series=sid, horizon=h, scheme=scheme)
        out['predictions'] = {m: [r['y_pred'] for r in by_model[m]] for m in MODELS}
        out['training_means'] = {m: [r['train_mean_pred'] for r in by_model[m]] for m in MODELS}
        out['folds'] = {m: [r['fold'] for r in by_model[m]] for m in MODELS}
        lookup = dict(zip(raw[sid]['dates'], raw[sid]['level']))
        out['origin_level'] = [lookup[d] for d in out['origin_date']]
        dump(DATA / 'forecasts' / f'{sid}-h{h}-{scheme}.json', out)

    agroups = collections.defaultdict(list)
    for r in read_csv(cross / 'attributions.csv'):
        agroups[(r['model'], r['series'], r['horizon'], r['scheme'])].append(r)
    for (m, sid, h, scheme), rows in agroups.items():
        probes = sorted(set(r['probe'] for r in rows)); folds = sorted(set(r['fold'] for r in rows))
        lookup = {(r['fold'], r['probe'], r['feature']): r for r in rows}
        first = [lookup[(folds[0], p, 'lag_1')] for p in probes]
        out = {'model': m, 'method': rows[0]['method'], 'folds': folds, 'probes': probes,
               'dates': [r['origin_date'] for r in first], 'regimes': [r['regime'] for r in first],
               'origins': [r['origin_index'] for r in first],
               'values': [[[lookup[(f, p, 'lag_' + str(l))]['attribution'] for l in [1, 5, 21, 252]] for p in probes] for f in folds]}
        dump(DATA / 'attributions' / f'{m}-{sid}-h{h}-{scheme}.json', out)

    # Reconstruct the actual OLS folds and verify ALL native exported predictions.
    # This enables honest out-of-sample feature sliders without inventing nonlinear-model weights.
    import numpy as np
    linear = {}
    max_error = 0.0
    checked = 0
    native = read_csv(DEST / 'results/Linear_study/tables/observation_predictions.csv')
    for sid in SERIES:
        signal = np.asarray(raw[sid]['signal'][1:] if sid != 'monash_sunspot' else raw[sid]['signal'], dtype=float)
        n = len(signal) - 252 - 5
        frame_positions = np.arange(252, len(signal)-5)
        X = np.column_stack([signal[frame_positions-l] for l in [1, 5, 21, 252]])
        for h in [1, 5]:
            y = sum(signal[frame_positions+i] for i in range(1, h+1)) if sid != 'monash_sunspot' else signal[frame_positions+h]
            for scheme in ['rolling', 'expanding']:
                for fold in range(6):
                    te = n-6*252+fold*252 if scheme == 'rolling' else 510+fold*252
                    end = te-10; start = max(0, end-500) if scheme == 'rolling' else 0
                    xt, yt = X[start:end], y[start:end]
                    mean, scale, intercept = xt.mean(axis=0), xt.std(axis=0), float(yt.mean())
                    scale[scale == 0] = 1
                    weights = np.linalg.lstsq((xt-mean)/scale, yt-intercept, rcond=None)[0]
                    key = f'{sid}-h{h}-{scheme}-f{fold}'
                    linear[key] = {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(),
                                   'intercept': intercept, 'train_start': start, 'train_end': end,
                                   'test_start': te, 'test_end': te+252, 'n_train': end-start}
                    rs = [r for r in native if r['series']==sid and r['horizon']==h and r['scheme']==scheme and r['fold']==fold]
                    for r in rs:
                        pred = float(intercept + np.dot((X[r['origin_index']]-mean)/scale, weights))
                        max_error = max(max_error, abs(pred-r['y_pred'])); checked += 1
    assert checked == 30240 and max_error < 1e-9, (checked, max_error)
    dump(DATA / 'linear.json', {'folds': linear, 'verification': {'predictions_checked':checked, 'max_abs_error':max_error,
                            'method':'OLS reconstructed using the executed standardized features and exact fold training intervals; not a surrogate.'}})

    references = len(json.loads((DATA / 'guide.json').read_text())['references'])
    inventory = []
    for f in sorted(DEST.rglob('*')):
        if not f.is_file(): continue
        rel = str(f.relative_to(DEST))
        ext = f.suffix[1:]
        category = ('Notebooks' if ext=='ipynb' else 'Figures' if ext=='png' else 'Data & tables' if ext in ('csv','tsf')
                    else 'Audit & evidence')
        inventory.append({'path':'research/'+rel, 'name':f.name, 'category':category, 'ext':ext,
                          'bytes':f.stat().st_size, 'sha256':sha(f),
                          'legacy': rel.startswith('source/')})
    dump(DATA / 'inventory.json', inventory)
    provenance = {m:json.loads((DEST/f'results/{m}_study/evidence/provenance.json').read_text()) for m in MODELS}
    dump(DATA / 'provenance.json', provenance)
    info = {'title':'Towards Trustworthy Financial Time Series Forecasting',
            'subtitle':'An Experimental Evaluation of Explainable Machine Learning Models for Stock Price Prediction',
            'series':series_meta, 'models':MODELS, 'files':len(inventory), 'matched_origins':18480,
            'matched_predictions':len(predictions), 'native_predictions':188160, 'matched_cells':140,
            'attribution_rows':sum(len(v) for v in agroups.values()), 'references':references, 'manuscript_excluded':True,
            'source_bytes':sum(f['bytes'] for f in inventory), 'version':'2.1.0', 'compiled':'2026-11-18',
            'execution_version':'corrected-independent-rerun-v1',
            'source_priority':['results/cross_model','notebooks/*_Executed.ipynb'],
            'source_note':'Public research companion. The complete thesis manuscript and any full-text reader are excluded. Corrected submission CSV exports and the eight executed notebooks drive all reported results.',
            'linear_verification':{'predictions_checked':checked,'max_abs_error':max_error}}
    dump(DATA / 'manifest.json', info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__': main()
