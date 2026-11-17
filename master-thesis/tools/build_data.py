#!/usr/bin/env python3
"""Compile supplied research artifacts into a static, traceable browser dataset.

Usage: python3 tools/build_data.py [path/to/Master_Thesis_Package]
Only regeneration needs numpy. The delivered website has no build dependencies.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

APP = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else APP / 'research'
DEST = APP / 'research'
DATA = APP / 'data'
MODELS = ['Linear', 'ARIMA', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM', 'TFT']
SERIES = ['sp500', 'dax', 'nasdaq', 'djia', 'monash_sunspot']
LABELS = ['S&P 500', 'DAX', 'NASDAQ Composite', 'Dow Jones', 'Monash Sunspot']
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


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


def body_text(node):
    # Include Office Math text as well as ordinary Word runs, in source order.
    return ''.join(('\t' if x.tag.endswith('}tab') else '\n' if x.tag.endswith('}br') else x.text or '')
                   for x in node.iter() if x.tag.split('}')[-1] in ('t', 'tab', 'br'))


def manuscript():
    file = DEST / 'Master Thesis.docx'
    with zipfile.ZipFile(file) as z:
        root = ET.fromstring(z.read('word/document.xml'))
        rels = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
        chapters = [{'id': 'abstract', 'title': 'Abstract', 'number': '00', 'blocks': []}]
        chapter = chapters[0]
        started = False
        counters = [0, 0, 0]
        refs = []
        headings = []
        for el in root.find('w:body', NS):
            tag = el.tag.split('}')[-1]
            if tag == 'p':
                text = body_text(el).strip()
                style_el = el.find('w:pPr/w:pStyle', NS)
                style = style_el.attrib.get('{' + NS['w'] + '}val', '') if style_el is not None else ''
                if text == 'Abstract': started = True; continue
                if not started and not re.fullmatch(r'Heading[123]', style): continue
                if text in ('Table of Contents', 'List of Tables', 'Table of Figures'): started = False; continue
                if re.fullmatch(r'Heading[123]', style):
                    started = True
                    level = int(style[-1])
                    if level == 1:
                        counters[0] += 1; counters[1:] = [0, 0]
                        chapter = {'id': 'chapter-' + str(counters[0]), 'title': text,
                                   'number': f'{counters[0]:02}', 'blocks': []}
                        chapters.append(chapter)
                    else:
                        counters[level-1] += 1
                        if level == 2: counters[2] = 0
                        anchor = 's-' + '-'.join(str(n) for n in counters[:level])
                        chapter['blocks'].append({'type': 'heading', 'level': level, 'text': text,
                                                  'number': '.'.join(str(n) for n in counters[:level]), 'id': anchor})
                        headings.append({'chapter': chapter['id'], 'id': anchor, 'text': text})
                elif not style.startswith(('TOC', 'TableofFigures')) and text:
                    kind = 'caption' if style == 'Caption' else 'reference' if style == 'Bibliography' else 'paragraph'
                    chapter['blocks'].append({'type': kind, 'text': text})
                    if kind == 'reference':
                        urls = re.findall(r'https?://[^\s]+', text)
                        # Hyperlink target can be hidden behind a Word field.
                        for link in el.findall('.//w:hyperlink', NS):
                            rid = link.attrib.get('{' + NS['r'] + '}id')
                            target = rels.get(rid, '')
                            if target.startswith('http'): urls.append(target)
                        refs.append({'text': text, 'urls': list(dict.fromkeys(u.rstrip('.,;)') for u in urls))})
                for blip in el.findall('.//a:blip', NS):
                    rid = blip.attrib.get('{' + NS['r'] + '}embed')
                    target = rels.get(rid, '')
                    if target.startswith('media/'):
                        img = APP / 'assets' / 'manuscript' / Path(target).name
                        img.parent.mkdir(exist_ok=True, parents=True)
                        img.write_bytes(z.read('word/' + target))
                        chapter['blocks'].append({'type': 'image', 'src': str(img.relative_to(APP))})
            elif tag == 'tbl' and started:
                rows = [[body_text(c).strip() for c in row.findall('w:tc', NS)] for row in el.findall('w:tr', NS)]
                chapter['blocks'].append({'type': 'table', 'rows': rows})
            # Heading1 follows generated Word TOC: re-enable before next iteration.
            if tag == 'p':
                pass
        # Headings occur after TOC while started is false; recover with a second pass if needed.
        if len(chapters) < 5:
            raise AssertionError('Manuscript chapter extraction did not reach the main text')
    dump(DATA / 'manuscript.json', {'source': 'research/Master Thesis.docx', 'sha256': sha(file),
                                  'chapters': chapters, 'headings': headings, 'references': refs,
                                  'rendering': 'Reading edition extracted from Word. Tables and embedded images are retained; formulas are plain text. Download Word for submission layout.'})
    return len(chapters), len(refs)


def main():
    if not SRC.is_dir(): raise SystemExit('Research source folder not found: ' + str(SRC))
    if SRC != DEST:
        DEST.mkdir(parents=True, exist_ok=True)
        for f in SRC.rglob('*'):
            if not f.is_file() or any(part.startswith('.') or part == '__MACOSX' for part in f.relative_to(SRC).parts): continue
            if f.name.endswith('.docx.txt'): continue
            target = DEST / f.relative_to(SRC)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
    DATA.mkdir(exist_ok=True)
    cross = DEST / 'results/cross_model/tables'
    names = ['financial_model_summary', 'model_summary', 'matched_metrics', 'matched_fold_metrics', 'native_fold_metrics',
             'relative_importance_by_model', 'relative_importance_by_cell', 'faithfulness_summary', 'stability_summary',
             'stability_by_scheme', 'seed_stability_summary', 'regime_contrasts', 'dm_pairwise',
             'native_benchmark_summary', 'native_benchmark_block_dm', 'tft_native_signals', 'arima_native_params']
    tables = {name: read_csv(cross / (name + '.csv')) for name in names}
    tables['faithfulness'] = read_csv(cross / 'faithfulness.csv')
    dump(DATA / 'summary.json', tables)

    series_meta = []
    raw = {}
    for sid, label in zip(SERIES, LABELS):
        fname = f'{sid}_observations.csv' if sid != 'monash_sunspot' else 'monash_sunspot_no_missing_T1_85babb464a3e.csv'
        rows = read_csv(DEST / 'results/data' / fname)
        dates = [r.get('Date', r.get('date')) for r in rows]
        levels = [r.get('Adj Close', r.get('value')) for r in rows]
        returns = [None] + [math.log(levels[i] / levels[i-1]) for i in range(1, len(levels))] if sid != 'monash_sunspot' else levels
        raw[sid] = {'dates': dates, 'level': levels, 'signal': returns}
        dump(DATA / 'series' / (sid + '.json'), raw[sid])
        series_meta.append({'id': sid, 'label': label, 'rows': len(rows), 'start': dates[0], 'end': dates[-1],
                            'financial': sid != 'monash_sunspot', 'file': 'research/results/data/' + fname,
                            'sha256': sha(DEST / 'results/data' / fname)})

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

    chapters, references = manuscript()
    inventory = []
    for f in sorted(DEST.rglob('*')):
        if not f.is_file(): continue
        rel = str(f.relative_to(DEST))
        ext = f.suffix[1:]
        category = ('Notebooks' if ext=='ipynb' else 'Figures' if ext=='png' else 'Data & tables' if ext in ('csv','tsf')
                    else 'Manuscripts' if ext in ('docx','pptx') or 'Revised_Master_Thesis' in rel else 'Code' if ext=='py'
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
            'attribution_rows':sum(len(v) for v in agroups.values()), 'chapters':chapters, 'references':references,
            'source_bytes':sum(f['bytes'] for f in inventory), 'version':'1.0.0', 'compiled':'2026-11-17',
            'execution_version':'corrected-independent-rerun-v1',
            'source_priority':['Master Thesis.docx','results/cross_model','notebooks/*_Executed.ipynb'],
            'source_note':'source/Revised_Master_Thesis.md describes an earlier experiment and is retained as legacy material. Word is the primary reading source; corrected CSV exports drive all charts.',
            'linear_verification':{'predictions_checked':checked,'max_abs_error':max_error}}
    dump(DATA / 'manifest.json', info)
    print(json.dumps(info, indent=2))


if __name__ == '__main__': main()
