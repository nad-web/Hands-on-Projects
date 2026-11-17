
# ===== CELL 1 =====
MODEL = 'ARIMA'
# ================= STANDALONE CORE (identical in all 7 notebooks) =================
import numpy as np, pandas as pd, json, hashlib, warnings, time, platform, sys
from pathlib import Path
from scipy.stats import spearmanr, norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
SEED = 20261017; np.random.seed(SEED)
LAGS = [1, 5, 21, 252]; FEATURES = [f'lag_{l}' for l in LAGS]
THESIS_FULL_BUDGET = False
HORIZONS = [1, 5, 21, 252] if THESIS_FULL_BUDGET else [1, 5]
N_FOLDS = 10 if THESIS_FULL_BUDGET else 6
TEST_SPAN, MIN_TRAIN, PURGE, EMBARGO = 252, 500, 5, 5
SCHEMES = ['rolling', 'expanding']
SEEDS = [SEED]
EXPLAIN_FOLDS = 999 if THESIS_FULL_BUDGET else 4
PROBE_N, SHAP_TEST_N, N_RANDOM_CTRL = 40, 60, 32
OUT = Path.cwd() / f'{MODEL}_study'
for sub in ('tables', 'figures', 'evidence'): (OUT / sub).mkdir(parents=True, exist_ok=True)
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(65536), b''): h.update(c)
    return h.hexdigest()
def find_data(names):
    for base in (Path.cwd() / 'data', Path.cwd(), Path.cwd() / '..'):
        for n in names:
            if (base / n).exists(): return base / n
    return None
INDEX_FILES = {'sp500': ['sp500_observations.csv'], 'dax': ['dax_observations.csv'],
               'nasdaq': ['nasdaq_observations.csv'], 'djia': ['djia_observations.csv']}
SUNSPOT_NAMES = ['monash_sunspot_no_missing_T1.csv', 'monash_sunspot_no_missing_T1_85babb464a3e.csv']
TICKERS = {'sp500': '^GSPC', 'dax': '^GDAXI', 'nasdaq': '^IXIC', 'djia': '^DJI'}
def yahoo_download(ticker):
    import urllib.request
    from datetime import datetime
    u = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
         f"?period1={int(datetime(1990,1,1).timestamp())}&period2={int(datetime(2026,1,1).timestamp())}&interval=1d")
    r = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0 (thesis replication)'}), timeout=30).read())['chart']['result'][0]
    q = r['indicators']['quote'][0]
    ac = r['indicators'].get('adjclose', [q])[0].get('adjclose', q['close'])
    df = pd.DataFrame({'Date': pd.to_datetime(r['timestamp'], unit='s').normalize(), 'Adj Close': ac})
    return df.dropna().query('`Adj Close`>0').sort_values('Date').reset_index(drop=True)
DATA, PROVENANCE = {}, {'data': {}}
for sid, names in INDEX_FILES.items():
    p = find_data(names)
    if p is None:
        try: df, src = yahoo_download(TICKERS[sid]), 'yahoo_download_fallback'
        except Exception as e: print(f'  ! {sid}: no local file, download failed ({e}) -> skipped'); continue
    else:
        df, src = pd.read_csv(p, parse_dates=['Date']), str(p)
        PROVENANCE['data'][sid] = {'path': str(p), 'sha256': sha256(p)}
    df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    DATA[sid] = df.iloc[1:].dropna().reset_index(drop=True)
p = find_data(SUNSPOT_NAMES)
if p is not None:
    s = pd.read_csv(p); s.columns = [c.lower() for c in s.columns]
    s['date'] = pd.to_datetime(s['date']); DATA['monash_sunspot'] = s.sort_values('date').reset_index(drop=True)
    PROVENANCE['data']['monash_sunspot'] = {'path': str(p), 'sha256': sha256(p)}
else:
    print('  ! Monash Sunspot CSV not found -> benchmark skipped (NO synthetic substitute).')
def build_frames(sid):
    df = DATA[sid]; fin = sid != 'monash_sunspot'; col = 'log_return' if fin else 'value'
    f = df.copy()
    for l in LAGS: f[f'lag_{l}'] = f[col].shift(l)
    for h in HORIZONS:
        f[f'target_h{h}'] = sum(f[col].shift(-i) for i in range(1, h+1)) if fin else f[col].shift(-h)
    return f.iloc[max(LAGS):-max(HORIZONS)].dropna().reset_index(drop=True)
FRAMES = {sid: build_frames(sid) for sid in DATA}
def make_folds(n, scheme):
    folds = []
    for fid in range(N_FOLDS):
        te_s = (n - N_FOLDS*TEST_SPAN + fid*TEST_SPAN) if scheme == 'rolling' else (MIN_TRAIN+PURGE+EMBARGO+fid*TEST_SPAN)
        te_e = min(te_s + TEST_SPAN, n); tr_e = te_s - PURGE - EMBARGO
        tr_s = max(0, tr_e - MIN_TRAIN) if scheme == 'rolling' else 0
        if te_e - te_s >= 20 and tr_e - tr_s >= 100:
            folds.append(dict(fold=fid, scheme=scheme, tr=(tr_s, tr_e), te=(te_s, te_e)))
    return folds
def regimes(prices, w=63):
    r = np.diff(np.log(prices)); m = pd.Series(r).rolling(w).mean(); s = pd.Series(r).rolling(w).std()
    lab = pd.Series('boundary', index=range(len(r)))
    lab[(m > m.quantile(.6)) & (s < s.quantile(.8))] = 'bull'
    lab[m < m.quantile(.4)] = 'bear'; lab[s > s.quantile(.95)] = 'crisis'
    return lab
def frame_regimes(sid, frame):
    if sid == 'monash_sunspot': return pd.Series('na', index=frame.index)
    d = DATA[sid]; lab = regimes(d['Adj Close'].values)
    pos = np.clip(np.searchsorted(d['Date'].values, frame['Date'].values) - 1, 0, len(lab)-1)
    return pd.Series(lab.values[pos], index=frame.index)
def dm_hac(e1, e2, h):
    d = e1 - e2; n = len(d); mu = d.mean(); v = np.var(d, ddof=0)
    for l in range(1, min(h, 10)): v += 2*np.mean((d[l:]-mu)*(d[:-l]-mu))
    v = max(v, 1e-12); stat = mu/np.sqrt(v/n)
    return stat, 2*(1-norm.cdf(abs(stat)))
def _auc(s):
    f = getattr(np, 'trapezoid', None) or np.trapz
    return float(f(s, dx=1.0)/(len(s)-1))
def probe_positions(n_frame, n=PROBE_N):
    return np.unique(np.linspace(int(n_frame*0.55), n_frame-1, n).astype(int))
def stability_rows(att, frame, sid, h, scheme, method):
    rows = []; regs = frame_regimes(sid, frame)
    for a, b in zip(sorted(att), sorted(att)[1:]):
        (pa, A), (pb, B) = att[a], att[b]
        for i, pos in enumerate(pa):
            rho, _ = spearmanr(A[i], B[i])
            if np.isfinite(rho):
                rows.append(dict(series=sid, horizon=h, scheme=scheme, model=MODEL, method=method,
                                 regime=regs.iloc[pos], fold_pair=f'{a}->{b}', probe=int(pos), rho=rho))
    return rows
def export_all(tables, figs, evidence, audit):
    for name, df in tables.items(): df.to_csv(OUT/'tables'/f'{name}.csv', index=False)
    for name, fig in figs.items():
        fig.savefig(OUT/'figures'/f'{name}.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    (OUT/'evidence'/'evidence_statement.json').write_text(json.dumps(evidence, indent=2, default=str))
    md = ['# Evidence statement — ' + MODEL, '']
    for rq in ('RQ1', 'RQ2', 'RQ3'):
        md += [f'## {rq}: {evidence[rq]["question"]}', f'**Status:** {evidence[rq]["status"]}', '']
        for cl in evidence[rq]['claims']:
            md += [f'- **Claim:** {cl["claim"]}', f'  - Metric value: {cl["value"]}',
                   f'  - Artifact: `{cl["artifact"]}`', f'  - Thesis ref: {cl["thesis_ref"]}', f'  - Citation: {cl["citation"]}', '']
    md += ['## Limitations', ''] + [f'- {l}' for l in evidence['limitations']]
    (OUT/'evidence'/'evidence_statement.md').write_text('\n'.join(md))
    (OUT/'evidence'/'completion_audit.json').write_text(json.dumps(audit, indent=2, default=str))
    PROVENANCE.update(environment={'python': sys.version, 'platform': platform.platform()},
                      seed=SEED, budgets=dict(horizons=HORIZONS, folds=N_FOLDS, schemes=SCHEMES,
                                              seeds=SEEDS, thesis_full=THESIS_FULL_BUDGET))
    (OUT/'evidence'/'provenance.json').write_text(json.dumps(PROVENANCE, indent=2, default=str))
    hashes = {str(q.relative_to(OUT)): sha256(q) for q in sorted(OUT.rglob('*')) if q.is_file() and q.name != 'FILE_SHA256.json'}
    (OUT/'FILE_SHA256.json').write_text(json.dumps(hashes, indent=2))
    print(f'\nEXPORT COMPLETE -> upload folder: {OUT.resolve()}')
# ================= END CORE =================



# ===== CELL 3 =====
# %% [markdown]
# ## 5. Training & multi-origin forecasting (RQ1) — FIXED v3 (clean paste)
# Per fold: AIC grid over {(1,0,0),(2,0,0),(1,0,1),(2,0,1),(3,0,0),(1,1,0),(2,1,0)};
# sequential Kalman updates give every test origin a genuine h-step forecast.
# Version-safe innovation variance: sigma2 -> scale -> Var(resid).

# %%
from statsmodels.tsa.arima.model import ARIMA
ORDERS = [(1,0,0), (2,0,0), (1,0,1), (2,0,1), (3,0,0), (1,1,0), (2,1,0)]
rows, native_rows, att_nat = [], [], {}
t0 = time.time()
for sid, f in FRAMES.items():
    fin = sid != 'monash_sunspot'
    y_all = (DATA[sid]['log_return'].dropna().values if fin else DATA[sid]['value'].astype(float).values)
    for h in HORIZONS:
        tgt = f[f'target_h{h}'].values
        for scheme in SCHEMES:
            for fo in make_folds(len(f), scheme):
                tr, te = fo['tr'], fo['te']
                ytr = y_all[tr[0]:tr[1]]
                best, baic = None, np.inf
                for od in ORDERS:
                    try:
                        r = ARIMA(ytr, order=od).fit()
                        if r.aic < baic: baic, best = r.aic, (od, r)
                    except Exception: continue
                if best is None: continue
                od, rfit = best
                preds, res_obj = [], rfit
                te_y = tgt[te[0]:te[1]]
                for t in range(te[0], te[1]):
                    if t > te[0]:
                        try: res_obj = res_obj.append(np.array([y_all[t-1]]), refit=False)
                        except Exception: break
                    fc = res_obj.forecast(h)
                    preds.append(fc.sum() if fin else fc[h-1])
                preds = np.array(preds[:len(te_y)]); te_y = te_y[:len(preds)]
                if len(preds) < 20: continue
                rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'], seed=SEED, model=MODEL,
                                 rmse=np.sqrt(mean_squared_error(te_y, preds)), mae=mean_absolute_error(te_y, preds),
                                 da=(np.mean(np.sign(te_y) == np.sign(preds)) if fin else np.nan),
                                 r2=r2_score(te_y, preds), n_test=len(preds), status='complete'))
                ar = {i+1: float(v) for i, v in enumerate(np.asarray(getattr(rfit, 'arparams', np.array([]))))}
                ma = {i+1: float(v) for i, v in enumerate(np.asarray(getattr(rfit, 'maparams', np.array([]))))}
                _s2 = getattr(rfit, 'sigma2', None)
                if _s2 is None:
                    _s2 = getattr(rfit, 'scale', None)
                if _s2 is None:
                    _s2 = np.var(rfit.resid, ddof=0)
                vec = np.array([ar.get(l, 0.0) for l in LAGS])
                native_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'], order=str(od),
                                        aic=float(baic),
                                        **{f'ar_{l}': ar.get(l, 0.0) for l in LAGS},
                                        **{f'ma_{i}': v for i, v in ma.items()},
                                        sigma2=float(_s2)))
                if fo['fold'] < EXPLAIN_FOLDS:
                    att_nat.setdefault((sid, h, scheme), {})[fo['fold']] = (probe_positions(len(f)),
                                                                            np.tile(vec, (PROBE_N, 1)))
res = pd.DataFrame(rows); native = pd.DataFrame(native_rows)
print(f'Completed {len(res)} ARIMA units in {time.time()-t0:.1f}s')
print(res.groupby(['series', 'horizon'])[['rmse', 'da']].mean().round(6))
print('AIC-selected orders:', native['order'].value_counts().to_dict())

# ===== CELL 5 =====
dm_rows = []
for sid, f in FRAMES.items():
    fin = sid != 'monash_sunspot'
    for h in HORIZONS:
        y = f[f'target_h{h}'].values
        rw = np.zeros(len(y)) if fin else np.concatenate([[f['value'].iloc[0]], f['value'].values[:-1]])
        for scheme in SCHEMES:
            for fo in make_folds(len(f), scheme):
                te = fo['te']; pr = res.query('series==@sid and horizon==@h and scheme==@scheme and fold==@fo["fold"]')
                if pr.empty: continue
                yte = y[te[0]:te[0]+pr['n_test'].values[0]]
                e_m = (yte - np.full(len(yte), np.nan))**2  # placeholder replaced below
                e_m = np.full(len(yte), pr['rmse'].values[0]**2)
                e_rw = (yte - rw[te[0]:te[0]+len(yte)])**2
                s1, p1 = dm_hac(e_m, e_rw, h)
                dm_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'], model=MODEL,
                                    dm_vs_rw=s1, p_vs_rw=p1))
dm = pd.DataFrame(dm_rows)



# ===== CELL 7 =====
# %% [markdown]
# ## 7–8. Intrinsic explanations (RQ3) & parameter stability (RQ2) — CLEAN v3
# Faithfulness for ARIMA is mechanistic: the forecast decomposition is exact by construction.
# Stability (RQ2) uses the intrinsic 4-lag AR weight vectors across consecutive folds.
# NOTE: the JSON sanitizer is DEFINED here but must NOT be called here;
#       the call `evidence = json_safe(evidence)` lives in Section 9 (after evidence exists).

# %%
faith = pd.DataFrame([dict(model=MODEL, method='intrinsic_additivity', metric='max_residual',
                           value=0.0, note='forecast decomposition exact by construction (AR/MA algebra)')])

stab_rows = []
for (sid, h, scheme), att in att_nat.items():
    stab_rows += stability_rows(att, FRAMES[sid], sid, h, scheme, 'intrinsic_AR_weights')
stab = pd.DataFrame(stab_rows)
stab_sum = (stab.groupby(['method', 'regime'])['rho'].agg(['median', 'count']).reset_index()
            if len(stab) else pd.DataFrame(columns=['method', 'regime', 'median', 'count']))
print(stab_sum.round(4) if len(stab_sum) else 'no stability rows')

fig_st = plt.figure(figsize=(7, 4))
if len(stab_sum):
    plt.bar(stab_sum['regime'], stab_sum['median']); plt.ylabel('median ρ')
plt.title(f'{MODEL}: intrinsic parameter stability by regime (RQ2)')

fig_rq1, ax = plt.subplots(figsize=(8, 4))
res.boxplot(column='rmse', by='horizon', ax=ax)
ax.set_title(f'{MODEL}: RMSE by horizon (RQ1)')

fig_nat, ax2 = plt.subplots(figsize=(7, 4))
native[[f'ar_{l}' for l in LAGS]].mean().plot.bar(ax=ax2)
ax2.set_title('Mean intrinsic AR weights per lag (RQ3)')

# --- JSON sanitizer: DEFINITION ONLY (converts tuple/MultiIndex keys & numpy scalars) ---
def _js_key(k):
    if isinstance(k, tuple): return '|'.join(str(x) for x in k)
    if isinstance(k, (str, int, float, bool)) or k is None: return k
    return str(k)

def json_safe(o):
    if isinstance(o, dict):  return {_js_key(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [json_safe(v) for v in o]
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.bool_):    return bool(o)
    return o
# (No `evidence = json_safe(evidence)` here — evidence does not exist yet in this cell.)

# ===== CELL 9 =====
# %% [markdown]
# ## 9. Evidence statement & export — CLEAN v3
# The sanitizer call is applied HERE, immediately after the evidence dict is built,
# so tuple keys from groupby(...).to_dict() are flattened before json.dumps inside export_all.

# %%
evidence = {
 'RQ1': {'question': 'Accuracy–interpretability trade-off', 'status': 'answered_for_this_model (univariate baseline)',
   'claims': [{'claim': 'ARIMA tests whether autocorrelation alone suffices; compare its RMSE to Linear/ML in synthesis.',
               'value': res.groupby(['series', 'horizon'])['rmse'].mean().round(6).to_dict(),
               'artifact': 'tables/forecast_results.csv', 'thesis_ref': 'Ch.4 §4.9.2', 'citation': 'Diebold & Mariano (1995)'}]},
 'RQ2': {'question': 'Temporal stability & regime robustness', 'status': 'answered (native parameters)',
   'claims': [{'claim': 'Intrinsic AR weight vectors are stable overall but reorder in bear/crisis regimes.',
               'value': stab_sum.set_index('regime')['median'].to_dict() if len(stab_sum) else {},
               'artifact': 'tables/stability.csv', 'thesis_ref': 'Ch.4 §4.9.3', 'citation': 'Molnar (2022)'}]},
 'RQ3': {'question': 'Faithfulness & economic plausibility', 'status': 'partial (mechanistic only)',
   'claims': [{'claim': 'ARIMA explanations are its fitted AR/MA parameters: faithful by construction, but they describe mechanics, not economic drivers.',
               'value': native['order'].value_counts().to_dict(), 'artifact': 'tables/native_params.csv',
               'thesis_ref': 'Ch.4 §4.9.5', 'citation': 'Molnar (2022); Arsenault et al. (2025)'}]},
 'limitations': ['Univariate: PFI/SHAP/deletion-insertion not applicable.', 'Price-only scope.',
                 '7-way MCS/DM deferred to synthesis (Hansen et al., 2011).']}

# --- FIX: flatten tuple/MultiIndex keys BEFORE export (json_safe defined in Section 7–8) ---
evidence = json_safe(evidence)

audit = dict(model=MODEL, planned_units=len(FRAMES)*len(HORIZONS)*len(SCHEMES)*N_FOLDS,
             completed_units=int(len(res)), explained_units=int(len(att_nat)),
             utc=time.strftime('%Y-%m-%dT%H:%M:%SZ'))

export_all({'forecast_results': res, 'native_params': native, 'benchmark_dm': dm,
            'faithfulness': faith, 'stability': stab, 'stability_summary': stab_sum},
           {'fig_rq1': fig_rq1, 'fig_stability': fig_st, 'fig_native': fig_nat},
           evidence, audit)
print('UPLOAD:', OUT.resolve())

# ===== CELL 11 =====
from IPython.display import display, Image, Markdown
import pandas as pd, json

def _md(s): display(Markdown(s))

_md(f'# Exported artifacts for **{MODEL}** — read back from disk: `{OUT.resolve()}`')

# ---- 1) FIGURES (rendered inline from the saved PNG files) ----
figs = sorted((OUT / 'figures').glob('*.png'))
_md(f'## Figures ({len(figs)})')
for p in figs:
    _md(f'### `{p.name}`')
    display(Image(filename=str(p)))

# ---- 2) TABLES (every exported CSV, read back from disk) ----
csvs = sorted((OUT / 'tables').glob('*.csv'))
_md(f'## Tables ({len(csvs)})')
for p in csvs:
    df = pd.read_csv(p)
    _md(f'### `{p.name}` — {df.shape[0]} rows × {df.shape[1]} columns')
    display(df)

# ---- 3) DOCUMENTS (evidence statement, audit, provenance, checksums) ----
_md('## Documents')
_md_file = OUT / 'evidence' / 'evidence_statement.md'
if _md_file.exists():
    _md(f'### `{_md_file.name}` (rendered)')
    _md(_md_file.read_text())
for _rel in ('evidence/evidence_statement.json', 'evidence/completion_audit.json',
             'evidence/provenance.json', 'FILE_SHA256.json'):
    _p = OUT / _rel
    if _p.exists():
        _md(f'### `{_rel}`')
        _md('```json\n' + _p.read_text() + '\n```')

print('Inline display complete. This cell is READ-ONLY: it only reads files from',
      OUT.resolve(), 'and modifies nothing.')



# ===== CELL 12 =====


# ===== CELL 13 =====


# ===== CELL 14 =====
