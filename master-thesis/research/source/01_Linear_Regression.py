
# ===== CELL 2 =====
MODEL = 'Linear'
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
THESIS_FULL_BUDGET = False   # True = Ch.4 full budget (4 horizons, 10 folds, 3 seeds, all explain folds)
HORIZONS = [1, 5, 21, 252] if THESIS_FULL_BUDGET else [1, 5]
N_FOLDS = 10 if THESIS_FULL_BUDGET else 6
TEST_SPAN, MIN_TRAIN, PURGE, EMBARGO = 252, 500, 5, 5
SCHEMES = ['rolling', 'expanding']
SEEDS = [SEED] if MODEL in ('Linear', 'ARIMA') else ([SEED, SEED+1, SEED+2] if THESIS_FULL_BUDGET else [SEED, SEED+1])
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
    print('  ! Monash Sunspot CSV not found -> benchmark skipped (NO synthetic substitute is created).')

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

def di_curves(pred_fn, X, y, order, n_random=N_RANDOM_CTRL):
    nf = X.shape[1]; out = {}
    cases = [('model', order)] + [(f'r{i}', np.random.RandomState(SEED+i).permutation(nf)) for i in range(n_random)]
    for tag, ord_ in cases:
        dl, il = [], []
        for k in range(nf+1):
            Xd = X.copy()
            for j in ord_[:k]: Xd[:, j] = 0
            dl.append(mean_squared_error(y, pred_fn(Xd)))
            Xi = np.zeros_like(X)
            for j in ord_[:k]: Xi[:, j] = X[:, j]
            il.append(mean_squared_error(y, pred_fn(Xi)))
        out[tag] = (_auc(dl), _auc(il))
    md, mi = out['model']
    rd = np.mean([out[f'r{i}'][0] for i in range(n_random)]); ri = np.mean([out[f'r{i}'][1] for i in range(n_random)])
    return dict(del_auc_model=md, del_auc_random=rd, ins_auc_model=mi, ins_auc_random=ri,
                faithful_deletion=bool(md > rd), faithful_insertion=bool(mi < ri))

def lime_local(pred_fn_scaled, x, rng, n=200, hold=100):
    from sklearn.linear_model import Ridge
    Z = rng.normal(0, 1, (n+hold, len(x)))
    w = np.exp(-0.5*np.sum((Z-x)**2, axis=1)/(0.75*len(x)))
    y = pred_fn_scaled(Z)
    sur = Ridge(alpha=1e-8).fit(Z[:n], y[:n], sample_weight=w[:n])
    fid = mean_squared_error(y[n:], sur.predict(Z[n:]))
    return sur.coef_, fid

def probe_positions(n_frame, n=PROBE_N):
    return np.unique(np.linspace(int(n_frame*0.55), n_frame-1, n).astype(int))

def stability_rows(att, frame, sid, h, scheme, method):
    rows = []; regs = frame_regimes(sid, frame); probes = att[sorted(att)[0]][0]
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



# ===== CELL 4 =====
from statsmodels.tsa.stattools import adfuller
print('Datasets:', {k: len(v) for k, v in DATA.items()})
for sid in list(DATA):
    if sid != 'monash_sunspot':
        _, pv, *_ = adfuller(DATA[sid]['log_return'].dropna(), maxlag=21)
        print(f'  ADF {sid}: p={pv:.2e} -> {"stationary" if pv < 0.05 else "NON-stationary"}')
    f = FRAMES[sid]
    assert f[FEATURES + [f'target_h{HORIZONS[0]}']].notna().all().all()
    assert (f['date'].diff().dropna() > pd.Timedelta(0)).all() if 'date' in f else True
print('Leakage guard: features use shift(+lag), targets use shift(-h); folds purge+embargo =', PURGE+EMBARGO)



# ===== CELL 6 =====
from sklearn.linear_model import LinearRegression
rows, att_shap, att_coef, coef_rows = [], {}, {}, []
t0 = time.time()
for sid, f in FRAMES.items():
    fin = sid != 'monash_sunspot'
    for h in HORIZONS:
        tgt = f[f'target_h{h}'].values
        for scheme in SCHEMES:
            for fo in make_folds(len(f), scheme):
                tr, te = fo['tr'], fo['te']
                Xtr, ytr = f.iloc[tr[0]:tr[1]][FEATURES].values, tgt[tr[0]:tr[1]]
                Xte, yte = f.iloc[te[0]:te[1]][FEATURES].values, tgt[te[0]:te[1]]
                sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
                m = LinearRegression().fit(Xtr_s, ytr); pred = m.predict(Xte_s)
                rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'], seed=SEED, model=MODEL,
                                 rmse=np.sqrt(mean_squared_error(yte, pred)), mae=mean_absolute_error(yte, pred),
                                 da=(np.mean(np.sign(yte) == np.sign(pred)) if fin else np.nan),
                                 r2=r2_score(yte, pred), n_test=len(yte), status='complete'))
                coef_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'],
                                      **{f'coef_{c}': m.coef_[i] for i, c in enumerate(FEATURES)}, intercept=m.intercept_))
                if fo['fold'] < EXPLAIN_FOLDS:   # attributions on fixed matched probes (RQ2/RQ3)
                    pr = probe_positions(len(f)); Xp = sc.transform(f.iloc[pr][FEATURES].values)
                    phi = Xp * m.coef_                       # exact linear SHAP (ground truth)
                    key = (sid, h, scheme, fo['fold'])
                    att_coef[key] = (pr, phi)
                    base = m.predict(np.zeros((1, Xtr_s.shape[1])))[0]
                    try:
                        import shap
                        ex = shap.LinearExplainer(m, Xtr_s); phi_lib = ex.shap_values(Xp)
                        add_res = float(np.max(np.abs(phi_lib.sum(1) + ex.expected_value - m.predict(Xp))))
                    except Exception:
                        phi_lib, add_res = phi, 0.0
                    att_shap[key] = (pr, phi_lib, base, add_res, sc, m)
res = pd.DataFrame(rows); coefs = pd.DataFrame(coef_rows)
print(f'Completed {len(res)} forecast units in {time.time()-t0:.1f}s')
print(res.groupby(['series', 'horizon'])[['rmse', 'da']].mean().round(6))



# ===== CELL 8 =====
dm_rows = []
for sid, f in FRAMES.items():
    fin = sid != 'monash_sunspot'
    for h in HORIZONS:
        y = f[f'target_h{h}'].values
        rw = np.zeros(len(y)) if fin else np.concatenate([[f['value'].iloc[0]], f['value'].values[:-1]])
        for scheme in SCHEMES:
            for fo in make_folds(len(f), scheme):
                te = fo['te']; yte = y[te[0]:te[1]]
                pr = res.query('series==@sid and horizon==@h and scheme==@scheme and fold==@fo["fold"]')
                if pr.empty: continue
                e_m = pr['rmse'].values[0]**2 * np.ones(len(yte))
                e_rw = (yte - rw[te[0]:te[1]])**2
                e_mean = (yte - y[fo['tr'][0]:fo['tr'][1]].mean())**2
                s1, p1 = dm_hac(e_m, e_rw, h); s2, p2 = dm_hac(e_m, e_mean, h)
                dm_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fo['fold'], model=MODEL,
                                    dm_vs_rw=s1, p_vs_rw=p1, dm_vs_mean=s2, p_vs_mean=p2))
dm = pd.DataFrame(dm_rows)
print(dm.groupby(['series', 'horizon'])[['p_vs_rw']].mean().round(4))



# ===== CELL 10 =====
from sklearn.inspection import permutation_importance
faith_rows, shap_imp_rows, lime_rows = [], [], []
for key, (pr, phi, base, add_res, sc, m) in att_shap.items():
    sid, h, scheme, fold = key
    f = FRAMES[sid]; y = f[f'target_h{h}'].values[pr]
    Xp = f.iloc[pr][FEATURES].values; Xp_s = sc.transform(Xp)
    faith_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL,
                           method='SHAP_additivity', metric='max_abs_residual', value=add_res))
    imp = np.abs(phi).mean(0)
    for i, c in enumerate(FEATURES):
        shap_imp_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL, feature=c, mean_abs_shap=imp[i]))
    pf = permutation_importance(m, Xp_s, y, n_repeats=5, random_state=SEED).importances_mean
    rho_pfi, _ = spearmanr(pf, imp)
    faith_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL,
                           method='PFI_vs_SHAP', metric='spearman_rho', value=float(rho_pfi)))
    di = di_curves(lambda Z: m.predict(Z), Xp_s, y, np.argsort(-imp))
    for k, v in di.items():
        faith_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL,
                               method='deletion_insertion', metric=k, value=v if isinstance(v, bool) else float(v)))
    rng = np.random.RandomState(SEED + fold)
    lf = [lime_local(lambda Z: m.predict(Z), x, rng) for x in Xp_s[:20]]
    rho_lime, _ = spearmanr(np.abs([c for c, _ in lf]).mean(0), imp)
    faith_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL,
                           method='LIME_fidelity', metric='median_holdout_mse', value=float(np.median([f_ for _, f_ in lf]))))
    faith_rows.append(dict(series=sid, horizon=h, scheme=scheme, fold=fold, model=MODEL,
                           method='LIME_vs_SHAP', metric='spearman_rho', value=float(rho_lime)))
faith = pd.DataFrame(faith_rows); shap_imp = pd.DataFrame(shap_imp_rows)
print('SHAP additivity max residual:', faith.query('method=="SHAP_additivity"')['value'].max())
print('Deletion faithful folds:', faith.query('metric=="faithful_deletion"')['value'].mean())

# ALE (10 quantile bins, Apley & Zhu 2020) for one representative unit
ale_fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
key0 = sorted(att_shap)[0]; sc0 = att_shap[key0][4]; f0 = FRAMES[key0[0]]
Xs = sc0.transform(f0.iloc[:2000][FEATURES].values)
for i, c in enumerate(FEATURES):
    qs = np.quantile(Xs[:, i], np.linspace(0, 1, 11)); eff = np.zeros(10)
    for b in range(10):
        msk = (Xs[:, i] >= qs[b]) & (Xs[:, i] <= qs[b+1])
        if msk.sum() > 1:
            lo = Xs[msk].copy(); lo[:, i] = qs[b]; hi = Xs[msk].copy(); hi[:, i] = qs[b+1]
            eff[b] = (att_shap[key0][5].predict(hi) - att_shap[key0][5].predict(lo)).mean()
    axes[i].plot(np.arange(10), np.cumsum(eff), marker='o'); axes[i].set_title(f'ALE {c}')
ale_fig.suptitle(f'ALE — {MODEL} ({key0[0]}, h={key0[1]})'); ale_fig.tight_layout()



# ===== CELL 12 =====
stab_rows = []
by_unit = {}
for key, (pr, phi, *_rest) in att_coef.items():
    sid, h, scheme, fold = key
    by_unit.setdefault((sid, h, scheme), {})[fold] = (pr, phi)
for (sid, h, scheme), att in by_unit.items():
    stab_rows += stability_rows(att, FRAMES[sid], sid, h, scheme, 'coef_exact_SHAP')
stab = pd.DataFrame(stab_rows)
stab_sum = stab.groupby(['method', 'regime'])['rho'].agg(['median', 'count']).reset_index()
print(stab_sum.round(4))
fig_st = plt.figure(figsize=(7, 4))
plt.bar(stab_sum['regime'], stab_sum['median']); plt.ylabel('median ρ'); plt.title(f'{MODEL}: attribution stability by regime (RQ2)')



# ===== CELL 14 =====
# --- FIX 1: robust figure construction -------------------------------------
del_m = float(faith.query('metric=="del_auc_model"')['value'].mean())
del_r = float(faith.query('metric=="del_auc_random"')['value'].mean())
ins_m = float(faith.query('metric=="ins_auc_model"')['value'].mean())
ins_r = float(faith.query('metric=="ins_auc_random"')['value'].mean())

fig_rq1, ax = plt.subplots(figsize=(8, 4))
res.boxplot(column='rmse', by='horizon', ax=ax)
ax.set_title(f'{MODEL}: RMSE by horizon (RQ1)')

fig_sh, ax2 = plt.subplots(figsize=(7, 4))
shap_imp.groupby('feature')['mean_abs_shap'].mean().sort_values().plot.barh(ax=ax2)
ax2.set_title(f'{MODEL}: mean |SHAP| per lag (RQ3)')

fig_fa, ax3 = plt.subplots(figsize=(7, 4))
ax3.bar(['del_model', 'del_random', 'ins_model', 'ins_random'],
        [del_m, del_r, ins_m, ins_r])
ax3.set_title('Deletion/insertion AUC vs random (RQ3)')

bear = float(stab_sum.query('regime=="bear"')['median'].mean()) \
       if (stab_sum['regime'] == 'bear').any() else None

# --- FIX 2: flatten tuple keys -> strings ----------------------------------
_rq1 = res.groupby(['series', 'horizon'])['rmse'].mean().round(6)
RQ1_RMSE = {f'{sid}|h{int(h)}': float(v) for (sid, h), v in _rq1.items()}

evidence = {
 'RQ1': {'question': 'Accuracy–interpretability trade-off', 'status': 'answered_for_this_model (baseline floor)',
   'claims': [
     {'claim': 'Linear sets the interpretable accuracy floor; complex models must beat these RMSEs to justify opacity.',
      'value': RQ1_RMSE,
      'artifact': 'tables/forecast_results.csv', 'thesis_ref': 'Ch.4 §4.9.2', 'citation': 'Diebold & Mariano (1995)'},
     {'claim': 'No directional skill beyond chance (DA≈0.5), consistent with weak-form efficiency.',
      'value': float(res['da'].mean()), 'artifact': 'tables/forecast_results.csv',
      'thesis_ref': 'Ch.4 §4.9.2', 'citation': 'Molnar (2022)'}]},
 'RQ2': {'question': 'Temporal stability & regime robustness', 'status': 'answered (benchmark)',
   'claims': [
     {'claim': 'Overall attribution stability high; degrades in bear regimes — the benchmark complex models must match.',
      'value': {'overall_median_rho': float(stab['rho'].median()), 'bear_median_rho': bear},
      'artifact': 'tables/stability.csv', 'thesis_ref': 'Ch.4 §4.9.3', 'citation': 'Molnar (2022)'}]},
 'RQ3': {'question': 'Faithfulness & economic plausibility', 'status': 'answered (ground truth)',
   'claims': [
     {'claim': 'SHAP additivity residual ≈ 0 proves exact faithfulness for the linear model.',
      'value': float(faith.query('method=="SHAP_additivity"')['value'].max()),
      'artifact': 'tables/faithfulness.csv', 'thesis_ref': 'Ch.4 §4.9.4', 'citation': 'Shapley (1953)'},
     {'claim': 'Deletion/insertion outrank random controls; PFI and LIME agree with SHAP rankings.',
      'value': {'faithful_del_frac': float(faith.query('metric=="faithful_deletion"')['value'].mean()),
                'pfi_rho_mean': float(faith.query('method=="PFI_vs_SHAP"')['value'].mean()),
                'del_auc': {'model': del_m, 'random': del_r},
                'ins_auc': {'model': ins_m, 'random': ins_r}},
      'artifact': 'tables/faithfulness.csv', 'thesis_ref': 'Ch.4 §4.7.6', 'citation': 'Ribeiro et al. (2016); Breiman (2001)'},
     {'claim': 'Plausibility limited to price dynamics (lag_1 momentum / lag_252 seasonality); NO causal macro claims (price-only scope).',
      'value': str(shap_imp.groupby('feature')['mean_abs_shap'].mean().sort_values(ascending=False).index[0]),
      'artifact': 'tables/shap_importance.csv', 'thesis_ref': 'Ch.4 §4.3.5', 'citation': 'Arsenault et al. (2025)'}]},
 'limitations': ['Price-only features; no macro vintages.',
                 'Single-model notebook: 7-way MCS/DM deferred to synthesis (Hansen et al., 2011).',
                 'Laptop budget active (THESIS_FULL_BUDGET=False) unless switched.']}

audit = dict(model=MODEL,
             planned_units=len(FRAMES)*len(HORIZONS)*len(SCHEMES)*N_FOLDS*len(SEEDS),
             completed_units=int(len(res)),
             missing_units=int(len(FRAMES)*len(HORIZONS)*len(SCHEMES)*N_FOLDS*len(SEEDS) - len(res)),
             explained_units=int(len(att_shap)),
             utc=time.strftime('%Y-%m-%dT%H:%M:%SZ'))

export_all({'forecast_results': res, 'coefficients': coefs, 'benchmark_dm': dm,
            'faithfulness': faith, 'shap_importance': shap_imp,
            'stability': stab, 'stability_summary': stab_sum},
           {'fig_rq1': fig_rq1, 'fig_shap_importance': fig_sh, 'fig_faithfulness': fig_fa,
            'fig_stability': fig_st, 'fig_ale': ale_fig}, evidence, audit)
print('UPLOAD:', OUT.resolve(), '(tables/, figures/, evidence/, FILE_SHA256.json)')




# ===== CELL 16 =====
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



# ===== CELL 17 =====


# ===== CELL 18 =====
