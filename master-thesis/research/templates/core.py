MODEL = '__MODEL__'
EXECUTION_VERSION = 'corrected-independent-rerun-v1'

import os, sys, json, time, hashlib, warnings, platform
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
RUN_STARTED = time.time()
SEED = 20261017
np.random.seed(SEED)
LAGS = [1, 5, 21, 252]
FEATURES = [f'lag_{l}' for l in LAGS]
THESIS_FULL_BUDGET = False
HORIZONS = [1, 5]
IS_DEEP = MODEL in ('LSTM', 'TFT')
N_FOLDS = 4 if IS_DEEP else 6
TEST_SPAN, MIN_TRAIN, PURGE, EMBARGO = 252, 500, 5, 5
SCHEMES = ['rolling', 'expanding']
SEEDS = [SEED, SEED + 1] if MODEL in ('RandomForest', 'XGBoost', 'LightGBM') else [SEED]
EXPLAIN_FOLDS = 3 if IS_DEEP else 4
PROBE_N = 30 if IS_DEEP else 40
LOOKBACK = 21 if IS_DEEP else 0
N_RANDOM_CTRL = 32
OUT = Path.cwd() / f'{MODEL}_study'
for sub in ('tables', 'figures', 'evidence'):
    (OUT / sub).mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def find_data(names):
    for base in (Path.cwd() / 'data', Path.cwd(), Path.cwd() / '..'):
        for name in names:
            candidate = base / name
            if candidate.exists():
                return candidate
    return None


INDEX_FILES = {
    'sp500': 'sp500_observations.csv',
    'dax': 'dax_observations.csv',
    'nasdaq': 'nasdaq_observations.csv',
    'djia': 'djia_observations.csv'
}
SUNSPOT_FILE = 'monash_sunspot_no_missing_T1_85babb464a3e.csv'
SUNSPOT_TSF = 'sunspot_dataset_without_missing_values.tsf'
DATA, PROVENANCE = {}, {'data': {}, 'execution_version': EXECUTION_VERSION}

# Only the supplied, frozen inputs are used. No network fallback is permitted.
for sid, filename in INDEX_FILES.items():
    path = find_data([filename])
    if path is None:
        raise FileNotFoundError(f'Required frozen input not found: {filename}')
    frame = pd.read_csv(path, parse_dates=['Date'])
    if frame['Date'].duplicated().any() or frame['Adj Close'].isna().any() or (frame['Adj Close'] <= 0).any():
        raise ValueError(f'Invalid adjusted-price input: {sid}')
    frame['log_return'] = np.log(frame['Adj Close'] / frame['Adj Close'].shift(1))
    DATA[sid] = frame.iloc[1:].dropna().reset_index(drop=True)
    PROVENANCE['data'][sid] = {'path': str(path), 'sha256': sha256(path)}

sunspot_path = find_data([SUNSPOT_FILE])
tsf_path = find_data([SUNSPOT_TSF])
if sunspot_path is None or tsf_path is None:
    raise FileNotFoundError('The traceable Monash CSV and original TSF are both required.')
sunspot = pd.read_csv(sunspot_path)
sunspot['date'] = pd.to_datetime(sunspot['date'])
sunspot = sunspot.rename(columns={'date': 'Date'}).sort_values('Date').reset_index(drop=True)
DATA['monash_sunspot'] = sunspot
PROVENANCE['data']['monash_sunspot'] = {'path': str(sunspot_path), 'sha256': sha256(sunspot_path)}
PROVENANCE['data']['monash_sunspot_tsf'] = {'path': str(tsf_path), 'sha256': sha256(tsf_path)}

# Verify the execution CSV against every aligned value in the supplied original TSF.
tsf_text = tsf_path.read_text()
data_line = next(line for line in tsf_text.splitlines() if line.startswith('T1:'))
_, start_text, values_text = data_line.split(':', 2)
start_text = start_text.replace('00-00-00', '00:00:00')
tsf_values = np.asarray([float(v) for v in values_text.split(',')], dtype=float)
tsf_dates = pd.date_range(pd.to_datetime(start_text), periods=len(tsf_values), freq='D')
tsf_frame = pd.DataFrame({'Date': tsf_dates, 'tsf_value': tsf_values})
sunspot_check = DATA['monash_sunspot'][['Date', 'value']].merge(tsf_frame, on='Date', how='left', validate='one_to_one')
SUNSPOT_AUDIT = {
    'tsf_rows': int(len(tsf_frame)),
    'execution_rows': int(len(DATA['monash_sunspot'])),
    'aligned_rows': int(sunspot_check['tsf_value'].notna().sum()),
    'value_mismatches': int((sunspot_check['value'].astype(float) != sunspot_check['tsf_value']).sum()),
    'source_start': str(tsf_frame['Date'].min().date()),
    'source_end': str(tsf_frame['Date'].max().date()),
    'execution_start': str(DATA['monash_sunspot']['Date'].min().date()),
    'execution_end': str(DATA['monash_sunspot']['Date'].max().date())
}
assert SUNSPOT_AUDIT['aligned_rows'] == len(DATA['monash_sunspot']) == 11109
assert SUNSPOT_AUDIT['value_mismatches'] == 0
PROVENANCE['sunspot_source_verification'] = SUNSPOT_AUDIT


def build_frames(sid):
    source = DATA[sid]
    financial = sid != 'monash_sunspot'
    column = 'log_return' if financial else 'value'
    frame = source.copy()
    frame['source_row'] = np.arange(len(frame), dtype=int)
    for lag in LAGS:
        frame[f'lag_{lag}'] = frame[column].shift(lag)
    for horizon in HORIZONS:
        if financial:
            frame[f'target_h{horizon}'] = sum(frame[column].shift(-i) for i in range(1, horizon + 1))
        else:
            frame[f'target_h{horizon}'] = frame[column].shift(-horizon)
    return frame.iloc[max(LAGS):-max(HORIZONS)].dropna().reset_index(drop=True)


FRAMES = {sid: build_frames(sid) for sid in DATA}


def make_folds(n, scheme):
    folds = []
    for fold_id in range(N_FOLDS):
        test_start = ((n - N_FOLDS * TEST_SPAN + fold_id * TEST_SPAN) if scheme == 'rolling'
                      else (MIN_TRAIN + PURGE + EMBARGO + fold_id * TEST_SPAN))
        test_end = min(test_start + TEST_SPAN, n)
        train_end = test_start - PURGE - EMBARGO
        train_start = max(0, train_end - MIN_TRAIN) if scheme == 'rolling' else 0
        if test_end - test_start >= 20 and train_end - train_start >= 100:
            folds.append({'fold': fold_id, 'scheme': scheme,
                          'tr': (train_start, train_end), 'te': (test_start, test_end)})
    return folds


def explanation_positions(frame, scheme):
    """Matched origins from the last explained test block through the frame end.

    Every position is out-of-sample for every explained fold, while the wider
    span preserves bear/bull/crisis coverage instead of restricting RQ2 to one
    potentially quiet test block.
    """
    folds = [fo for fo in make_folds(len(frame), scheme) if fo['fold'] < EXPLAIN_FOLDS]
    last = folds[-1]
    start = last['te'][0] + LOOKBACK
    stop = len(frame)
    positions = np.unique(np.linspace(start, stop - 1, PROBE_N).astype(int))
    assert len(positions) == PROBE_N
    assert positions.min() >= last['te'][0] + LOOKBACK
    return positions


def regimes(prices, window=63):
    returns = np.diff(np.log(prices))
    mean = pd.Series(returns).rolling(window).mean()
    std = pd.Series(returns).rolling(window).std()
    labels = pd.Series('boundary', index=range(len(returns)))
    labels[(mean > mean.quantile(.6)) & (std < std.quantile(.8))] = 'bull'
    labels[mean < mean.quantile(.4)] = 'bear'
    labels[std > std.quantile(.95)] = 'crisis'
    return labels


def frame_regimes(sid, frame):
    if sid == 'monash_sunspot':
        return pd.Series('na', index=frame.index)
    source = DATA[sid]
    labels = regimes(source['Adj Close'].values)
    positions = np.clip(np.searchsorted(source['Date'].values, frame['Date'].values) - 1, 0, len(labels) - 1)
    return pd.Series(labels.values[positions], index=frame.index)


def target_date(sid, frame_position, horizon):
    frame = FRAMES[sid]
    source_row = int(frame['source_row'].iloc[frame_position]) + horizon
    return DATA[sid]['Date'].iloc[source_row]


def add_predictions(store, sid, horizon, scheme, fold, origin_positions, predictions, train_mean):
    frame = FRAMES[sid]
    financial = sid != 'monash_sunspot'
    origin_positions = np.asarray(origin_positions, dtype=int)
    predictions = np.asarray(predictions, dtype=float)
    assert len(origin_positions) == len(predictions)
    assert np.isfinite(predictions).all()
    for position, prediction in zip(origin_positions, predictions):
        realized = float(frame[f'target_h{horizon}'].iloc[position])
        origin_source_row = int(frame['source_row'].iloc[position])
        expected_target_date = DATA[sid]['Date'].iloc[origin_source_row + horizon]
        persistence = 0.0 if financial else float(frame['value'].iloc[position])
        store.append({
            'model': MODEL, 'series': sid, 'horizon': int(horizon), 'scheme': scheme,
            'fold': int(fold), 'origin_index': int(position),
            'origin_date': frame['Date'].iloc[position].strftime('%Y-%m-%d'),
            'target_date': expected_target_date.strftime('%Y-%m-%d'),
            'y_true': realized, 'y_pred': float(prediction),
            'random_walk_pred': float(persistence), 'train_mean_pred': float(train_mean),
            'seed': SEED
        })


def benchmark_semantics_audit(predictions):
    """Verify every fold's exported training mean against its supervised target.

    This check intentionally uses the horizon-specific target column and the
    exact frame-level training interval. A raw univariate training mean is not
    an accepted substitute, including for ARIMA.
    """
    max_abs_error = 0.0
    checked_groups = 0
    for (sid, horizon, scheme, fold_id), group in predictions.groupby(
            ['series', 'horizon', 'scheme', 'fold'], sort=True):
        frame = FRAMES[sid]
        fold_lookup = {fold['fold']: fold for fold in make_folds(len(frame), scheme)}
        assert fold_id in fold_lookup, (sid, horizon, scheme, fold_id)
        tr = fold_lookup[fold_id]['tr']
        expected = float(frame[f'target_h{horizon}'].iloc[tr[0]:tr[1]].mean())
        exported = group['train_mean_pred'].dropna().unique()
        assert len(exported) == 1, (sid, horizon, scheme, fold_id, exported)
        error = abs(float(exported[0]) - expected)
        max_abs_error = max(max_abs_error, error)
        assert np.isclose(float(exported[0]), expected, rtol=1e-12, atol=1e-12), (
            sid, horizon, scheme, fold_id, float(exported[0]), expected)
        checked_groups += 1
    return {'all_passed': True, 'checked_fold_groups': int(checked_groups),
            'max_abs_error': float(max_abs_error), 'rtol': 1e-12, 'atol': 1e-12}


def finalize_predictions(rows):
    predictions = pd.DataFrame(rows)
    predictions['error'] = predictions['y_true'] - predictions['y_pred']
    predictions['squared_error'] = predictions['error'] ** 2
    predictions['absolute_error'] = predictions['error'].abs()
    metric_rows = []
    keys = ['model', 'series', 'horizon', 'scheme', 'fold']
    for values, group in predictions.groupby(keys, sort=True):
        model, sid, horizon, scheme, fold = values
        metric_rows.append({
            'model': model, 'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
            'seed': SEED, 'rmse': float(np.sqrt(mean_squared_error(group['y_true'], group['y_pred']))),
            'mae': float(mean_absolute_error(group['y_true'], group['y_pred'])),
            'da': (float(np.mean(np.sign(group['y_true']) == np.sign(group['y_pred'])))
                   if sid != 'monash_sunspot' else np.nan),
            'r2': float(r2_score(group['y_true'], group['y_pred'])),
            'n_test': int(len(group)), 'status': 'complete'
        })
    return predictions, pd.DataFrame(metric_rows)


def dm_hac(loss_a, loss_b, horizon):
    """HAC Diebold–Mariano test on observation-level losses, with h-1 lags."""
    loss_a, loss_b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    differential = loss_a - loss_b
    n = len(differential)
    if n < 20:
        return np.nan, np.nan
    centered = differential - differential.mean()
    long_run_variance = np.dot(centered, centered) / n
    for lag in range(1, min(max(horizon - 1, 0), 9) + 1):
        covariance = np.dot(centered[lag:], centered[:-lag]) / n
        long_run_variance += 2 * covariance
    if not np.isfinite(long_run_variance) or long_run_variance <= 0:
        return np.nan, np.nan
    statistic = differential.mean() / np.sqrt(long_run_variance / n)
    p_value = 2 * (1 - norm.cdf(abs(statistic)))
    return float(statistic), float(p_value)


def bh_adjust(p_values):
    p = np.asarray(p_values, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def benchmark_tests(predictions):
    rows = []
    for (sid, horizon, scheme), group in predictions.groupby(['series', 'horizon', 'scheme']):
        group = group.sort_values('origin_index')
        for label, column in [('random_walk_or_persistence', 'random_walk_pred'),
                              ('training_mean', 'train_mean_pred')]:
            benchmark_loss = (group['y_true'] - group[column]) ** 2
            statistic, p_value = dm_hac(group['squared_error'], benchmark_loss, horizon)
            rows.append({'model': MODEL, 'series': sid, 'horizon': horizon, 'scheme': scheme,
                         'comparison': label, 'n': int(len(group)), 'dm': statistic, 'p_raw': p_value})
    result = pd.DataFrame(rows)
    result['p_bh_within_series_horizon_scheme'] = np.nan
    for _, indices in result.groupby(['series', 'horizon', 'scheme']).groups.items():
        result.loc[indices, 'p_bh_within_series_horizon_scheme'] = bh_adjust(result.loc[indices, 'p_raw'])
    return result


def _auc(values):
    return float(np.trapezoid(values, dx=1.0) / (len(values) - 1))


def di_curves(pred_fn, X, y, order, n_random=N_RANDOM_CTRL):
    n_features = X.shape[1]
    cases = [('model', order)] + [(f'r{i}', np.random.RandomState(SEED + i).permutation(n_features))
                                  for i in range(n_random)]
    output = {}
    for tag, ordering in cases:
        deletion_loss, insertion_loss = [], []
        for k in range(n_features + 1):
            deleted = X.copy()
            for feature in ordering[:k]:
                deleted[:, feature] = 0
            deletion_loss.append(mean_squared_error(y, pred_fn(deleted)))
            inserted = np.zeros_like(X)
            for feature in ordering[:k]:
                inserted[:, feature] = X[:, feature]
            insertion_loss.append(mean_squared_error(y, pred_fn(inserted)))
        output[tag] = (_auc(deletion_loss), _auc(insertion_loss))
    model_deletion, model_insertion = output['model']
    random_deletion = np.mean([output[f'r{i}'][0] for i in range(n_random)])
    random_insertion = np.mean([output[f'r{i}'][1] for i in range(n_random)])
    return {'del_auc_model': model_deletion, 'del_auc_random': random_deletion,
            'ins_auc_model': model_insertion, 'ins_auc_random': random_insertion,
            'faithful_deletion': bool(model_deletion > random_deletion),
            'faithful_insertion': bool(model_insertion < random_insertion)}


def di_curves_grouped(pred_fn, Z, y, order, n_random=N_RANDOM_CTRL):
    """Deep interventions permute the same four lag groups for model and random orderings."""
    n = len(Z)
    X = Z.reshape(n, LOOKBACK, len(FEATURES))
    cases = [('model', order)] + [(f'r{i}', np.random.RandomState(SEED + i).permutation(len(FEATURES)))
                                  for i in range(n_random)]
    output = {}
    for tag, ordering in cases:
        deletion_loss, insertion_loss = [], []
        for k in range(len(FEATURES) + 1):
            deleted = X.copy()
            for feature in ordering[:k]:
                deleted[:, :, feature] = 0
            deletion_loss.append(mean_squared_error(y, pred_fn(deleted.reshape(n, -1))))
            inserted = np.zeros_like(X)
            for feature in ordering[:k]:
                inserted[:, :, feature] = X[:, :, feature]
            insertion_loss.append(mean_squared_error(y, pred_fn(inserted.reshape(n, -1))))
        output[tag] = (_auc(deletion_loss), _auc(insertion_loss))
    model_deletion, model_insertion = output['model']
    random_deletion = np.mean([output[f'r{i}'][0] for i in range(n_random)])
    random_insertion = np.mean([output[f'r{i}'][1] for i in range(n_random)])
    return {'del_auc_model': model_deletion, 'del_auc_random': random_deletion,
            'ins_auc_model': model_insertion, 'ins_auc_random': random_insertion,
            'faithful_deletion': bool(model_deletion > random_deletion),
            'faithful_insertion': bool(model_insertion < random_insertion)}


def lime_local(pred_fn, x, rng, n=200, holdout=100):
    from sklearn.linear_model import Ridge
    Z = rng.normal(0, 1, (n + holdout, len(x)))
    weights = np.exp(-0.5 * np.sum((Z - x) ** 2, axis=1) / (0.75 * len(x)))
    y = pred_fn(Z)
    surrogate = Ridge(alpha=1e-8).fit(Z[:n], y[:n], sample_weight=weights[:n])
    fidelity = mean_squared_error(y[n:], surrogate.predict(Z[n:]))
    return surrogate.coef_, fidelity


def permutation_importance_groups(pred_fn, X, y, rng):
    baseline = mean_squared_error(y, pred_fn(X))
    values = []
    if IS_DEEP:
        n = len(X)
        sequences = X.reshape(n, LOOKBACK, len(FEATURES))
        for feature in range(len(FEATURES)):
            changed = sequences.copy()
            changed[:, :, feature] = changed[rng.permutation(n), :, feature]
            values.append(mean_squared_error(y, pred_fn(changed.reshape(n, -1))) - baseline)
    else:
        for feature in range(X.shape[1]):
            changed = X.copy()
            changed[:, feature] = rng.permutation(changed[:, feature])
            values.append(mean_squared_error(y, pred_fn(changed)) - baseline)
    return np.asarray(values)


def run_posthoc_diagnostics(explanations):
    faith_rows, importance_rows, attribution_rows, seed_rows = [], [], [], []
    method = ('KernelSHAP_lag_aggregated' if IS_DEEP else
              'exact_linear_SHAP' if MODEL == 'Linear' else 'TreeSHAP_full_ensemble')
    for key in sorted(explanations):
        sid, horizon, scheme, fold = key
        item = explanations[key]
        X, y, phi, positions = item['X'], item['y'], item['phi'], item['pr']
        pred_fn = item['pred_fn']
        importance = np.abs(phi).mean(axis=0)
        faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                           'model': MODEL, 'method': 'SHAP_additivity', 'metric': 'max_abs_residual',
                           'value': float(item['additivity_residual'])})
        for j, feature in enumerate(FEATURES):
            importance_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                                    'model': MODEL, 'feature': feature, 'mean_abs_shap': float(importance[j])})
        rng_pfi = np.random.RandomState(SEED + fold)
        pfi = permutation_importance_groups(pred_fn, X, y, rng_pfi)
        rho_pfi, _ = spearmanr(pfi, importance)
        faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                           'model': MODEL, 'method': 'PFI_vs_SHAP', 'metric': 'spearman_rho',
                           'value': float(rho_pfi)})
        intervention = (di_curves_grouped(pred_fn, X, y, np.argsort(-importance)) if IS_DEEP
                        else di_curves(pred_fn, X, y, np.argsort(-importance)))
        for metric, value in intervention.items():
            faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                               'model': MODEL, 'method': 'deletion_insertion_grouped' if IS_DEEP else 'deletion_insertion',
                               'metric': metric, 'value': value if isinstance(value, bool) else float(value)})
        rng = np.random.RandomState(SEED + fold)
        local = [lime_local(pred_fn, x, rng) for x in X[:min(20, len(X))]]
        coefficients = np.asarray([entry[0] for entry in local])
        if IS_DEEP:
            surrogate_importance = np.abs(coefficients.reshape(len(coefficients), LOOKBACK, len(FEATURES))).mean(axis=(0, 1))
        else:
            surrogate_importance = np.abs(coefficients).mean(axis=0)
        rho_lime, _ = spearmanr(surrogate_importance, importance)
        faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                           'model': MODEL, 'method': 'LIME_fidelity', 'metric': 'median_holdout_mse',
                           'value': float(np.median([entry[1] for entry in local]))})
        faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                           'model': MODEL, 'method': 'LIME_vs_SHAP', 'metric': 'spearman_rho',
                           'value': float(rho_lime)})
        if item.get('native') is not None:
            native_importance = item['native'].mean(axis=0)
            rho_native, _ = spearmanr(native_importance, importance)
            faith_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                               'model': MODEL, 'method': 'native_vs_posthoc', 'metric': 'spearman_rho',
                               'value': float(rho_native)})
        regimes_for_frame = frame_regimes(sid, FRAMES[sid])
        for probe_number, position in enumerate(positions):
            for j, feature in enumerate(FEATURES):
                attribution_rows.append({
                    'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold, 'model': MODEL,
                    'method': method, 'probe': probe_number, 'origin_index': int(position),
                    'origin_date': FRAMES[sid]['Date'].iloc[position].strftime('%Y-%m-%d'),
                    'target_date': target_date(sid, position, horizon).strftime('%Y-%m-%d'),
                    'regime': regimes_for_frame.iloc[position], 'feature': feature,
                    'attribution': float(phi[probe_number, j])
                })
            if item.get('phi_seeds') is not None and len(item['phi_seeds']) == 2:
                rho_seed, _ = spearmanr(item['phi_seeds'][0][probe_number], item['phi_seeds'][1][probe_number])
                seed_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                                  'model': MODEL, 'probe': probe_number, 'origin_index': int(position),
                                  'regime': regimes_for_frame.iloc[position], 'rho': float(rho_seed),
                                  'top1_match': bool(np.argmax(np.abs(item['phi_seeds'][0][probe_number])) ==
                                                     np.argmax(np.abs(item['phi_seeds'][1][probe_number])))})
    return (pd.DataFrame(faith_rows), pd.DataFrame(importance_rows), pd.DataFrame(attribution_rows),
            pd.DataFrame(seed_rows))


def stability_analysis(explanations, value_key='phi', method=None):
    if method is None:
        method = ('KernelSHAP_lag_aggregated' if IS_DEEP else
                  'exact_linear_SHAP' if MODEL == 'Linear' else 'TreeSHAP_full_ensemble')
    rows = []
    grouped = {}
    for (sid, horizon, scheme, fold), item in explanations.items():
        grouped.setdefault((sid, horizon, scheme), {})[fold] = item
    for (sid, horizon, scheme), units in grouped.items():
        regimes_for_frame = frame_regimes(sid, FRAMES[sid])
        for first, second in zip(sorted(units), sorted(units)[1:]):
            a, b = units[first], units[second]
            assert np.array_equal(a['pr'], b['pr'])
            for probe_number, position in enumerate(a['pr']):
                vector_a = np.asarray(a[value_key][probe_number], float)
                vector_b = np.asarray(b[value_key][probe_number], float)
                rho, _ = spearmanr(vector_a, vector_b)
                denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
                cosine = float(np.dot(vector_a, vector_b) / denominator) if denominator > 0 else np.nan
                rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'model': MODEL,
                             'method': method, 'regime': regimes_for_frame.iloc[position],
                             'fold_pair': f'{first}->{second}', 'probe': probe_number,
                             'origin_index': int(position), 'rho': float(rho), 'cosine': cosine,
                             'top1_match': bool(np.argmax(np.abs(vector_a)) == np.argmax(np.abs(vector_b)))})
    stability = pd.DataFrame(rows)
    if len(stability):
        summary = (stability.groupby(['method', 'regime'])
                   .agg(rho_median=('rho', 'median'), cosine_median=('cosine', 'median'),
                        top1_match_rate=('top1_match', 'mean'), count=('rho', 'size')).reset_index())
    else:
        summary = pd.DataFrame(columns=['method', 'regime', 'rho_median', 'cosine_median', 'top1_match_rate', 'count'])
    return stability, summary


def standard_figures(fold_metrics, importance, faithfulness, stability_summary):
    figures = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, financial, title in [(axes[0], True, 'Financial RMSE'), (axes[1], False, 'Sunspot RMSE')]:
        subset = fold_metrics[(fold_metrics['series'] == 'monash_sunspot') != financial]
        values = [subset[subset['horizon'] == h]['rmse'].values for h in HORIZONS]
        ax.boxplot(values, tick_labels=[f'h={h}' for h in HORIZONS])
        ax.set_title(title)
        ax.set_ylabel('Fold RMSE')
    fig.suptitle(f'{MODEL}: predictive error (RQ1)')
    fig.tight_layout()
    figures['fig_rq1'] = fig

    if len(importance):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, financial, title in [(axes[0], True, 'Financial'), (axes[1], False, 'Sunspot')]:
            subset = importance[(importance['series'] == 'monash_sunspot') != financial]
            means = subset.groupby('feature')['mean_abs_shap'].mean().reindex(FEATURES)
            ax.barh(FEATURES, means.values)
            ax.set_title(title)
            ax.set_xlabel('Mean absolute attribution')
        fig.suptitle(f'{MODEL}: attribution magnitude (RQ3)')
        fig.tight_layout()
        figures['fig_shap_importance'] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    success = []
    for metric in ['faithful_deletion', 'faithful_insertion']:
        subset = faithfulness[faithfulness['metric'] == metric]['value']
        success.append(float(subset.astype(float).mean()) if len(subset) else np.nan)
    ax.bar(['Deletion', 'Insertion'], success)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction better than random mean')
    ax.set_title(f'{MODEL}: grouped intervention diagnostics (RQ3)')
    figures['fig_faithfulness'] = fig

    fig, ax = plt.subplots(figsize=(9, 4))
    if len(stability_summary):
        methods = list(stability_summary['method'].unique())
        regimes_order = ['bear', 'boundary', 'bull', 'crisis', 'na']
        regimes_present = [r for r in regimes_order if r in set(stability_summary['regime'])]
        width = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            subset = stability_summary[stability_summary['method'] == method].set_index('regime')
            values = [subset['rho_median'].get(regime, np.nan) for regime in regimes_present]
            ax.bar(np.arange(len(regimes_present)) + i * width, values, width=width, label=method)
        ax.set_xticks(np.arange(len(regimes_present)) + 0.4 * (len(methods) - 1) / max(len(methods), 1))
        ax.set_xticklabels(regimes_present)
        ax.legend(fontsize=8)
    ax.set_ylabel('Median Spearman correlation')
    ax.set_title(f'{MODEL}: attribution stability (RQ2)')
    fig.tight_layout()
    figures['fig_stability'] = fig
    return figures


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def standard_export(predictions, fold_metrics, benchmark_dm, faithfulness, importance,
                    attributions, stability, stability_summary, figures,
                    extra_tables=None, extra_evidence=None, correction_details=None):
    extra_tables = extra_tables or {}
    correction_details = correction_details or []
    packages = {}
    for package in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'statsmodels', 'shap',
                    'xgboost', 'lightgbm', 'torch', 'nbformat', 'nbclient']:
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            pass
    expected_fold_rows = len(FRAMES) * len(HORIZONS) * len(SCHEMES) * N_FOLDS
    expected_explained = len(FRAMES) * len(HORIZONS) * len(SCHEMES) * EXPLAIN_FOLDS
    explained_units = int(attributions[['series', 'horizon', 'scheme', 'fold']].drop_duplicates().shape[0]) if len(attributions) else 0
    benchmark_semantics = benchmark_semantics_audit(predictions)
    validations = {
        'all_planned_fold_units_complete': len(fold_metrics) == expected_fold_rows,
        'all_explained_units_complete': explained_units == expected_explained,
        'predictions_finite': bool(np.isfinite(predictions[['y_true', 'y_pred']]).all().all()),
        'prediction_metrics_recompute': bool(np.allclose(
            fold_metrics.set_index(['series', 'horizon', 'scheme', 'fold'])['rmse'].sort_index().values,
            predictions.groupby(['series', 'horizon', 'scheme', 'fold'])['squared_error'].mean().pipe(np.sqrt).sort_index().values)),
        'sunspot_matches_original_tsf': SUNSPOT_AUDIT['value_mismatches'] == 0,
        'observation_level_predictions_exported': len(predictions) > 0,
        'no_missing_prediction_targets': not predictions[['origin_date', 'target_date']].isna().any().any(),
        'training_mean_matches_supervised_fold_target': benchmark_semantics['all_passed']
    }
    assert all(validations.values()), validations
    native_rmse = {f'{sid}|h{int(h)}': float(value) for (sid, h), value in
                   fold_metrics.groupby(['series', 'horizon'])['rmse'].mean().items()}
    stability_values = {f'{method}|{regime}': float(value) for (method, regime), value in
                        stability_summary.set_index(['method', 'regime'])['rho_median'].items()} if len(stability_summary) else {}
    additivity = faithfulness[faithfulness['metric'] == 'max_abs_residual']['value']
    evidence = {
        'RQ1': {'question': 'Accuracy–interpretability trade-off', 'status': 'model_execution_complete_cross_model_answer_in_synthesis',
                'claims': [{'claim': 'Native-coverage fold metrics are exported with observation-level predictions; seven-way claims use only matched origins in the cross-model synthesis.',
                            'value': native_rmse, 'artifact': 'tables/fold_metrics.csv and tables/observation_predictions.csv',
                            'thesis_ref': 'Chapters 4–6', 'citation': 'Diebold & Mariano (1995)'}]},
        'RQ2': {'question': 'Temporal stability and regime robustness', 'status': 'model_execution_complete_cross_model_answer_in_synthesis',
                'claims': [{'claim': 'Consecutive-fold attributions use identical, contiguous, out-of-sample probe origins; Spearman, cosine, and top-feature agreement are exported.',
                            'value': stability_values, 'artifact': 'tables/stability.csv',
                            'thesis_ref': 'Chapters 4–6', 'citation': 'Molnar (2022)'}]},
        'RQ3': {'question': 'Faithfulness and economic drivers', 'status': 'model_execution_complete_economic_identification_bounded',
                'claims': [{'claim': 'Faithfulness is evaluated diagnostic by diagnostic; no feature ranking is interpreted as a causal economic driver.',
                            'value': {'max_shap_additivity_residual': float(additivity.max()) if len(additivity) else None},
                            'artifact': 'tables/faithfulness.csv', 'thesis_ref': 'Chapters 4–6',
                            'citation': 'Shapley (1953); Ribeiro et al. (2016)'}]},
        'limitations': ['Only lagged prices or native Sunspot values are available; causal economic-driver identification is not possible.',
                        'Regime labels are ex post data-driven strata, not independently dated historical crises.',
                        'Zero-baseline interventions and synthetic local-surrogate samples can leave the natural data manifold.',
                        'The simplified TFT adapter is not the full PyTorch Forecasting architecture.' if MODEL == 'TFT' else 'Model-specific limitations are documented in the thesis.']
    }
    if extra_evidence:
        evidence = json_safe({**evidence, **extra_evidence})
    audit = {
        'model': MODEL, 'execution_version': EXECUTION_VERSION, 'executed_in_this_environment': True,
        'run_started_utc': RUN_STARTED_UTC, 'run_finished_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.time() - RUN_STARTED, 'planned_fold_units': int(expected_fold_rows),
        'completed_fold_units': int(len(fold_metrics)), 'missing_fold_units': int(expected_fold_rows - len(fold_metrics)),
        'observation_prediction_rows': int(len(predictions)), 'planned_explained_units': int(expected_explained),
        'completed_explained_units': explained_units, 'validation_checks': validations,
        'benchmark_semantics': benchmark_semantics
    }
    provenance = dict(PROVENANCE)
    provenance.update({
        'environment': {'python': sys.version, 'platform': platform.platform(), 'processor': platform.processor(),
                        'cpu_count': os.cpu_count(), 'packages': packages},
        'seed': SEED, 'seeds': SEEDS,
        'budgets': {'horizons': HORIZONS, 'folds': N_FOLDS, 'schemes': SCHEMES,
                    'explained_folds': EXPLAIN_FOLDS, 'probe_n': PROBE_N,
                    'thesis_full': THESIS_FULL_BUDGET},
        'documented_corrections': [
            'Use the supplied raw Monash TSF-traceable Sunspot values and verify every aligned value.',
            'Export every observation-level prediction with aligned origin and target dates.',
            'Use observation-level squared errors in HAC Diebold–Mariano tests.',
            'Use identical contiguous out-of-sample probe origins across explained folds.',
            'Extend probes from the last explained test block through the frame end to retain regime diversity without entering any explained training set.',
            'Use four lag groups for both model-ranked and random deep interventions.'
        ] + correction_details
    })
    tables = {
        'observation_predictions': predictions, 'fold_metrics': fold_metrics, 'benchmark_dm': benchmark_dm,
        'faithfulness': faithfulness, 'shap_importance': importance, 'attributions': attributions,
        'stability': stability, 'stability_summary': stability_summary
    }
    tables.update(extra_tables)
    for name, table in tables.items():
        table.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
    for name, figure in figures.items():
        figure.savefig(OUT / 'figures' / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.close(figure)
    (OUT / 'evidence' / 'evidence_statement.json').write_text(json.dumps(json_safe(evidence), indent=2, default=str))
    md = [f'# Executed evidence statement — {MODEL}', '',
          f'Execution version: `{EXECUTION_VERSION}`', '',
          'This notebook was executed in the recorded environment. Cross-model conclusions are defined only after matched-origin synthesis.', '']
    for rq in ('RQ1', 'RQ2', 'RQ3'):
        md += [f'## {rq}', f"Status: {evidence[rq]['status']}", '']
        for claim in evidence[rq]['claims']:
            md += [f"- {claim['claim']}", f"  - Value: `{claim['value']}`", f"  - Artifact: `{claim['artifact']}`", '']
    md += ['## Limitations', ''] + [f'- {item}' for item in evidence['limitations']]
    (OUT / 'evidence' / 'evidence_statement.md').write_text('\n'.join(md))
    (OUT / 'evidence' / 'completion_audit.json').write_text(json.dumps(json_safe(audit), indent=2, default=str))
    (OUT / 'evidence' / 'provenance.json').write_text(json.dumps(json_safe(provenance), indent=2, default=str))
    hashes = {str(path.relative_to(OUT)): sha256(path) for path in sorted(OUT.rglob('*'))
              if path.is_file() and path.name != 'FILE_SHA256.json'}
    (OUT / 'FILE_SHA256.json').write_text(json.dumps(hashes, indent=2))
    print(f'EXPORT COMPLETE: {OUT.resolve()}')
    print(json.dumps(json_safe(audit), indent=2, default=str))
    return audit
