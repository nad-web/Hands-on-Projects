# Independent common-origin synthesis of the seven executed model notebooks.
from pathlib import Path
import json, hashlib, itertools, platform, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path.cwd()
OUT = BASE / 'cross_model'
for sub in ('tables', 'figures', 'evidence'):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
MODEL_DIRS = {'Linear': 'Linear_study', 'ARIMA': 'ARIMA_study', 'RandomForest': 'RandomForest_study',
              'XGBoost': 'XGBoost_study', 'LightGBM': 'LightGBM_study', 'LSTM': 'LSTM_study', 'TFT': 'TFT_study'}
MODEL_ORDER = list(MODEL_DIRS)
FINANCIAL = ['sp500', 'dax', 'nasdaq', 'djia']
HORIZONS = [1, 5]
SCHEMES = ['rolling', 'expanding']
Q = 0.05


def read_table(model, name):
    path = BASE / MODEL_DIRS[model] / 'tables' / f'{name}.csv'
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def dm_hac_observation(loss_a, loss_b, horizon, segments=None):
    differential = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(differential)
    centered = differential - differential.mean()
    if segments is None:
        segments = np.zeros(n, dtype=int)
    segments = np.asarray(segments)
    # Estimate autocovariances only within contiguous matched forecast blocks.
    variance = 0.0
    max_lag = min(max(horizon - 1, 0), 9)
    covariance_sums = np.zeros(max_lag + 1)
    for segment in np.unique(segments):
        block = centered[segments == segment]
        covariance_sums[0] += np.dot(block, block)
        for lag in range(1, max_lag + 1):
            covariance_sums[lag] += np.dot(block[lag:], block[:-lag]) if len(block) > lag else 0.0
    variance = covariance_sums[0] / n + 2 * covariance_sums[1:].sum() / n
    if n < 20 or not np.isfinite(variance) or variance <= 0:
        return np.nan, np.nan
    statistic = differential.mean() / np.sqrt(variance / n)
    return float(statistic), float(2 * (1 - norm.cdf(abs(statistic))))


def bh(p_values):
    p = np.asarray(p_values, float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


prediction_parts = []
for model in MODEL_ORDER:
    part = read_table(model, 'observation_predictions')
    assert set(part['model']) == {model}
    prediction_parts.append(part)
predictions = pd.concat(prediction_parts, ignore_index=True)
assert len(predictions) == 188160, len(predictions)
assert not predictions.duplicated(['model', 'series', 'horizon', 'scheme', 'fold', 'origin_index']).any()

origin_keys = ['series', 'horizon', 'scheme', 'origin_index', 'origin_date', 'target_date']
model_counts = predictions.groupby(origin_keys)['model'].nunique().rename('n_models').reset_index()
common_origins = model_counts[model_counts['n_models'] == 7][origin_keys]
matched = predictions.merge(common_origins, on=origin_keys, how='inner', validate='many_to_one')
deep_folds = predictions[predictions['model'] == 'LSTM'][origin_keys + ['fold']].rename(columns={'fold': 'matched_fold'})
matched = matched.merge(deep_folds, on=origin_keys, how='left', validate='many_to_one')
assert len(common_origins) == 18480, len(common_origins)
assert len(matched) == 129360, len(matched)
counts = matched.groupby(['series', 'horizon', 'scheme'])['origin_index'].nunique()
assert (counts == 924).all(), counts
fold_counts = matched.groupby(['series', 'horizon', 'scheme', 'matched_fold'])['origin_index'].nunique()
assert (fold_counts == 231).all(), fold_counts
assert matched.groupby(origin_keys)['model'].nunique().eq(7).all()

metric_rows = []
for (model, sid, horizon, scheme), group in matched.groupby(['model', 'series', 'horizon', 'scheme']):
    group = group.sort_values('origin_index')
    persistence_rmse = np.sqrt(mean_squared_error(group['y_true'], group['random_walk_pred']))
    mean_rmse = np.sqrt(mean_squared_error(group['y_true'], group['train_mean_pred']))
    metric_rows.append({'model': model, 'series': sid, 'horizon': horizon, 'scheme': scheme,
                        'n': len(group), 'rmse': np.sqrt(mean_squared_error(group['y_true'], group['y_pred'])),
                        'mae': mean_absolute_error(group['y_true'], group['y_pred']),
                        'r2': r2_score(group['y_true'], group['y_pred']),
                        'da': np.mean(np.sign(group['y_true']) == np.sign(group['y_pred'])) if sid != 'monash_sunspot' else np.nan,
                        'persistence_rmse': persistence_rmse, 'train_mean_rmse': mean_rmse,
                        'rmse_relative_to_persistence': np.sqrt(mean_squared_error(group['y_true'], group['y_pred'])) / persistence_rmse,
                        'rmse_relative_to_train_mean': np.sqrt(mean_squared_error(group['y_true'], group['y_pred'])) / mean_rmse})
matched_metrics = pd.DataFrame(metric_rows)
matched_metrics['mean_rmse_rank'] = matched_metrics.groupby(['series', 'horizon', 'scheme'])['rmse'].rank(method='min')

fold_metric_rows = []
for (model, sid, horizon, scheme, fold), group in matched.groupby(['model', 'series', 'horizon', 'scheme', 'matched_fold']):
    fold_metric_rows.append({'model': model, 'series': sid, 'horizon': horizon, 'scheme': scheme, 'matched_fold': fold,
                             'n': len(group), 'rmse': np.sqrt(mean_squared_error(group['y_true'], group['y_pred'])),
                             'mae': mean_absolute_error(group['y_true'], group['y_pred']),
                             'r2': r2_score(group['y_true'], group['y_pred']),
                             'da': np.mean(np.sign(group['y_true']) == np.sign(group['y_pred'])) if sid != 'monash_sunspot' else np.nan})
matched_fold_metrics = pd.DataFrame(fold_metric_rows)

pair_rows = []
for (sid, horizon, scheme), cell in matched.groupby(['series', 'horizon', 'scheme']):
    wide = cell.pivot(index='origin_index', columns='model', values='squared_error').dropna()
    segment_map = cell.drop_duplicates('origin_index').set_index('origin_index')['matched_fold']
    segments = segment_map.loc[wide.index].values
    assert list(wide.columns) == sorted(wide.columns)
    assert len(wide) == 924
    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
        statistic, p_value = dm_hac_observation(wide[model_a], wide[model_b], horizon, segments)
        pair_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'model_a': model_a, 'model_b': model_b,
                          'n': len(wide), 'mean_squared_error_a': wide[model_a].mean(), 'mean_squared_error_b': wide[model_b].mean(),
                          'dm_statistic_a_minus_b': statistic, 'p_raw': p_value,
                          'lower_loss_model': model_a if wide[model_a].mean() < wide[model_b].mean() else model_b})
dm_pairs = pd.DataFrame(pair_rows)
dm_pairs['p_bh_21_pairs'] = np.nan
for _, indices in dm_pairs.groupby(['series', 'horizon', 'scheme']).groups.items():
    dm_pairs.loc[indices, 'p_bh_21_pairs'] = bh(dm_pairs.loc[indices, 'p_raw'])
dm_pairs['significant_q05'] = dm_pairs['p_bh_21_pairs'] < Q
assert len(dm_pairs) == 20 * 21

# Native-coverage fold metrics remain separate from the common-origin primary comparison.
native_fold_metrics = pd.concat([read_table(model, 'fold_metrics') for model in MODEL_ORDER], ignore_index=True)
assert len(native_fold_metrics) == 760
benchmark_dm = pd.concat([read_table(model, 'benchmark_dm') for model in MODEL_ORDER], ignore_index=True)

# Supplementary native-coverage benchmark inference. These tests retain every
# native origin, so static n=1,512 and deep n=924 are deliberately not used for
# seven-way rankings. HAC autocovariances remain within each native fold block.
native_benchmark_block_rows = []
for (model, sid, horizon, scheme), group in predictions.groupby(['model', 'series', 'horizon', 'scheme']):
    group = group.sort_values('origin_index')
    for comparison, column in [('random_walk_or_persistence', 'random_walk_pred'),
                               ('training_mean', 'train_mean_pred')]:
        benchmark_loss = (group['y_true'] - group[column]) ** 2
        statistic, p_value = dm_hac_observation(group['squared_error'], benchmark_loss, horizon, group['fold'])
        native_benchmark_block_rows.append({
            'model': model, 'series': sid, 'horizon': horizon, 'scheme': scheme,
            'comparison': comparison, 'n': int(len(group)), 'dm': statistic, 'p_raw': p_value})
native_benchmark_block = pd.DataFrame(native_benchmark_block_rows)
native_benchmark_block['p_bh_14_tests'] = np.nan
for _, indices in native_benchmark_block.groupby(['series', 'horizon', 'scheme']).groups.items():
    assert len(indices) == 14  # seven models against two benchmarks
    native_benchmark_block.loc[indices, 'p_bh_14_tests'] = bh(native_benchmark_block.loc[indices, 'p_raw'])
native_benchmark_block['significant_q05'] = native_benchmark_block['p_bh_14_tests'] < Q
native_benchmark_block['outcome'] = np.select(
    [native_benchmark_block['significant_q05'] & (native_benchmark_block['dm'] < 0),
     native_benchmark_block['significant_q05'] & (native_benchmark_block['dm'] > 0)],
    ['model_win', 'model_loss'], default='not_significant')
native_benchmark_block['sample'] = np.where(native_benchmark_block['series'] == 'monash_sunspot', 'sunspot', 'financial')
benchmark_keys = ['model', 'series', 'horizon', 'scheme', 'comparison']
native_benchmark_block = native_benchmark_block.merge(
    benchmark_dm[benchmark_keys + ['dm', 'p_raw', 'p_bh_within_series_horizon_scheme']].rename(columns={
        'dm': 'dm_unsegmented', 'p_raw': 'p_raw_unsegmented',
        'p_bh_within_series_horizon_scheme': 'p_bh_exported_2_tests'}),
    on=benchmark_keys, how='left', validate='one_to_one')
native_benchmark_block['significant_exported_bh2'] = native_benchmark_block['p_bh_exported_2_tests'] < Q
native_benchmark_block['outcome_exported_bh2'] = np.select(
    [native_benchmark_block['significant_exported_bh2'] & (native_benchmark_block['dm_unsegmented'] < 0),
     native_benchmark_block['significant_exported_bh2'] & (native_benchmark_block['dm_unsegmented'] > 0)],
    ['model_win', 'model_loss'], default='not_significant')
native_benchmark_summary = (native_benchmark_block.groupby(['sample', 'model', 'comparison'])
    .agg(tests=('dm', 'size'),
         raw_lower_loss=('dm', lambda values: int((values < 0).sum())),
         significant_wins=('outcome', lambda values: int(values.eq('model_win').sum())),
         significant_losses=('outcome', lambda values: int(values.eq('model_loss').sum())),
         not_significant=('outcome', lambda values: int(values.eq('not_significant').sum())),
         exported_bh2_wins=('outcome_exported_bh2', lambda values: int(values.eq('model_win').sum())),
         exported_bh2_losses=('outcome_exported_bh2', lambda values: int(values.eq('model_loss').sum())))
    .reset_index())
assert len(native_benchmark_block) == 280
assert native_benchmark_block.groupby(['series', 'horizon', 'scheme']).size().eq(14).all()

attributions = pd.concat([read_table(model, 'attributions') for model in MODEL_ORDER], ignore_index=True)
attributions['sample'] = np.where(attributions['series'] == 'monash_sunspot', 'sunspot', 'financial')
attributions['abs_attribution'] = attributions['attribution'].abs()
probe_keys = ['model', 'series', 'horizon', 'scheme', 'fold', 'method', 'probe']
totals = attributions.groupby(probe_keys)['abs_attribution'].transform('sum')
attributions['relative_abs_attribution'] = np.where(totals > 0, attributions['abs_attribution'] / totals, np.nan)
relative_importance = (attributions.groupby(['model', 'method', 'sample', 'series', 'horizon', 'scheme', 'feature'])
                       ['relative_abs_attribution'].agg(['mean', 'median', 'std', 'count']).reset_index())
model_importance = (attributions.groupby(['model', 'method', 'sample', 'feature'])['relative_abs_attribution']
                    .agg(['mean', 'median', 'std', 'count']).reset_index())

stability = pd.concat([read_table(model, 'stability') for model in MODEL_ORDER], ignore_index=True)
stability['sample'] = np.where(stability['series'] == 'monash_sunspot', 'sunspot', 'financial')
stability_summary = (stability.groupby(['model', 'method', 'sample', 'regime'])
                     .agg(rho_median=('rho', 'median'), rho_mean=('rho', 'mean'), rho_p10=('rho', lambda x: x.quantile(.1)),
                          cosine_median=('cosine', 'median'), top1_match_rate=('top1_match', 'mean'),
                          count=('rho', 'size')).reset_index())
stability_by_scheme = (stability.groupby(['model', 'method', 'sample', 'scheme', 'regime'])
                       .agg(rho_median=('rho', 'median'), cosine_median=('cosine', 'median'),
                            top1_match_rate=('top1_match', 'mean'), count=('rho', 'size')).reset_index())
regime_wide = stability_summary.pivot_table(index=['model', 'method', 'sample'], columns='regime', values='rho_median').reset_index()
for regime in ['bear', 'bull', 'crisis']:
    if regime in regime_wide and 'boundary' in regime_wide:
        regime_wide[f'{regime}_minus_boundary'] = regime_wide[regime] - regime_wide['boundary']

faithfulness = pd.concat([read_table(model, 'faithfulness') for model in MODEL_ORDER], ignore_index=True)

def numeric_value(value):
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if str(value) == 'True':
        return 1.0
    if str(value) == 'False':
        return 0.0
    return pd.to_numeric(value, errors='coerce')

faithfulness['value_numeric'] = faithfulness['value'].map(numeric_value)
faithfulness_summary = (faithfulness.groupby(['model', 'method', 'metric'])['value_numeric']
                        .agg(['count', 'mean', 'median', 'min', 'max']).reset_index())
seed_stability = pd.concat([read_table(model, 'seed_stability') for model in ['RandomForest', 'XGBoost', 'LightGBM']], ignore_index=True)
seed_stability['sample'] = np.where(seed_stability['series'] == 'monash_sunspot', 'sunspot', 'financial')
seed_stability_summary = (seed_stability.groupby(['model', 'sample', 'regime'])
                          .agg(rho_median=('rho', 'median'), top1_match_rate=('top1_match', 'mean'), count=('rho', 'size')).reset_index())
forecast_diagnostics = pd.concat([read_table(model, 'forecast_diagnostics').assign(model=model) for model in MODEL_ORDER], ignore_index=True)
native_params = read_table('ARIMA', 'native_params')
tft_native_signals = read_table('TFT', 'native_signals')

# Model-level primary diagnostics. Financial and Sunspot scales are normalized within each cell before averaging.
model_summary = (matched_metrics.groupby('model')
                 .agg(mean_relative_rmse=('rmse_relative_to_persistence', 'mean'),
                      median_relative_rmse=('rmse_relative_to_persistence', 'median'),
                      mean_rank=('mean_rmse_rank', 'mean'), mean_da=('da', 'mean')).reset_index())
financial_summary = (matched_metrics[matched_metrics['series'] != 'monash_sunspot'].groupby('model')
                     .agg(mean_relative_rmse=('rmse_relative_to_persistence', 'mean'),
                          median_relative_rmse=('rmse_relative_to_persistence', 'median'),
                          mean_rank=('mean_rmse_rank', 'mean'), mean_da=('da', 'mean'),
                          mean_r2=('r2', 'mean')).reset_index())
sunspot_summary = matched_metrics[matched_metrics['series'] == 'monash_sunspot'].copy()

# Figures: primary accuracy, stability, faithfulness, and relative attribution evidence.
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9})
financial_metrics = matched_metrics[matched_metrics['series'] != 'monash_sunspot'].copy()
financial_metrics['cell'] = financial_metrics['series'] + ' h' + financial_metrics['horizon'].astype(str)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
for ax, scheme in zip(axes, SCHEMES):
    pivot = financial_metrics[financial_metrics['scheme'] == scheme].pivot(index='model', columns='cell', values='rmse_relative_to_persistence').reindex(MODEL_ORDER)
    image = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=min(.85, np.nanmin(pivot.values)), vmax=max(1.15, np.nanmax(pivot.values)))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title(f'{scheme.title()} walk-forward')
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f'{pivot.iloc[i, j]:.3f}', ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=ax, shrink=.75, label='RMSE / zero-return RMSE')
fig.suptitle('Matched-origin financial forecast accuracy (924 origins per cell)')
fig.savefig(OUT / 'figures' / 'matched_financial_relative_rmse.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
plot_data = financial_summary.set_index('model').reindex(MODEL_ORDER)
ax.bar(plot_data.index, plot_data['mean_da'])
ax.axhline(.5, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Mean directional accuracy across financial cells')
ax.set_title('Matched-origin directional accuracy')
ax.tick_params(axis='x', rotation=35)
fig.savefig(OUT / 'figures' / 'matched_financial_directional_accuracy.png', bbox_inches='tight')
plt.close(fig)

primary_methods = {'Linear': 'exact_linear_SHAP', 'ARIMA': 'ARIMA_native_counterfactual_lag',
                   'RandomForest': 'TreeSHAP_full_ensemble', 'XGBoost': 'TreeSHAP_full_ensemble',
                   'LightGBM': 'TreeSHAP_full_ensemble', 'LSTM': 'KernelSHAP_lag_aggregated', 'TFT': 'KernelSHAP_lag_aggregated'}
primary_stability = stability_summary[(stability_summary['sample'] == 'financial') &
                                      stability_summary.apply(lambda row: row['method'] == primary_methods[row['model']], axis=1)]
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
regimes = ['bear', 'boundary', 'bull', 'crisis']
width = .8 / len(MODEL_ORDER)
for i, model in enumerate(MODEL_ORDER):
    subset = primary_stability[primary_stability['model'] == model].set_index('regime')
    ax.bar(np.arange(len(regimes)) + i * width, [subset['rho_median'].get(regime, np.nan) for regime in regimes], width=width, label=model)
ax.set_xticks(np.arange(len(regimes)) + .4 - width / 2, regimes)
ax.set_ylabel('Median consecutive-fold Spearman correlation')
ax.set_title('Financial attribution stability by ex post regime stratum')
ax.legend(ncol=4, fontsize=8)
fig.savefig(OUT / 'figures' / 'financial_stability_by_regime.png', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
for ax, metric, title in [(axes[0], 'faithful_deletion', 'Deletion worse than random mean'),
                          (axes[1], 'faithful_insertion', 'Insertion better than random mean')]:
    subset = faithfulness[faithfulness['metric'] == metric]
    rates = subset.groupby('model')['value_numeric'].mean().reindex(MODEL_ORDER)
    ax.bar(rates.index, rates.values)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=35)
fig.suptitle('Faithfulness intervention diagnostics (ARIMA not applicable)')
fig.savefig(OUT / 'figures' / 'faithfulness_interventions.png', bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
importance_pivot = model_importance[model_importance['sample'] == 'financial'].pivot_table(index='model', columns='feature', values='mean').reindex(MODEL_ORDER)[['lag_1', 'lag_5', 'lag_21', 'lag_252']]
image = ax.imshow(importance_pivot.values, aspect='auto', cmap='viridis')
ax.set_xticks(range(4), importance_pivot.columns)
ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
for i in range(importance_pivot.shape[0]):
    for j in range(importance_pivot.shape[1]):
        ax.text(j, i, f'{importance_pivot.iloc[i, j]:.3f}', ha='center', va='center', color='white' if importance_pivot.iloc[i, j] < np.nanmax(importance_pivot.values) * .65 else 'black')
ax.set_title('Mean within-probe relative absolute attribution — financial series')
fig.colorbar(image, ax=ax, shrink=.8)
fig.savefig(OUT / 'figures' / 'financial_relative_attribution.png', bbox_inches='tight')
plt.close(fig)

# Data-derived evidence summary used by the thesis and independent claim checks.
def records(frame):
    return json.loads(frame.to_json(orient='records'))

best_rows = matched_metrics.loc[matched_metrics.groupby(['series', 'horizon', 'scheme'])['rmse'].idxmin()]
significant_pairs = dm_pairs[dm_pairs['significant_q05']]
additivity = faithfulness[faithfulness['metric'].isin(['max_abs_residual'])].groupby('model')['value_numeric'].agg(['max', 'median']).reset_index()
summary = {
    'execution_utc': datetime.now(timezone.utc).isoformat(),
    'prediction_rows_native': int(len(predictions)), 'matched_origin_count': int(len(common_origins)),
    'matched_prediction_rows': int(len(matched)), 'origins_per_series_horizon_scheme': int(counts.iloc[0]),
    'origins_per_matched_fold': int(fold_counts.iloc[0]), 'dm_pairwise_tests': int(len(dm_pairs)),
    'dm_significant_pairs_q05': int(len(significant_pairs)),
    'best_rmse_by_cell': records(best_rows[['model', 'series', 'horizon', 'scheme', 'rmse', 'rmse_relative_to_persistence']]),
    'financial_model_summary': records(financial_summary), 'sunspot_model_metrics': records(sunspot_summary),
    'stability_summary': records(stability_summary), 'regime_contrasts': records(regime_wide),
    'faithfulness_summary': records(faithfulness_summary), 'additivity_summary': records(additivity),
    'relative_importance': records(model_importance), 'seed_stability_summary': records(seed_stability_summary),
    'native_benchmark_summary': records(native_benchmark_summary),
    'identification_boundary': 'Lagged prices or native Sunspot values alone cannot identify genuine causal economic drivers; RQ3 economic-driver claims remain unsupported by the design.'
}

tables = {'matched_observation_predictions': matched, 'matched_metrics': matched_metrics,
          'matched_fold_metrics': matched_fold_metrics, 'dm_pairwise': dm_pairs,
          'native_fold_metrics': native_fold_metrics, 'native_benchmark_dm': benchmark_dm,
          'native_benchmark_block_dm': native_benchmark_block, 'native_benchmark_summary': native_benchmark_summary,
          'attributions': attributions, 'relative_importance_by_cell': relative_importance,
          'relative_importance_by_model': model_importance, 'stability': stability,
          'stability_summary': stability_summary, 'stability_by_scheme': stability_by_scheme,
          'regime_contrasts': regime_wide, 'faithfulness': faithfulness,
          'faithfulness_summary': faithfulness_summary, 'seed_stability_summary': seed_stability_summary,
          'forecast_diagnostics': forecast_diagnostics, 'arima_native_params': native_params,
          'tft_native_signals': tft_native_signals, 'model_summary': model_summary,
          'financial_model_summary': financial_summary}
for name, table in tables.items():
    table.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
(OUT / 'evidence' / 'analysis_summary.json').write_text(json.dumps(summary, indent=2, default=str))

# Independent numerical and artifact checks.
validation = {
    'all_seven_notebook_prediction_tables_present': len(prediction_parts) == 7,
    'native_prediction_row_count_188160': len(predictions) == 188160,
    'native_fold_metric_row_count_760': len(native_fold_metrics) == 760,
    'matched_origin_count_18480': len(common_origins) == 18480,
    'matched_prediction_row_count_129360': len(matched) == 129360,
    'each_cell_has_924_origins': bool((counts == 924).all()),
    'each_matched_fold_has_231_origins': bool((fold_counts == 231).all()),
    'each_origin_has_exactly_seven_models': bool(matched.groupby(origin_keys)['model'].nunique().eq(7).all()),
    'twenty_one_dm_pairs_per_cell': bool(dm_pairs.groupby(['series', 'horizon', 'scheme']).size().eq(21).all()),
    'bh_adjustment_complete_for_finite_p': bool(dm_pairs.loc[dm_pairs['p_raw'].notna(), 'p_bh_21_pairs'].notna().all()),
    'native_benchmark_block_row_count_280': len(native_benchmark_block) == 280,
    'fourteen_native_benchmark_tests_per_cell': bool(native_benchmark_block.groupby(['series', 'horizon', 'scheme']).size().eq(14).all()),
    'native_benchmark_bh14_complete': bool(native_benchmark_block['p_bh_14_tests'].notna().all()),
    'economic_identification_boundary_explicit': True
}
assert all(validation.values()), validation
(OUT / 'evidence' / 'validation.json').write_text(json.dumps(validation, indent=2))
provenance = {'source_notebooks': {model: str(BASE / MODEL_DIRS[model]) for model in MODEL_ORDER},
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'method': 'All-seven intersection by series, horizon, scheme, origin index, origin date, and target date; rolling static folds 2–5 align to deep folds 0–3; expanding folds 0–3 align directly.',
              'dm': 'Observation-level squared-error loss; HAC lag h-1 with autocovariances estimated only within matched forecast blocks; two-sided asymptotic normal p; BH within each 21-pair family at q=0.05.',
              'native_benchmark_dm': 'Supplementary native-coverage model-minus-benchmark squared-loss tests; HAC lag h-1 within native fold blocks; BH across all 14 model-benchmark tests within each series-horizon-scheme cell. Static n=1512 and deep n=924; these are not matched seven-way tests. The original unsegmented per-model two-test BH export is retained for sensitivity.'}
(OUT / 'evidence' / 'provenance.json').write_text(json.dumps(provenance, indent=2))
print(json.dumps(validation, indent=2))
print('\nFinancial model summary:\n', financial_summary.round(5).to_string(index=False))
print('\nBest RMSE models by cell:\n', best_rows[['series', 'horizon', 'scheme', 'model', 'rmse']].to_string(index=False))
print('\nSignificant DM pairs:', len(significant_pairs), 'of', len(dm_pairs))
print('\nNative benchmark outcomes (block-aware HAC; BH over 14 tests per cell):')
print(native_benchmark_block.groupby(['sample', 'comparison', 'outcome']).size().to_string())
print('\nStability summary:\n', stability_summary.round(4).to_string(index=False))
print('\nAdditivity summary:\n', additivity.to_string(index=False))
print('CROSS-MODEL SYNTHESIS COMPLETE:', OUT.resolve())
