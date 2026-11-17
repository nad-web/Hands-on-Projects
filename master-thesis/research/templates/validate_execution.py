from pathlib import Path
import hashlib, json
import nbformat
import numpy as np
import pandas as pd

ROOT = Path('/mnt/agents/output/thesis_execution')
R = ROOT / 'results'
A = ROOT / 'audit'
A.mkdir(exist_ok=True)
MODELS = {
    'Linear': ('01_Linear_Regression_Executed.ipynb', 'Linear_study'),
    'ARIMA': ('02_ARIMA_Executed.ipynb', 'ARIMA_study'),
    'RandomForest': ('03_Random_Forest_Executed.ipynb', 'RandomForest_study'),
    'XGBoost': ('04_XGBoost_Executed.ipynb', 'XGBoost_study'),
    'LightGBM': ('05_LightGBM_Executed.ipynb', 'LightGBM_study'),
    'LSTM': ('06_LSTM_Executed.ipynb', 'LSTM_study'),
    'TFT': ('07_Temporal_Fusion_Transformer_Executed.ipynb', 'TFT_study')
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


checks = {}
model_reports = {}
prediction_parts = []
fold_parts = []
attribution_parts = []
for model, (filename, directory) in MODELS.items():
    notebook_path = ROOT / 'notebooks' / filename
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
    errors = [output for cell in code_cells for output in cell.get('outputs', []) if output.output_type == 'error']
    execution_counts = [cell.execution_count for cell in code_cells]
    out = R / directory
    audit = json.loads((out / 'evidence' / 'completion_audit.json').read_text())
    provenance = json.loads((out / 'evidence' / 'provenance.json').read_text())
    expected_folds = 80 if model in ('LSTM', 'TFT') else 120
    expected_explanations = 60 if model in ('LSTM', 'TFT') else 80
    expected_predictions = 18480 if model in ('LSTM', 'TFT') else 30240
    hashes = json.loads((out / 'FILE_SHA256.json').read_text())
    actual_hashes = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*'))
                     if path.is_file() and path.name != 'FILE_SHA256.json'}
    predictions = pd.read_csv(out / 'tables' / 'observation_predictions.csv', low_memory=False)
    folds = pd.read_csv(out / 'tables' / 'fold_metrics.csv')
    attributions = pd.read_csv(out / 'tables' / 'attributions.csv')
    prediction_parts.append(predictions)
    fold_parts.append(folds)
    attribution_parts.append(attributions)
    model_checks = {
        'notebook_schema_valid': True,
        'five_code_cells_executed_sequentially': execution_counts == [1, 2, 3, 4, 5],
        'no_notebook_errors': len(errors) == 0,
        'full_budget_flag_false': 'THESIS_FULL_BUDGET = False' in '\n'.join(cell.source for cell in code_cells),
        'fold_units_complete': audit['completed_fold_units'] == expected_folds and audit['missing_fold_units'] == 0,
        'explanation_units_complete': audit['completed_explained_units'] == expected_explanations,
        'prediction_count_correct': len(predictions) == expected_predictions == audit['observation_prediction_rows'],
        'predictions_finite': bool(np.isfinite(predictions[['y_true', 'y_pred']]).all().all()),
        'no_duplicate_model_fold_origins': not predictions.duplicated(['series', 'horizon', 'scheme', 'fold', 'origin_index']).any(),
        'fold_metric_count_correct': len(folds) == expected_folds,
        'all_internal_validations_true': all(audit['validation_checks'].values()),
        'artifact_hashes_match': actual_hashes == hashes,
        'raw_sunspot_verified': provenance['sunspot_source_verification']['value_mismatches'] == 0,
        'correction_record_present': len(provenance['documented_corrections']) >= 8
    }
    model_reports[model] = {'checks': model_checks, 'notebook_sha256': sha(notebook_path),
                            'wall_seconds': audit['wall_seconds'], 'prediction_rows': len(predictions),
                            'attribution_rows': len(attributions)}
    checks.update({f'{model}:{key}': value for key, value in model_checks.items()})

predictions = pd.concat(prediction_parts, ignore_index=True)
folds = pd.concat(fold_parts, ignore_index=True)
attributions = pd.concat(attribution_parts, ignore_index=True)
checks['total_native_predictions_188160'] = len(predictions) == 188160
checks['total_native_fold_metrics_760'] = len(folds) == 760
checks['total_attribution_rows_78400'] = len(attributions) == 78400
checks['all_fold_metric_status_complete'] = folds['status'].eq('complete').all()

# Independent raw-series reconstruction of every exported target and target date.
reconstruction_errors = {}
audit_frames = {}
for sid, group in predictions.groupby('series'):
    if sid == 'monash_sunspot':
        raw = pd.read_csv(R / 'data' / 'monash_sunspot_no_missing_T1_85babb464a3e.csv', parse_dates=['date']).rename(columns={'date': 'Date'}).set_index('Date')
    else:
        filename = {'sp500': 'sp500_observations.csv', 'dax': 'dax_observations.csv',
                    'nasdaq': 'nasdaq_observations.csv', 'djia': 'djia_observations.csv'}[sid]
        raw = pd.read_csv(R / 'data' / filename, parse_dates=['Date'])
        raw['value'] = np.log(raw['Adj Close'] / raw['Adj Close'].shift(1))
        raw = raw.dropna().set_index('Date')
    # Rebuild the supervised frame independently for benchmark-semantics checks.
    modeling = raw.copy().reset_index()
    modeling['source_row'] = np.arange(len(modeling), dtype=int)
    value_column = 'value'
    for lag in [1, 5, 21, 252]:
        modeling[f'lag_{lag}'] = modeling[value_column].shift(lag)
    for horizon in [1, 5]:
        if sid == 'monash_sunspot':
            modeling[f'target_h{horizon}'] = modeling[value_column].shift(-horizon)
        else:
            modeling[f'target_h{horizon}'] = sum(modeling[value_column].shift(-i) for i in range(1, horizon + 1))
    audit_frames[sid] = modeling.iloc[252:-5].dropna().reset_index(drop=True)
    origin_dates = pd.to_datetime(group['origin_date'])
    target_dates = pd.to_datetime(group['target_date'])
    position = pd.Series(np.arange(len(raw)), index=raw.index)
    origins = position.loc[origin_dates].values
    targets = position.loc[target_dates].values
    checks[f'{sid}:target_date_matches_origin_plus_horizon'] = bool(np.all(targets == origins + group['horizon'].values))
    if sid == 'monash_sunspot':
        expected = raw['value'].values[targets]
    else:
        expected = np.asarray([raw['value'].values[i + 1:i + h + 1].sum() for i, h in zip(origins, group['horizon'])])
    error = float(np.max(np.abs(expected - group['y_true'].values)))
    reconstruction_errors[sid] = error
    checks[f'{sid}:realized_targets_reconstruct_from_raw'] = error < 1e-10

# Explicit benchmark-semantics and native-origin checks for every model/series/fold.
# In particular, ARIMA's raw counterfactual replacement mean is not accepted as
# the supervised-target forecast benchmark.
benchmark_semantics_errors = {}
for (model, sid, horizon, scheme, fold_id), group in predictions.groupby(
        ['model', 'series', 'horizon', 'scheme', 'fold'], sort=True):
    frame = audit_frames[sid]
    n = len(frame)
    n_folds = 4 if model in ('LSTM', 'TFT') else 6
    folds = []
    for candidate in range(n_folds):
        test_start = (n - n_folds * 252 + candidate * 252 if scheme == 'rolling'
                      else 500 + 5 + 5 + candidate * 252)
        test_end = min(test_start + 252, n)
        train_end = test_start - 5 - 5
        train_start = max(0, train_end - 500) if scheme == 'rolling' else 0
        folds.append({'fold': candidate, 'tr': (train_start, train_end), 'te': (test_start, test_end)})
    fold_definition = next(item for item in folds if item['fold'] == fold_id)
    tr, te = fold_definition['tr'], fold_definition['te']
    expected_mean = float(frame[f'target_h{horizon}'].iloc[tr[0]:tr[1]].mean())
    exported = group['train_mean_pred'].dropna().unique()
    error = abs(float(exported[0]) - expected_mean) if len(exported) == 1 else float('inf')
    benchmark_semantics_errors[f'{model}|{sid}|h{horizon}|{scheme}|fold{fold_id}'] = error
    checks[f'{model}:{sid}:h{horizon}:{scheme}:fold{fold_id}:training_mean_matches_supervised_target'] = bool(
        len(exported) == 1 and np.isclose(float(exported[0]), expected_mean, rtol=1e-12, atol=1e-12))
    lookback = 21 if model in ('LSTM', 'TFT') else 0
    expected_origins = np.arange(te[0] + lookback, te[1])
    checks[f'{model}:{sid}:h{horizon}:{scheme}:fold{fold_id}:native_origins_match_design'] = bool(
        np.array_equal(np.sort(group['origin_index'].unique()), expected_origins))

# Synthesis artifact checks, if the synthesis has completed.
synthesis_validation_path = R / 'cross_model' / 'evidence' / 'validation.json'
if synthesis_validation_path.exists():
    synthesis_validation = json.loads(synthesis_validation_path.read_text())
    checks.update({f'synthesis:{key}': value for key, value in synthesis_validation.items()})
    synthesis_notebook = nbformat.read(ROOT / 'notebooks' / '08_Cross_Model_Synthesis_Executed.ipynb', as_version=4)
    synthesis_errors = [output for cell in synthesis_notebook.cells if cell.cell_type == 'code'
                        for output in cell.get('outputs', []) if output.output_type == 'error']
    checks['synthesis:no_notebook_errors'] = len(synthesis_errors) == 0
    checks['synthesis:sequential_execution'] = [cell.execution_count for cell in synthesis_notebook.cells if cell.cell_type == 'code'] == [1, 2]

passed = sum(bool(value) for value in checks.values())
report = {'checks_passed': passed, 'checks_total': len(checks), 'all_passed': all(checks.values()),
          'checks': checks, 'target_reconstruction_max_abs_errors': reconstruction_errors,
          'benchmark_semantics': {
              'definition': 'For every model/series/horizon/scheme/fold, the exported train_mean_pred must equal the horizon-specific supervised target mean over the exact training interval.',
              'tolerance': {'rtol': 1e-12, 'atol': 1e-12},
              'fold_groups_checked': len(benchmark_semantics_errors),
              'max_abs_error': max(benchmark_semantics_errors.values()),
              'max_abs_error_by_model_series': {
                  f'{model}|{sid}': max(error for key, error in benchmark_semantics_errors.items()
                                       if key.startswith(f'{model}|{sid}|'))
                  for model in MODELS for sid in audit_frames},
              'errors_by_fold': benchmark_semantics_errors},
          'models': model_reports}
(A / 'Execution_Validation_Report.json').write_text(json.dumps(report, indent=2, default=str))
md = ['# Execution Validation Report', '', f"**Result:** {passed}/{len(checks)} checks passed.", '',
      f"Native predictions: {len(predictions):,}; native fold metrics: {len(folds):,}; attribution rows: {len(attributions):,}.", '',
      '| Model | Predictions | Attributions | Runtime (s) | Checks |', '|---|---:|---:|---:|---|']
for model, model_report in model_reports.items():
    model_checks = model_report['checks']
    md.append(f"| {model} | {model_report['prediction_rows']:,} | {model_report['attribution_rows']:,} | {model_report['wall_seconds']:.2f} | {sum(model_checks.values())}/{len(model_checks)} |")
md += ['', '## Raw-target reconstruction', ''] + [f'- {sid}: maximum absolute error {error:.3g}' for sid, error in reconstruction_errors.items()]
md += ['', '## Benchmark semantics', '',
       f"Supervised-target training-mean fold groups checked: {len(benchmark_semantics_errors):,}.",
       f"Maximum absolute discrepancy: {max(benchmark_semantics_errors.values()):.3g}.",
       'Native test-origin coverage is also reconstructed independently for every fold.']
(A / 'Execution_Validation_Report.md').write_text('\n'.join(md) + '\n')
print(json.dumps({'checks_passed': passed, 'checks_total': len(checks), 'all_passed': all(checks.values()),
                  'failed': [key for key, value in checks.items() if not value]}, indent=2))
if not all(checks.values()):
    raise SystemExit(1)
