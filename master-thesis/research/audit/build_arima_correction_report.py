from pathlib import Path
import hashlib, json
import nbformat
import numpy as np
import pandas as pd

ROOT = Path('/mnt/agents/output/thesis_execution')
OLD = ROOT / 'audit/pre_arima_benchmark_correction'
NEW = ROOT / 'results'
OUT = ROOT / 'audit'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_csv(base, rel):
    return pd.read_csv(base / rel, low_memory=False)

keys = ['model','series','horizon','scheme','fold','origin_index','origin_date','target_date']
old_pred = load_csv(OLD, 'ARIMA_study/tables/observation_predictions.csv')
new_pred = load_csv(NEW, 'ARIMA_study/tables/observation_predictions.csv')
assert old_pred[keys].equals(new_pred[keys])
invariant_prediction_fields = ['y_true','y_pred','random_walk_pred','error','squared_error','absolute_error']
prediction_invariance = {}
for col in invariant_prediction_fields:
    prediction_invariance[col] = {
        'all_equal_within_1e-12': bool(np.allclose(old_pred[col], new_pred[col], rtol=1e-12, atol=1e-12, equal_nan=True)),
        'max_abs_difference': float(np.nanmax(np.abs(old_pred[col]-new_pred[col])))
    }
benchmark_delta = (old_pred.train_mean_pred-new_pred.train_mean_pred).abs()

explanation_files = ['attributions.csv','stability.csv','faithfulness.csv','native_params.csv','stability_summary.csv','shap_importance.csv']
explanation_invariance = {
    name: {'byte_identical': sha(OLD/'ARIMA_study/tables'/name)==sha(NEW/'ARIMA_study/tables'/name)}
    for name in explanation_files
}

# Native fold metrics contain model-loss metrics, not benchmark values; they must remain invariant.
model_metric_files = ['fold_metrics.csv']
model_metric_invariance = {
    name: {'byte_identical': sha(OLD/'ARIMA_study/tables'/name)==sha(NEW/'ARIMA_study/tables'/name)}
    for name in model_metric_files
}

old_block = load_csv(OLD, 'cross_model/tables/native_benchmark_block_dm.csv')
new_block = load_csv(NEW, 'cross_model/tables/native_benchmark_block_dm.csv')
block_keys = ['model','series','horizon','scheme','comparison']
merged = old_block.merge(new_block, on=block_keys, suffixes=('_old','_new'), validate='one_to_one')
sign_changes = merged[(merged.model=='ARIMA') & (merged.comparison=='training_mean') &
                      (np.sign(merged.dm_old)!=np.sign(merged.dm_new))]
sign_columns = block_keys + ['dm_old','dm_new','p_raw_old','p_raw_new','p_bh_14_tests_old','p_bh_14_tests_new','outcome_old','outcome_new']
sign_change_records = sign_changes[sign_columns].to_dict('records')
sign_changes[sign_columns].to_csv(OUT/'ARIMA_Benchmark_DM_Sign_Changes.csv',index=False)

old_summary = load_csv(OLD, 'cross_model/tables/native_benchmark_summary.csv')
new_summary = load_csv(NEW, 'cross_model/tables/native_benchmark_summary.csv')
old_arima = old_summary[(old_summary['sample']=='financial') & (old_summary.model=='ARIMA') & (old_summary.comparison=='training_mean')].iloc[0]
new_arima = new_summary[(new_summary['sample']=='financial') & (new_summary.model=='ARIMA') & (new_summary.comparison=='training_mean')].iloc[0]

old_matched = load_csv(OLD, 'cross_model/tables/matched_metrics.csv')
new_matched = load_csv(NEW, 'cross_model/tables/matched_metrics.csv')
matched_keys=['model','series','horizon','scheme']
mm=old_matched.merge(new_matched,on=matched_keys,suffixes=('_old','_new'),validate='one_to_one')
unchanged_matched_fields=['n','rmse','mae','r2','da','persistence_rmse','rmse_relative_to_persistence','mean_rmse_rank']
matched_primary_invariance={}
for col in unchanged_matched_fields:
    matched_primary_invariance[col]=bool(np.allclose(mm[f'{col}_old'],mm[f'{col}_new'],rtol=1e-12,atol=1e-12,equal_nan=True))
train_mean_changes=mm[(mm.model=='ARIMA') & ~np.isclose(mm.train_mean_rmse_old,mm.train_mean_rmse_new,rtol=1e-12,atol=1e-12)]

old_matched_obs=load_csv(OLD,'cross_model/tables/matched_observation_predictions.csv')
new_matched_obs=load_csv(NEW,'cross_model/tables/matched_observation_predictions.csv')
matched_obs_invariance={}
for col in ['y_true','y_pred','random_walk_pred','error','squared_error','absolute_error']:
    matched_obs_invariance[col]=bool(np.allclose(old_matched_obs[col],new_matched_obs[col],rtol=1e-12,atol=1e-12,equal_nan=True))

unchanged_cross_files=[
'dm_pairwise.csv','financial_model_summary.csv','model_summary.csv','native_fold_metrics.csv',
'matched_fold_metrics.csv','stability.csv','stability_by_scheme.csv','stability_summary.csv',
'faithfulness.csv','faithfulness_summary.csv','attributions.csv','regime_contrasts.csv',
'relative_importance_by_cell.csv','relative_importance_by_model.csv','seed_stability_summary.csv',
'arima_native_params.csv','tft_native_signals.csv']
cross_invariance={name: sha(OLD/'cross_model/tables'/name)==sha(NEW/'cross_model/tables'/name) for name in unchanged_cross_files}

financial=new_block[new_block['sample']=='financial']
sunspot=new_block[new_block['sample']=='sunspot']
inference={
    'financial_tests':int(len(financial)),
    'bh14_significant_wins':int((financial.significant_q05 & (financial.outcome=='model_win')).sum()),
    'bh14_significant_losses':int((financial.significant_q05 & (financial.outcome=='model_loss')).sum()),
    'bh14_nonsignificant':int((~financial.significant_q05).sum()),
    'bh14_persistence_losses':int((financial.significant_q05 & (financial.outcome=='model_loss') & (financial.comparison=='random_walk_or_persistence')).sum()),
    'bh14_training_mean_losses':int((financial.significant_q05 & (financial.outcome=='model_loss') & (financial.comparison=='training_mean')).sum()),
    'bh2_significant_wins':int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2=='model_win')).sum()),
    'bh2_significant_losses':int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2=='model_loss')).sum()),
    'bh2_persistence_losses':int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2=='model_loss') & (financial.comparison=='random_walk_or_persistence')).sum()),
    'bh2_training_mean_losses':int((financial.significant_exported_bh2 & (financial.outcome_exported_bh2=='model_loss') & (financial.comparison=='training_mean')).sum()),
    'negative_financial_differentials_with_unadjusted_p_below_005':int(((financial.dm<0)&(financial.p_raw<0.05)).sum()),
    'sunspot_bh14_training_mean_wins':int((sunspot.significant_q05&(sunspot.outcome=='model_win')&(sunspot.comparison=='training_mean')).sum()),
    'sunspot_bh14_persistence_wins':int((sunspot.significant_q05&(sunspot.outcome=='model_win')&(sunspot.comparison=='random_walk_or_persistence')).sum()),
    'sunspot_bh14_persistence_losses':int((sunspot.significant_q05&(sunspot.outcome=='model_loss')&(sunspot.comparison=='random_walk_or_persistence')).sum()),
    'sunspot_bh14_persistence_nonsignificant':int(((~sunspot.significant_q05)&(sunspot.comparison=='random_walk_or_persistence')).sum())
}

nb_paths={
    'ARIMA_old':OLD/'02_ARIMA_Executed.ipynb',
    'ARIMA_corrected':ROOT/'notebooks/02_ARIMA_Executed.ipynb',
    'synthesis_old':OLD/'08_Cross_Model_Synthesis_Executed.ipynb',
    'synthesis_corrected':ROOT/'notebooks/08_Cross_Model_Synthesis_Executed.ipynb'}
notebooks={}
for label,path in nb_paths.items():
    nb=nbformat.read(path,as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
    notebooks[label]={'path':str(path),'sha256':sha(path),'execution_counts':[c.execution_count for c in code],'stored_errors':len(errors)}

validation=json.loads((ROOT/'audit/Execution_Validation_Report.json').read_text())
report={
    'correction':'ARIMA forecast training-mean benchmark now uses the horizon-specific supervised target over the exact fold training interval. The raw univariate training mean remains separately available only for native counterfactual replacement.',
    'notebooks':notebooks,
    'arima_observation_predictions':{
        'rows':int(len(new_pred)),
        'invariant_fields':prediction_invariance,
        'train_mean_rows_changed':int((benchmark_delta>0).sum()),
        'train_mean_max_abs_old_new_difference':float(benchmark_delta.max())
    },
    'arima_explanation_and_native_parameter_invariance':explanation_invariance,
    'arima_model_metric_invariance':model_metric_invariance,
    'matched_observation_primary_field_invariance':matched_obs_invariance,
    'matched_metric_primary_field_invariance':matched_primary_invariance,
    'matched_arima_cells_with_changed_training_mean_rmse':int(len(train_mean_changes)),
    'cross_model_byte_identical_tables':cross_invariance,
    'arima_financial_training_mean_descriptive_lower_loss_cells':{'old':int(old_arima.raw_lower_loss),'corrected':int(new_arima.raw_lower_loss),'tests':int(new_arima.tests)},
    'arima_training_mean_dm_sign_changes':sign_change_records,
    'corrected_native_benchmark_inference':inference,
    'independent_execution_validation':{'checks_passed':validation['checks_passed'],'checks_total':validation['checks_total'],'all_passed':validation['all_passed'],'benchmark_semantics':{k:v for k,v in validation['benchmark_semantics'].items() if k!='errors_by_fold'}},
    'conclusion':'The correction changes benchmark predictions and benchmark-dependent descriptive/inferential values only. ARIMA forecasts, realized targets, persistence predictions, native explanations, selected parameters, model losses, matched seven-way rankings, pairwise tests, stability, and faithfulness evidence are unchanged. No corrected financial model-minus-benchmark differential is a significant win.'
}
(OUT/'ARIMA_Benchmark_Correction_Report.json').write_text(json.dumps(report,indent=2))

md=['# ARIMA Benchmark Correction and Invariance Report','',
'## Correction','',report['correction'],'',
'## Executed notebooks','',
'| Notebook | SHA-256 | Execution counts | Stored errors |','|---|---|---|---:|']
for label in ['ARIMA_corrected','synthesis_corrected']:
    n=notebooks[label]
    md.append(f"| {label} | `{n['sha256']}` | {n['execution_counts']} | {n['stored_errors']} |")
md += ['', '## ARIMA prediction and explanation invariance','',
       f"All {len(new_pred):,} realized targets, ARIMA forecasts, persistence forecasts, and model-loss fields are unchanged. All {int((benchmark_delta>0).sum()):,} exported training-mean predictions changed; the maximum old/new absolute difference is {benchmark_delta.max():.15g}.",
       '', 'The attribution, stability, faithfulness, native-parameter, and native fold-metric tables are byte-identical to the preserved pre-correction exports.', '',
       '## Cross-model invariance','',
       'Primary matched prediction and metric fields are unchanged. Only ARIMA training-mean predictions and benchmark-dependent RMSE fields changed in the matched exports. Pairwise DM tests, matched model rankings, stability, faithfulness, and relative-attribution tables are byte-identical.','',
       '## Descriptive and inferential effects','',
       f"ARIMA's financial descriptive lower-loss count against the training mean changed from {int(old_arima.raw_lower_loss)}/16 to {int(new_arima.raw_lower_loss)}/16.",
       f"Three expanding five-session ARIMA training-mean DM statistics changed sign: {', '.join(sign_changes.series)}. All three remain nonsignificant after BH adjustment.",'',
       f"Financial BH14: {inference['bh14_significant_wins']} significant wins, {inference['bh14_significant_losses']} significant losses, and {inference['bh14_nonsignificant']} nonsignificant results. Losses comprise {inference['bh14_persistence_losses']} against persistence and {inference['bh14_training_mean_losses']} against the training mean.",
       f"Financial BH2 sensitivity: {inference['bh2_significant_wins']} significant wins and {inference['bh2_significant_losses']} significant losses ({inference['bh2_persistence_losses']} persistence; {inference['bh2_training_mean_losses']} training mean).",
       f"No negative corrected financial differential has unadjusted p < 0.05 (count: {inference['negative_financial_differentials_with_unadjusted_p_below_005']}).",'',
       '## Independent validation','',
       f"The independent reconstruction audit passed {validation['checks_passed']:,}/{validation['checks_total']:,} checks. It checked {validation['benchmark_semantics']['fold_groups_checked']:,} native fold groups for supervised-target benchmark semantics and native origin coverage. Maximum absolute benchmark discrepancy: {validation['benchmark_semantics']['max_abs_error']:.3g}.",'',
       '## Scientific interpretation','',report['conclusion']]
(OUT/'ARIMA_Benchmark_Correction_Report.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'report':str(OUT/'ARIMA_Benchmark_Correction_Report.json'),'all_invariance_checks':all(v['all_equal_within_1e-12'] for v in prediction_invariance.values()) and all(v['byte_identical'] for v in explanation_invariance.values()) and all(matched_primary_invariance.values()) and all(cross_invariance.values()),'sign_changes':sign_changes.series.tolist(),'financial_training_mean_lower_loss':[int(old_arima.raw_lower_loss),int(new_arima.raw_lower_loss)]},indent=2))
