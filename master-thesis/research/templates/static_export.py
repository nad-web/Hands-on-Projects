predictions, fold_metrics = finalize_predictions(PREDICTION_ROWS)
benchmark_dm = benchmark_tests(predictions)
faithfulness, importance, attributions, seed_stability = run_posthoc_diagnostics(EXPL)
stability, stability_summary = stability_analysis(EXPL)
figures = standard_figures(fold_metrics, importance, faithfulness, stability_summary)
extra_tables = {'seed_stability': seed_stability, 'forecast_diagnostics': pd.DataFrame(FORECAST_DIAGNOSTICS)}
corrections = [
    'Forecast origins and target dates are exported at observation level for independent recomputation.',
    'Explanations use the complete fitted predictor, including every seed constituent where applicable.',
    'Model and random deletion/insertion orderings operate on the same four lag features.'
]
if MODEL in ('RandomForest', 'XGBoost', 'LightGBM'):
    corrections.append('TreeSHAP values and expected values are averaged across both forecast seeds, making the explanation additive for the actual two-seed prediction.')
standard_export(predictions, fold_metrics, benchmark_dm, faithfulness, importance,
                attributions, stability, stability_summary, figures,
                extra_tables=extra_tables, correction_details=corrections)
print(fold_metrics.groupby(['series', 'horizon'])[['rmse', 'mae', 'da', 'r2']].mean().round(6))
print(stability_summary.round(4))
