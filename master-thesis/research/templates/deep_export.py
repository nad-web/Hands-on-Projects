predictions, fold_metrics = finalize_predictions(PREDICTION_ROWS)
benchmark_dm = benchmark_tests(predictions)
faithfulness, importance, attributions, seed_stability = run_posthoc_diagnostics(EXPL)
stability, stability_summary = stability_analysis(EXPL)
native_attribution_rows = []
if MODEL == 'TFT':
    native_stability, native_summary = stability_analysis(EXPL, value_key='native', method='native_variable_selection')
    stability = pd.concat([stability, native_stability], ignore_index=True)
    stability_summary = pd.concat([stability_summary, native_summary], ignore_index=True)
    for (sid, horizon, scheme, fold), item in EXPL.items():
        regimes_for_frame = frame_regimes(sid, FRAMES[sid])
        for probe, position in enumerate(item['pr']):
            for j, feature in enumerate(FEATURES):
                native_attribution_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                                                'model': MODEL, 'method': 'native_variable_selection', 'probe': probe,
                                                'origin_index': int(position), 'regime': regimes_for_frame.iloc[position],
                                                'feature': feature, 'attribution': float(item['native'][probe, j])})
figures = standard_figures(fold_metrics, importance, faithfulness, stability_summary)
extra_tables = {'forecast_diagnostics': pd.DataFrame(FORECAST_DIAGNOSTICS),
                'native_signals': pd.DataFrame(NATIVE_SIGNAL_ROWS),
                'native_attributions': pd.DataFrame(native_attribution_rows)}
corrections = [
    'Deep evaluation origins are explicitly te_start+lookback through te_end-1, and each input sequence is contiguous frame history.',
    'Explanation probes are contiguous 21-row histories ending immediately before each selected target origin.',
    'KernelSHAP additivity is computed as an actual residual against the fitted network prediction.',
    'KernelSHAP coordinates are aggregated by net summation over the 21 positions of each lag feature.',
    'Permutation importance shuffles complete lag channels across samples while preserving within-sequence temporal structure.',
    'Deletion/insertion and random controls intervene on the same four lag groups.',
    'The final 10% of training sequences is used for validation without an inner purge or validation-tail refit, as in the preserved design.'
]
standard_export(predictions, fold_metrics, benchmark_dm, faithfulness, importance,
                attributions, stability, stability_summary, figures,
                extra_tables=extra_tables, correction_details=corrections)
print(fold_metrics.groupby(['series', 'horizon'])[['rmse', 'mae', 'da', 'r2']].mean().round(6))
print(stability_summary.round(4))
