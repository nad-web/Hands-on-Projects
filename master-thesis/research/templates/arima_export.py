predictions, fold_metrics = finalize_predictions(PREDICTION_ROWS)
benchmark_dm = benchmark_tests(predictions)
faithfulness = pd.DataFrame(FAITH_ROWS)
importance = pd.DataFrame(IMPORTANCE_ROWS)
stability, stability_summary = stability_analysis(EXPL, method='ARIMA_native_counterfactual_lag')
attribution_rows = []
for (sid, horizon, scheme, fold), item in EXPL.items():
    regimes_for_frame = frame_regimes(sid, FRAMES[sid])
    for probe, position in enumerate(item['pr']):
        for j, feature in enumerate(FEATURES):
            attribution_rows.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold,
                                     'model': MODEL, 'method': 'ARIMA_native_counterfactual_lag', 'probe': probe,
                                     'origin_index': int(position),
                                     'origin_date': FRAMES[sid]['Date'].iloc[position].strftime('%Y-%m-%d'),
                                     'target_date': target_date(sid, position, horizon).strftime('%Y-%m-%d'),
                                     'regime': regimes_for_frame.iloc[position], 'feature': feature,
                                     'attribution': float(item['phi'][probe, j])})
attributions = pd.DataFrame(attribution_rows)
figures = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, financial, title in [(axes[0], True, 'Financial RMSE'), (axes[1], False, 'Sunspot RMSE')]:
    subset = fold_metrics[(fold_metrics['series'] == 'monash_sunspot') != financial]
    ax.boxplot([subset[subset['horizon'] == h]['rmse'].values for h in HORIZONS], tick_labels=[f'h={h}' for h in HORIZONS])
    ax.set_title(title)
fig.suptitle('ARIMA: predictive error (RQ1)')
fig.tight_layout()
figures['fig_rq1'] = fig
fig, ax = plt.subplots(figsize=(7, 4))
importance.groupby('feature')['mean_abs_shap'].mean().reindex(FEATURES).plot.bar(ax=ax)
ax.set_ylabel('Mean absolute native counterfactual contribution')
ax.set_title('ARIMA: selected-lag forecast sensitivity (RQ3)')
figures['fig_native_importance'] = fig
fig, ax = plt.subplots(figsize=(8, 4))
if len(stability_summary):
    stability_summary.set_index('regime')['rho_median'].plot.bar(ax=ax)
ax.set_ylabel('Median Spearman correlation')
ax.set_title('ARIMA: native attribution stability (RQ2)')
figures['fig_stability'] = fig
extra_tables = {'native_params': pd.DataFrame(NATIVE_PARAM_ROWS),
                'forecast_diagnostics': pd.DataFrame(FORECAST_DIAGNOSTICS)}
corrections = [
    'Every lag-trimmed frame row is mapped to its original raw-series row before ARIMA fitting and state updating.',
    'At each origin, the state is updated with every newly observed raw value through that origin, including all pre-test-gap observations.',
    'Native explanations are observation-level counterfactual contributions from replacing each of four selected historical observations with the raw training mean.',
    'The forecast training-mean benchmark is computed from the horizon-specific supervised target over the exact fold training interval; it is distinct from the raw counterfactual replacement mean.',
    'A fold-by-fold benchmark-semantics audit independently checks every exported training-mean prediction against that supervised target mean.',
    'A joint four-lag replacement checks exact linear additivity; PFI, SHAP, LIME, and deletion/insertion are not imputed for the univariate ARIMA model.'
]
standard_export(predictions, fold_metrics, benchmark_dm, faithfulness, importance,
                attributions, stability, stability_summary, figures,
                extra_tables=extra_tables, correction_details=corrections)
print(fold_metrics.groupby(['series', 'horizon'])[['rmse', 'mae', 'da', 'r2']].mean().round(6))
print(stability_summary.round(4))
