# ARIMA implementation with raw-row alignment and observation-level native counterfactuals.
from statsmodels.tsa.arima.model import ARIMA

ORDERS = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (3, 0, 0), (1, 1, 0), (2, 1, 0)]


def forecast_from_history(history, order, params, horizon, financial):
    result = ARIMA(np.asarray(history, dtype=float), order=order).filter(params)
    forecast = np.asarray(result.forecast(horizon), dtype=float)
    return float(forecast.sum() if financial else forecast[horizon - 1])


PREDICTION_ROWS, NATIVE_PARAM_ROWS, FORECAST_DIAGNOSTICS = [], [], []
ARIMA_MODELS = {}
FIT_CACHE = {}
forecast_start = time.time()
for sid, frame in FRAMES.items():
    financial = sid != 'monash_sunspot'
    raw = (DATA[sid]['log_return'].values.astype(float) if financial else DATA[sid]['value'].values.astype(float))
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            for fold in make_folds(len(frame), scheme):
                tr, te = fold['tr'], fold['te']
                raw_train_start = int(frame['source_row'].iloc[tr[0]])
                raw_train_end = int(frame['source_row'].iloc[tr[1] - 1]) + 1
                train = raw[raw_train_start:raw_train_end]
                cache_key = (sid, scheme, fold['fold'], raw_train_start, raw_train_end)
                if cache_key not in FIT_CACHE:
                    best_result, best_order, best_aic = None, None, np.inf
                    for order in ORDERS:
                        try:
                            fitted = ARIMA(train, order=order).fit()
                            if np.isfinite(fitted.aic) and fitted.aic < best_aic:
                                best_result, best_order, best_aic = fitted, order, float(fitted.aic)
                        except Exception:
                            continue
                    if best_result is None:
                        raise RuntimeError(f'No ARIMA candidate converged for {sid} {scheme} fold={fold["fold"]}')
                    FIT_CACHE[cache_key] = (best_order, np.asarray(best_result.params, dtype=float).copy(), best_aic,
                                            np.asarray(getattr(best_result, 'arparams', np.asarray([]))),
                                            np.asarray(getattr(best_result, 'maparams', np.asarray([]))),
                                            float(getattr(best_result, 'sigma2', np.nan)))
                best_order, params, best_aic, ar_params, ma_params, sigma2 = FIT_CACHE[cache_key]
                state = ARIMA(train, order=best_order).filter(params)
                next_raw = raw_train_end
                predictions = []
                positions = np.arange(te[0], te[1])
                for position in positions:
                    origin_raw = int(frame['source_row'].iloc[position])
                    while next_raw <= origin_raw:
                        state = state.append(np.asarray([raw[next_raw]]), refit=False)
                        next_raw += 1
                    forecast = np.asarray(state.forecast(horizon), dtype=float)
                    predictions.append(float(forecast.sum() if financial else forecast[horizon - 1]))
                predictions = np.asarray(predictions)
                # The forecast benchmark is the mean of the horizon-specific
                # supervised training target. This is deliberately distinct
                # from the raw training mean used for native counterfactuals.
                target_mean = float(frame[f'target_h{horizon}'].iloc[tr[0]:tr[1]].mean())
                assert np.isfinite(target_mean)
                add_predictions(PREDICTION_ROWS, sid, horizon, scheme, fold['fold'], positions, predictions, target_mean)
                ARIMA_MODELS[(sid, horizon, scheme, fold['fold'])] = {
                    'order': best_order, 'params': params, 'aic': best_aic,
                    'raw_train_start': raw_train_start, 'raw_train_end': raw_train_end,
                    'train_mean': float(train.mean()),
                    'benchmark_target_mean': target_mean
                }
                NATIVE_PARAM_ROWS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                          'order': str(best_order), 'aic': best_aic, 'n_train_raw': len(train),
                                          **{f'ar_{i + 1}': float(value) for i, value in enumerate(ar_params)},
                                          **{f'ma_{i + 1}': float(value) for i, value in enumerate(ma_params)},
                                          'sigma2': sigma2})
                FORECAST_DIAGNOSTICS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                             'n_train_raw': len(train), 'raw_train_start': raw_train_start,
                                             'raw_train_end': raw_train_end, 'selected_order': str(best_order),
                                             'supervised_target_train_mean': target_mean,
                                             'raw_train_mean_for_native_counterfactual': float(train.mean())})
                print(f"forecast {sid} h={horizon} {scheme} fold={fold['fold']} order={best_order} n={len(predictions)}")
print(f'Forecast stage completed in {time.time() - forecast_start:.1f}s')

EXPL, FAITH_ROWS, IMPORTANCE_ROWS = {}, [], []
explanation_start = time.time()
for sid, frame in FRAMES.items():
    financial = sid != 'monash_sunspot'
    raw = (DATA[sid]['log_return'].values.astype(float) if financial else DATA[sid]['value'].values.astype(float))
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            positions = explanation_positions(frame, scheme)
            for fold in [fo for fo in make_folds(len(frame), scheme) if fo['fold'] < EXPLAIN_FOLDS]:
                key = (sid, horizon, scheme, fold['fold'])
                info = ARIMA_MODELS[key]
                phi_rows = []
                for position in positions:
                    origin_raw = int(frame['source_row'].iloc[position])
                    history = raw[info['raw_train_start']:origin_raw + 1].copy()
                    assert len(history) > max(LAGS)
                    base = forecast_from_history(history, info['order'], info['params'], horizon, financial)
                    contributions = []
                    for lag in LAGS:
                        changed = history.copy()
                        changed[-lag] = info['train_mean']
                        counterfactual = forecast_from_history(changed, info['order'], info['params'], horizon, financial)
                        contributions.append(base - counterfactual)
                    changed_all = history.copy()
                    for lag in LAGS:
                        changed_all[-lag] = info['train_mean']
                    counterfactual_all = forecast_from_history(changed_all, info['order'], info['params'], horizon, financial)
                    additivity_residual = abs((base - counterfactual_all) - sum(contributions))
                    phi_rows.append(contributions + [additivity_residual])
                phi_with_residual = np.asarray(phi_rows)
                phi = phi_with_residual[:, :len(FEATURES)]
                max_residual = float(phi_with_residual[:, -1].max())
                EXPL[key] = {'phi': phi, 'pr': positions, 'native': None, 'phi_seeds': None}
                FAITH_ROWS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                   'model': MODEL, 'method': 'native_selected_lag_counterfactual_additivity',
                                   'metric': 'max_abs_residual', 'value': max_residual})
                for j, feature in enumerate(FEATURES):
                    IMPORTANCE_ROWS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                            'model': MODEL, 'feature': feature,
                                            'mean_abs_shap': float(np.abs(phi[:, j]).mean())})
                print(f"explain {sid} h={horizon} {scheme} fold={fold['fold']} probes={len(positions)} additivity={max_residual:.3g}")
print(f'Explanation stage completed in {time.time() - explanation_start:.1f}s')
