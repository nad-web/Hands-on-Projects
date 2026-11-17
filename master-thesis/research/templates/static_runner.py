# Model-specific implementation; hyperparameters are unchanged from the supplied study.
__MODEL_IMPORTS__


def fit_static_unit(sid, horizon, scheme, fold, evaluation_positions=None, keep_details=False):
    frame = FRAMES[sid]
    target = frame[f'target_h{horizon}'].values
    tr, te = fold['tr'], fold['te']
    X_train = frame.iloc[tr[0]:tr[1]][FEATURES].values
    y_train = target[tr[0]:tr[1]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    positions = (np.arange(te[0], te[1]) if evaluation_positions is None
                 else np.asarray(evaluation_positions, dtype=int))
    X_evaluation = scaler.transform(frame.iloc[positions][FEATURES].values)
    y_evaluation = target[positions]
    __FIT_MODELS__
    predictions = predict_fn(X_evaluation)
    details = None
    if keep_details:
        details = {'models': models, 'scaler': scaler, 'predict_fn': predict_fn,
                   'X_train_scaled': X_train_scaled, 'X_evaluation': X_evaluation,
                   'y_evaluation': y_evaluation, 'positions': positions}
    return predictions, positions, float(y_train.mean()), details


PREDICTION_ROWS = []
FORECAST_DIAGNOSTICS = []
EXPL = {}
forecast_start = time.time()
for sid, frame in FRAMES.items():
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            for fold in make_folds(len(frame), scheme):
                pred, positions, train_mean, detail = fit_static_unit(sid, horizon, scheme, fold, keep_details=True)
                add_predictions(PREDICTION_ROWS, sid, horizon, scheme, fold['fold'], positions, pred, train_mean)
                constituent_predictions = np.asarray([m.predict(detail['X_evaluation']) for m in detail['models']])
                best_iterations = [getattr(m, 'best_iteration_', getattr(m, 'best_iteration', np.nan)) for m in detail['models']]
                FORECAST_DIAGNOSTICS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                             'n_train': len(detail['X_train_scaled']),
                                             'seed_prediction_sigma': float(constituent_predictions.std(axis=0).mean()),
                                             'best_iteration_mean': float(np.nanmean(np.asarray(best_iterations, dtype=float)))})
                print(f"forecast {sid} h={horizon} {scheme} fold={fold['fold']} n={len(pred)}")
print(f'Forecast stage completed in {time.time() - forecast_start:.1f}s')

explanation_start = time.time()
for sid, frame in FRAMES.items():
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            positions = explanation_positions(frame, scheme)
            for fold in [fo for fo in make_folds(len(frame), scheme) if fo['fold'] < EXPLAIN_FOLDS]:
                _, _, _, detail = fit_static_unit(sid, horizon, scheme, fold, positions, keep_details=True)
                X = detail['X_evaluation']
                models = detail['models']
                __EXPLAIN_ATTRIBUTIONS__
                expected_value = float(np.asarray(expected_value).reshape(-1)[0])
                additivity_residual = float(np.max(np.abs(phi.sum(axis=1) + expected_value - detail['predict_fn'](X))))
                EXPL[(sid, horizon, scheme, fold['fold'])] = {
                    'pred_fn': detail['predict_fn'], 'X': X, 'y': detail['y_evaluation'],
                    'phi': phi, 'pr': positions, 'native': None, 'phi_seeds': phi_seeds,
                    'expected_value': expected_value, 'additivity_residual': additivity_residual
                }
                print(f"explain {sid} h={horizon} {scheme} fold={fold['fold']} probes={len(positions)} additivity={additivity_residual:.3g}")
print(f'Explanation stage completed in {time.time() - explanation_start:.1f}s')
