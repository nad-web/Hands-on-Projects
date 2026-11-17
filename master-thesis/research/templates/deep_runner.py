# Deep-learning implementation; architecture and budget are unchanged from the supplied study.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap

torch.set_num_threads(1)
torch.manual_seed(SEED)
HIDDEN, LAYERS, HEADS = 64, 2, 4
EPOCHS, BS, LR, PATIENCE = (40 if THESIS_FULL_BUDGET else 25), 64, 1e-3, 8

__NETWORK_CLASS__


def network_prediction(net, sequences):
    output = net(sequences)
    return output[0] if isinstance(output, tuple) else output


def make_training_sequences(X, y):
    if len(X) < LOOKBACK + 1:
        return np.empty((0, LOOKBACK, len(FEATURES))), np.empty(0)
    sequences = np.stack([X[i - LOOKBACK:i] for i in range(LOOKBACK, len(X))])
    return sequences, y[LOOKBACK:]


def make_evaluation_sequences(frame, positions, scaler):
    positions = np.asarray(positions, dtype=int)
    assert positions.min() >= LOOKBACK
    return np.stack([scaler.transform(frame.iloc[pos - LOOKBACK:pos][FEATURES].values) for pos in positions])


def train_network(sequences, targets):
    n_validation = max(50, int(0.1 * len(sequences)))
    assert len(sequences) - n_validation >= 100
    loader = DataLoader(TensorDataset(torch.FloatTensor(sequences[:-n_validation]),
                                      torch.FloatTensor(targets[:-n_validation])),
                        batch_size=BS, shuffle=True)
    net = DeepNet(len(FEATURES))
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best, wait, best_state = np.inf, 0, None
    validation_x = torch.FloatTensor(sequences[-n_validation:])
    validation_y = torch.FloatTensor(targets[-n_validation:])
    for epoch in range(EPOCHS):
        net.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(network_prediction(net, x_batch), y_batch)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            validation_loss = criterion(network_prediction(net, validation_x), validation_y).item()
        if validation_loss < best - 1e-6:
            best, wait = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, epoch + 1, float(best)


def fit_deep_unit(sid, horizon, scheme, fold, evaluation_positions=None):
    frame = FRAMES[sid]
    target = frame[f'target_h{horizon}'].values
    tr, te = fold['tr'], fold['te']
    X_train = frame.iloc[tr[0]:tr[1]][FEATURES].values
    y_train = target[tr[0]:tr[1]]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    train_sequences, train_targets = make_training_sequences(X_train_scaled, y_train)
    net, epochs_run, best_validation = train_network(train_sequences, train_targets)
    positions = (np.arange(te[0] + LOOKBACK, te[1]) if evaluation_positions is None
                 else np.asarray(evaluation_positions, dtype=int))
    evaluation_sequences = make_evaluation_sequences(frame, positions, scaler)
    with torch.no_grad():
        predictions = network_prediction(net, torch.FloatTensor(evaluation_sequences)).numpy()
    background_indices = np.random.RandomState(SEED).choice(len(train_sequences), min(40, len(train_sequences)), replace=False)
    detail = {'net': net, 'scaler': scaler, 'background': train_sequences[background_indices],
              'positions': positions, 'evaluation_sequences': evaluation_sequences,
              'y_evaluation': target[positions], 'epochs_run': epochs_run,
              'best_validation_loss': best_validation, 'n_train_sequences': len(train_sequences)}
    return predictions, positions, float(y_train.mean()), detail


PREDICTION_ROWS, FORECAST_DIAGNOSTICS, NATIVE_SIGNAL_ROWS = [], [], []
TRAINED_CACHE = {}
forecast_start = time.time()
for sid, frame in FRAMES.items():
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            for fold in make_folds(len(frame), scheme):
                pred, positions, train_mean, detail = fit_deep_unit(sid, horizon, scheme, fold)
                add_predictions(PREDICTION_ROWS, sid, horizon, scheme, fold['fold'], positions, pred, train_mean)
                FORECAST_DIAGNOSTICS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                             'n_train_sequences': detail['n_train_sequences'], 'epochs_run': detail['epochs_run'],
                                             'best_validation_loss': detail['best_validation_loss']})
                if fold['fold'] < EXPLAIN_FOLDS:
                    TRAINED_CACHE[(sid, horizon, scheme, fold['fold'])] = detail
                if MODEL == 'TFT':
                    with torch.no_grad():
                        _, weights, attention = detail['net'](torch.FloatTensor(detail['evaluation_sequences']))
                    recency = float(np.mean([np.average(np.arange(LOOKBACK), weights=attention[i, -1].numpy() + 1e-9)
                                             for i in range(min(50, len(attention)))]))
                    NATIVE_SIGNAL_ROWS.append({'series': sid, 'horizon': horizon, 'scheme': scheme, 'fold': fold['fold'],
                                               **{f'vs_{feature}': float(weights.mean(axis=0)[j]) for j, feature in enumerate(FEATURES)},
                                               'attention_recency': recency})
                print(f"forecast {sid} h={horizon} {scheme} fold={fold['fold']} n={len(pred)} epochs={detail['epochs_run']}")
print(f'Forecast stage completed in {time.time() - forecast_start:.1f}s')

EXPL = {}
explanation_start = time.time()
for sid, frame in FRAMES.items():
    for horizon in HORIZONS:
        for scheme in SCHEMES:
            positions = explanation_positions(frame, scheme)
            for fold in [fo for fo in make_folds(len(frame), scheme) if fo['fold'] < EXPLAIN_FOLDS]:
                key = (sid, horizon, scheme, fold['fold'])
                detail = TRAINED_CACHE[key]
                net = detail['net']
                sequences = make_evaluation_sequences(frame, positions, detail['scaler'])
                Z = sequences.reshape(len(sequences), -1)
                y = frame[f'target_h{horizon}'].values[positions]
                def predict_fn(flattened, net=net):
                    array = torch.FloatTensor(np.asarray(flattened).reshape(len(flattened), LOOKBACK, len(FEATURES)))
                    with torch.no_grad():
                        return network_prediction(net, array).numpy()
                background = detail['background'].reshape(len(detail['background']), -1)
                explainer = shap.KernelExplainer(predict_fn, background)
                phi_flat = np.asarray(explainer.shap_values(Z, nsamples=100, silent=True))
                if phi_flat.ndim == 3:
                    phi_flat = phi_flat[:, :, 0]
                phi = phi_flat.reshape(len(sequences), LOOKBACK, len(FEATURES)).sum(axis=1)
                expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
                additivity_residual = float(np.max(np.abs(phi_flat.sum(axis=1) + expected_value - predict_fn(Z))))
                native = None
                if MODEL == 'TFT':
                    with torch.no_grad():
                        _, native_weights, _ = net(torch.FloatTensor(sequences))
                    native = native_weights.numpy()
                EXPL[key] = {'pred_fn': predict_fn, 'X': Z, 'y': y, 'phi': phi, 'pr': positions,
                             'native': native, 'phi_seeds': None, 'expected_value': expected_value,
                             'additivity_residual': additivity_residual}
                print(f"explain {sid} h={horizon} {scheme} fold={fold['fold']} probes={len(positions)} additivity={additivity_residual:.3g}")
print(f'Explanation stage completed in {time.time() - explanation_start:.1f}s')
