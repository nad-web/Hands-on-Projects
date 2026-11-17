from pathlib import Path
import textwrap
import sys
import nbformat

ROOT = Path('/mnt/agents/output/thesis_execution')
T = ROOT / 'templates'
N = ROOT / 'notebooks'
N.mkdir(parents=True, exist_ok=True)
core = (T / 'core.py').read_text()
static_runner = (T / 'static_runner.py').read_text()
static_export = (T / 'static_export.py').read_text()
deep_runner = (T / 'deep_runner.py').read_text()
deep_export = (T / 'deep_export.py').read_text()
arima_runner = (T / 'arima_runner.py').read_text()
arima_export = (T / 'arima_export.py').read_text()

models = {
    'Linear': {
        'filename': '01_Linear_Regression_Executed.ipynb', 'title': 'Linear Regression',
        'imports': 'from sklearn.linear_model import LinearRegression',
        'fit': """models = [LinearRegression().fit(X_train_scaled, y_train)]
def predict_fn(Z):
    return models[0].predict(Z)""",
        'explain': """coefficients = models[0].coef_
phi = X * coefficients
expected_value = models[0].intercept_ + np.dot(coefficients, detail['X_train_scaled'].mean(axis=0))
phi_seeds = None""",
        'configuration': 'Ordinary least squares on standardized lag_1, lag_5, lag_21, and lag_252. Exact linear attributions are computed on the same standardized scale as the fitted predictor.'
    },
    'RandomForest': {
        'filename': '03_Random_Forest_Executed.ipynb', 'title': 'Random Forest',
        'imports': 'from sklearn.ensemble import RandomForestRegressor\nimport shap',
        'fit': """models = [RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=20,
                                  random_state=seed, n_jobs=1).fit(X_train_scaled, y_train)
          for seed in SEEDS]
def predict_fn(Z):
    return np.mean([model.predict(Z) for model in models], axis=0)""",
        'explain': """explainers = [shap.TreeExplainer(model) for model in models]
phi_seeds = [np.asarray(explainer.shap_values(X)) for explainer in explainers]
phi = np.mean(phi_seeds, axis=0)
expected_value = np.mean([np.asarray(explainer.expected_value).reshape(-1)[0] for explainer in explainers])""",
        'configuration': 'Two-seed forecast average (20261017 and 20261018), 100 trees per seed, maximum depth 10, and minimum leaf size 20. TreeSHAP values and expected values are averaged over both constituents.'
    },
    'XGBoost': {
        'filename': '04_XGBoost_Executed.ipynb', 'title': 'XGBoost',
        'imports': 'import xgboost as xgb\nimport shap',
        'fit': """n_validation = max(50, int(0.1 * len(X_train_scaled)))
models = []
for seed in SEEDS:
    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                             reg_lambda=1.0, early_stopping_rounds=25,
                             random_state=seed, n_jobs=1)
    model.fit(X_train_scaled[:-n_validation], y_train[:-n_validation],
              eval_set=[(X_train_scaled[-n_validation:], y_train[-n_validation:])], verbose=False)
    models.append(model)
def predict_fn(Z):
    return np.mean([model.predict(Z) for model in models], axis=0)""",
        'explain': """explainers = [shap.TreeExplainer(model) for model in models]
phi_seeds = [np.asarray(explainer.shap_values(X)) for explainer in explainers]
phi = np.mean(phi_seeds, axis=0)
expected_value = np.mean([np.asarray(explainer.expected_value).reshape(-1)[0] for explainer in explainers])""",
        'configuration': 'Two-seed average, at most 200 boosting rounds, depth 6, learning rate 0.05, row and column sampling 0.8, L1 0.1, L2 1.0, and 25-round early stopping on the final 10% of training rows (minimum 50).'
    },
    'LightGBM': {
        'filename': '05_LightGBM_Executed.ipynb', 'title': 'LightGBM',
        'imports': 'import lightgbm as lgb\nimport shap',
        'fit': """n_validation = max(50, int(0.1 * len(X_train_scaled)))
models = []
for seed in SEEDS:
    model = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                              reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=-1)
    model.fit(X_train_scaled[:-n_validation], y_train[:-n_validation],
              eval_set=[(X_train_scaled[-n_validation:], y_train[-n_validation:])],
              callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(0)])
    models.append(model)
def predict_fn(Z):
    return np.mean([model.predict(Z) for model in models], axis=0)""",
        'explain': """explainers = [shap.TreeExplainer(model) for model in models]
phi_seeds = [np.asarray(explainer.shap_values(X)) for explainer in explainers]
phi = np.mean(phi_seeds, axis=0)
expected_value = np.mean([np.asarray(explainer.expected_value).reshape(-1)[0] for explainer in explainers])""",
        'configuration': 'Two-seed average, at most 200 boosting rounds, depth 6, learning rate 0.05, row and column sampling 0.8, L1 0.1, L2 1.0, and 25-round early stopping on the final 10% of training rows (minimum 50).'
    }
}

lstm_class = """class DeepNet(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.lstm = nn.LSTM(din, HIDDEN, LAYERS, batch_first=True, dropout=0.1 if LAYERS > 1 else 0.0)
        self.fc = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1]).squeeze(-1)"""

tft_class = """class DeepNet(nn.Module):
    def __init__(self, din=4):
        super().__init__()
        self.vs = nn.Sequential(nn.Linear(din, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, din), nn.Softmax(dim=-1))
        self.enc = nn.LSTM(din, HIDDEN, batch_first=True)
        self.attn = nn.MultiheadAttention(HIDDEN, HEADS, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        weights = self.vs(x.mean(1))
        selected = x * weights.unsqueeze(1)
        encoded, _ = self.enc(selected)
        attended, attention_weights = self.attn(encoded, encoded, encoded)
        return self.out(attended[:, -1]).squeeze(-1), weights, attention_weights"""

for model, specification in models.items():
    runner = static_runner.replace('__MODEL_IMPORTS__', specification['imports'])
    runner = runner.replace('    __FIT_MODELS__', textwrap.indent(specification['fit'], '    '))
    runner = runner.replace('                __EXPLAIN_ATTRIBUTIONS__', textwrap.indent(specification['explain'], '                '))
    specification['runner'] = runner
    specification['export'] = static_export

models['ARIMA'] = {'filename': '02_ARIMA_Executed.ipynb', 'title': 'ARIMA', 'runner': arima_runner, 'export': arima_export,
                   'configuration': 'AIC selection among (1,0,0), (2,0,0), (1,0,1), (2,0,1), (3,0,0), (1,1,0), and (2,1,0). The state is updated with every raw observation available at each forecast origin.'}
models['LSTM'] = {'filename': '06_LSTM_Executed.ipynb', 'title': 'Long Short-Term Memory',
                  'runner': deep_runner.replace('__NETWORK_CLASS__', lstm_class), 'export': deep_export,
                  'configuration': 'Lookback 21, hidden size 64, two layers, dropout 0.1, at most 25 epochs, batch size 64, Adam learning rate 0.001, and patience 8. Validation uses the final 10% of training sequences (minimum 50) without an inner purge or validation-tail refit.'}
models['TFT'] = {'filename': '07_Temporal_Fusion_Transformer_Executed.ipynb', 'title': 'Simplified Temporal Fusion Transformer',
                 'runner': deep_runner.replace('__NETWORK_CLASS__', tft_class), 'export': deep_export,
                 'configuration': 'Simplified variable-selection softmax, one-layer LSTM encoder, four-head attention, and linear output; hidden size 64, lookback 21, at most 25 epochs, batch size 64, Adam 0.001, and patience 8. This is not the full PyTorch Forecasting TFT.'}

order = ['Linear', 'ARIMA', 'RandomForest', 'XGBoost', 'LightGBM', 'LSTM', 'TFT']
requested = set(sys.argv[1:])
for model in order:
    if requested and model not in requested:
        continue
    specification = models[model]
    model_core = core.replace('__MODEL__', model)
    introduction = f"""# {specification['title']} — corrected, fully executed thesis notebook

This notebook is a standalone, executable component of the thesis **“Towards Trustworthy Financial Time Series Forecasting: An Experimental Evaluation of Explainable Machine Learning Models for Stock Price Prediction.”** It preserves the supplied model family, datasets, indices, targets, feature lags, walk-forward design, random seeds, and training budget.

The run covers S&P 500, DAX, NASDAQ Composite, DJIA, and the Monash Sunspot T1 benchmark at one- and five-step horizons under rolling and expanding evaluation. No network fallback or untraceable normalized Sunspot input is used."""
    corrections = """## Execution and evaluation contract

- Export every observation-level forecast with origin and target dates, realized values, predictions, and benchmark predictions.
- Define each fold's training-mean forecast benchmark from the horizon-specific supervised training target, and validate it independently. For ARIMA, keep this benchmark separate from the raw training mean used in native counterfactual explanations.
- Retain native fold coverage and enable a later all-seven common-origin comparison; do not compare unequal test spans as if they were matched.
- Use observation-level squared errors and HAC lag h−1 for Diebold–Mariano benchmark tests.
- Use identical out-of-sample probe origins across consecutive explained folds and export stability, cosine, top-feature agreement, and regime strata.
- Keep statistical reliance distinct from causal economic identification. Lagged prices or native Sunspot values cannot establish genuine causal economic drivers.
- Record implementation corrections, package versions, input hashes, runtime, completion, and validation checks.

**Configuration preserved:** """ + specification['configuration']
    data_cell = """from IPython.display import display
print('Execution model:', MODEL)
print('Execution version:', EXECUTION_VERSION)
print('Sunspot verification:', SUNSPOT_AUDIT)
data_summary = pd.DataFrame([{'series': sid, 'rows_after_first_return_drop': len(DATA[sid]),
                              'modeling_frame_rows': len(FRAMES[sid]),
                              'start': DATA[sid]['Date'].min(), 'end': DATA[sid]['Date'].max()}
                             for sid in DATA])
display(data_summary)
fold_summary = pd.DataFrame([{'series': sid, 'scheme': scheme, 'fold': fold['fold'],
                              'train_start': fold['tr'][0], 'train_end': fold['tr'][1],
                              'test_start': fold['te'][0], 'test_end': fold['te'][1],
                              'test_n': fold['te'][1] - fold['te'][0]}
                             for sid, frame in FRAMES.items() for scheme in SCHEMES
                             for fold in make_folds(len(frame), scheme)])
display(fold_summary.head(12))
assert len(fold_summary) == len(FRAMES) * len(SCHEMES) * N_FOLDS"""
    diagnostic_md = """## Diagnostics, evidence, and export

The following cell computes native fold metrics from observation-level predictions, benchmark HAC-DM tests, explanation stability and faithfulness diagnostics, figures, provenance, and completion audits. The exported `observation_predictions.csv` is the authoritative prediction artifact for subsequent matched seven-model synthesis."""
    readback = """# Independent read-back of exported artifacts and checksum verification.
from IPython.display import Image, display
saved_audit = json.loads((OUT / 'evidence' / 'completion_audit.json').read_text())
assert saved_audit['missing_fold_units'] == 0
assert all(saved_audit['validation_checks'].values())
expected_hashes = json.loads((OUT / 'FILE_SHA256.json').read_text())
actual_hashes = {str(path.relative_to(OUT)): sha256(path) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and path.name != 'FILE_SHA256.json'}
assert actual_hashes == expected_hashes
print('Read-back validation: PASS')
print('CSV artifacts:', len(list((OUT / 'tables').glob('*.csv'))))
print('Figures:', len(list((OUT / 'figures').glob('*.png'))))
display(pd.read_csv(OUT / 'tables' / 'fold_metrics.csv').head())
display(pd.read_csv(OUT / 'tables' / 'stability_summary.csv'))
for path in sorted((OUT / 'figures').glob('*.png')):
    print(path.name)
    display(Image(filename=str(path)))"""
    boundary = """## Interpretation boundary

This notebook provides model-specific executed evidence. Seven-way accuracy rankings and pairwise significance statements are valid only after the separate common-origin synthesis. Stability describes consecutive-fold explanation agreement, not causal stability of the economy. Faithfulness diagnostics test specific properties of the fitted predictor; none converts a lagged-price association into evidence of a genuine causal economic driver."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update({'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                              'language_info': {'name': 'python', 'version': '3.12'}})
    notebook.cells = [nbformat.v4.new_markdown_cell(introduction),
                      nbformat.v4.new_markdown_cell(corrections),
                      nbformat.v4.new_code_cell(model_core),
                      nbformat.v4.new_code_cell(data_cell),
                      nbformat.v4.new_markdown_cell('## Forecast and explanation execution\n\nAll planned units are run below. The notebook stops on an incomplete or invalid unit rather than silently omitting it.'),
                      nbformat.v4.new_code_cell(specification['runner']),
                      nbformat.v4.new_markdown_cell(diagnostic_md),
                      nbformat.v4.new_code_cell(specification['export']),
                      nbformat.v4.new_code_cell(readback),
                      nbformat.v4.new_markdown_cell(boundary)]
    path = N / specification['filename']
    nbformat.write(notebook, path)
    compile(model_core, str(path) + ':core', 'exec')
    compile(specification['runner'], str(path) + ':runner', 'exec')
    compile(specification['export'], str(path) + ':export', 'exec')
    print(path, 'cells=', len(notebook.cells))
