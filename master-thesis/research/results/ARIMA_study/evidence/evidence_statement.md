# Executed evidence statement — ARIMA

Execution version: `thesis-execution-v1`

This notebook was executed in the recorded environment. Cross-model conclusions are defined only after matched-origin synthesis.

## RQ1
Status: model_execution_complete_cross_model_answer_in_synthesis

- Native-coverage fold metrics are exported with observation-level predictions; seven-way claims use only matched origins in the cross-model synthesis.
  - Value: `{'dax|h1': 0.011796400706931763, 'dax|h5': 0.026104778768317682, 'djia|h1': 0.010837941843617969, 'djia|h5': 0.023050819316431687, 'monash_sunspot|h1': 10.626520710083117, 'monash_sunspot|h5': 32.19973930218652, 'nasdaq|h1': 0.012876913501736862, 'nasdaq|h5': 0.02828114844710387, 'sp500|h1': 0.010295704267316103, 'sp500|h5': 0.021476971943867103}`
  - Artifact: `tables/fold_metrics.csv and tables/observation_predictions.csv`

## RQ2
Status: model_execution_complete_cross_model_answer_in_synthesis

- Consecutive-fold attributions use identical, contiguous, out-of-sample probe origins; Spearman, cosine, and top-feature agreement are exported.
  - Value: `{'ARIMA_native_counterfactual_lag|bear': 1.0, 'ARIMA_native_counterfactual_lag|boundary': 1.0, 'ARIMA_native_counterfactual_lag|bull': 1.0, 'ARIMA_native_counterfactual_lag|crisis': 1.0, 'ARIMA_native_counterfactual_lag|na': 1.0}`
  - Artifact: `tables/stability.csv`

## RQ3
Status: model_execution_complete_economic_identification_bounded

- Faithfulness is evaluated diagnostic by diagnostic; no feature ranking is interpreted as a causal economic driver.
  - Value: `{'max_shap_additivity_residual': 2.8421709430404007e-13}`
  - Artifact: `tables/faithfulness.csv`

## Limitations

- Only lagged prices or native Sunspot values are available; causal economic-driver identification is not possible.
- Regime labels are ex post data-driven strata, not independently dated historical crises.
- Zero-baseline interventions and synthetic local-surrogate samples can leave the natural data manifold.
- Model-specific limitations are documented in the thesis.